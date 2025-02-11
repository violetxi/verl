import sys
import hydra
import asyncio
import logging
import datasets
import sglang as sgl

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from omegaconf import OmegaConf
from code_execution import PythonREPL
from math_evaluator import MathEvaluator


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.math_evaluator = MathEvaluator()
        self.ds = self._load_dataset(config)
        self.llm = self._load_model(config)
        self.event_loop = asyncio.new_event_loop()
        self.executor = PythonREPL()
        self.step_token = " \n\n"
        self.evaluator = MathEvaluator()

    def _load_dataset(self, config):
        ds = datasets.load_dataset(config.dataset_id, split=config.split)        
        return ds.select(range(config.start_idx, config.end_idx))

    def _load_model(self, config):
        if config.sampling_params.temperature == 0:
            assert config.n_samples == 1 and config.sampling_params.n == 1, \
                "When temperature is 0, n_samples and n must be 1."
            
        self.sampling_params = OmegaConf.to_container(config.sampling_params)
        return sgl.Engine(model_path=config.model_id, dp_size=config.dp_size)
    

    def execute_completion(self, completion, return_status=True, last_code_block=False):
        import re
        executions = re.findall(r"```python(.*?)```", completion, re.DOTALL)
        # no code to execute
        if len(executions) == 0:
            return completion, False
        
        if last_code_block:
            executions = [executions[-1]]

        outputs = []
        successes = []
        for code in executions:
            success = False
            for lib in ("subprocess", "venv"):
                if lib in code:
                    output = f"{lib} is not allowed"
                    outputs.append(output)
                    successes.append(success)
                    continue
            try:
                success, output = self.executor(code)
            except TimeoutError as e:
                print("Code timed out", file=sys.stderr)
                output = e

            if not success and not return_status:
                output = ""
            outputs.append(output)
            successes.append(success)
        
        output = str(outputs[-1]).strip()
        success = successes[-1]
        if return_status:
            return output, True
        
        return output, True

    def run_one_sample(self, sample):
        prompt_template = "Question: {problem}\n\nSolution: "
        samples = []
        for _ in range(self.config.n_samples):
            sample_dict = {k : v for k, v in sample.items()}
            sample_dict["done"] = False
            sample_dict["prompt"] = prompt_template.format(problem=sample["problem"])
            sample_dict["is_correct"] = None
            sample_dict["model_solution"] = ""
            sample_dict["interact_num"] = 0
            samples.append(sample_dict)

        samples = Dataset.from_list(samples).to_pandas()
        for n in range(self.config.n_interactions):            
            dones = samples["done"].tolist()
            not_done_inds = [i for i, done in enumerate(dones) if not done]
            prompts = [samples.at[i, "prompt"] for i in not_done_inds]
            outputs = self.llm.generate(prompts, self.sampling_params)
            outputs = [output['text'] for output in outputs]

            for i, output in enumerate(outputs):
                idx = not_done_inds[i]
                answer = samples.at[idx, "answer"]
                if "boxed{" in output and "</solution>" in output:
                    samples.at[idx, "done"] = True
                    samples.at[idx, "is_correct"] = self.event_loop.run_until_complete(
                        self.evaluator.is_correct(correct_answer=answer, proposed_answer=output, use_judge=False)
                        )
                    samples.at[idx, "interact_num"] = n
                else:
                    code_result, executed = self.execute_completion(output, return_status=True, last_code_block=False)
                    truncation_limit = 200
                    if executed and len(code_result) > truncation_limit:
                        code_result = code_result[:truncation_limit] + " ... (output truncated)"

                    # only need to update output if code was executed
                    if executed:
                        assert "```python" in output, "```python ``` block not in the output"
                        output = output.strip() + f"\n```output {code_result} ```{self.step_token}"
                
                samples.at[idx, "model_solution"] = samples.at[idx, "model_solution"] + output
                samples.at[idx, "prompt"] = samples.at[idx, "prompt"] + output
                if n == self.config.n_interactions - 1:                    
                    samples.at[idx, "is_correct"] = False
                    samples.at[idx, "interact_num"] = n

        return Dataset.from_pandas(samples)
                

    def run(self):
        results = []
        for sample in tqdm(self.ds):
            results.append(self.run_one_sample(sample))

        ds = concatenate_datasets(results)
        model_name = self.config.model_id.split("/")[-1]
        dataset_name = self.config.dataset_id.split("/")[-1]
        upload_id = f"RLAIF/eval-{model_name}-{dataset_name}"
        ds.push_to_hub(upload_id, private=True)
        logging.info(f"Results uploaded to {upload_id}")
        breakpoint()



@hydra.main(version_base=None, config_path="conf", config_name="evaluate")
def main(config):
    evaluator = Evaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()