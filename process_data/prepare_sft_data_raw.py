import datasets
import numpy as np
from openai import AsyncClient


CRITIQUE_SYSTEM_MESSAGE = """Critique the given solution based on the problem and correct answer. Identify any errors, compare it to the correct answer."""

def critique_solution(rollout, problem, answer):
    # critique the solution
    pass


def merge_datasets(dataset_list):
    ds_list = [datasets.load_dataset(dataset_name, split="train") for dataset_name in dataset_list]    
    return datasets.concatenate_datasets(ds_list)


positive_feedbacks  = [
    "This answer is correct!",    
    "Looks good, this answer is correct!",
    "This is the correct answer!",
    "Let's verify the answer, this is correct!",
]
negative_feedbacks = [
    "This answer is incorrect, let me try again.",
    "Wait, this is not correct, let me try again.",
    "This is not the correct answer, let me try again.",
    "Something is wrong, this is not the correct answer.",
]


def create_trajectory_data(sample, max_iter):
    rollouts = sample["tir_seed_rollouts"]
    is_correct = sample["is_correct"]    
    inds = np.arange(len(rollouts))
    chosen_inds = np.random.choice(inds, max_iter, replace=False)
    
    # generate trajectory with critique     
    trajectory = []
    for i, ind in enumerate(chosen_inds):
        correct = is_correct[ind]
        rollout = rollouts[ind]        
        if i == max_iter - 1:
            if correct:
                trajectory.append(rollout)
                feedback = np.random.choice(positive_feedbacks)
                trajectory.append(feedback)
            else:    # if incorrect at the last iteration, find a correct one
                correct_inds = [item[0] for item in np.where(is_correct)]
                correct_ind = np.random.choice(correct_inds, 1)[0]
                trajectory.append(rollouts[correct_ind])
                feedback = np.random.choice(positive_feedbacks)
                trajectory.append(feedback)
        else:
            if correct:    # positive feedback
                trajectory.append(rollout)
                feedback = np.random.choice(positive_feedbacks)
                trajectory.append(feedback)
            else:    # negative feedback
                trajectory.append(rollout)
                feedback = np.random.choice(negative_feedbacks)
                trajectory.append(feedback)

        if correct:
            break
    
    sample["trajectory"] = trajectory    
    sample["num_attempts"] = len(trajectory) // 2
    # construct full trajectory
    full_trajectory = "<think>"
    for i, traj in enumerate(trajectory):
        full_trajectory += f" {traj.strip()}\n\n"
    full_trajectory += " </think>"
    full_trajectory += f"\n\n<solution> {trajectory[-2].strip()} </solution>"
    sample["full_trajectory"] = full_trajectory
    return sample
    


def main():
    max_length = 8000
    dataset_list = ["RLAIF/CS-PRM-Seed-Rollouts-10K", "RLAIF/CS-PRM-Seed-Rollouts-20K"]
    ds = merge_datasets(dataset_list)
    ds = ds.filter(lambda x: any(x["is_correct"]) and not all(x["is_correct"]))
    num_unique_problems = len(ds.unique("problem"))
    print(f"A total of {num_unique_problems} samples are in the dataset")
    max_iter = 3
    ds = ds.map(create_trajectory_data, num_proc=1, fn_kwargs={"max_iter": max_iter})
    ds = ds.filter(lambda x: len(x["full_trajectory"]) <= max_length)    
    ds.push_to_hub(f"RLAIF/Tool-SFT-iter{max_iter}-Raw", private=True)


if __name__ == "__main__":
    main()