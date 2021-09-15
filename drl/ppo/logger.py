import os
import pathlib

import wandb


def wandb_init(args, save_dir):
    """Define the experiment online location and files to upload
    """
    work_dir = os.getcwd()
    group_name = args.env_type + "/" + args.env if args.wandb_group is None else args.wandb_group
    wandb.init(
        config=args,
        project=args.wandb_project,
        group=group_name,
        job_type=args.wandb_job_type,
        name="seed_" + str(args.seed) + "_ppo_" + args.env,
        dir=save_dir,
    )
    work_dir = os.getcwd()
    for path in pathlib.Path(work_dir).rglob("*.py"):
        file = str(path)
        if "wandb" in file: continue
        wandb.save(file, base_path=work_dir)

def wandb_finish():
    wandb.run.finish()

def log(dir, T, stats):
    for k, v in stats.items():
        wandb.log({dir + "/" + k: v}, T)

if __name__ == "__main__":
    frequency = 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", default=frequency, type=int)
    parser.add_argument("--name", default="default/test0", type=str)
    args = parser.parse_args()
    wandb.init(project="test_wandb1", 
               group="test_code_new",
               job_type="ppo train",
               name="do_it_{}".format("struct"),
               dir="./test_results")
    import glob
    for path in glob.glob('*.py'):
        import os
        print(os.path.join(os.getcwd(), path))
        wandb.save(os.path.join(os.getcwd(), path), base_path="/home/liangyancheng")
    step = 1
    while True:
        import time
        time.sleep(1)
        import numpy as np
        wandb.log({"new_loss": {"actor": np.random.rand(), "critic": np.random.rand()}}, step)
        wandb.log({"reward/a1": np.random.rand()}, step)
        wandb.log({"actor_clip": np.random.rand(), "critic_clip": np.random.rand()}, step)
        step += frequency
        if step >= 100: break
    wandb.run.finish()