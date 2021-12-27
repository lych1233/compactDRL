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
        name="seed_" + str(args.seed) + "_rainbow_" + args.env,
        dir=save_dir,
    )
    work_dir = os.getcwd()
    for path in pathlib.Path(work_dir).rglob("*.py"):
        file = str(path)
        if "wandb" not in file:
            wandb.save(file, base_path=work_dir)

def wandb_finish():
    wandb.run.finish()

def log(dir, T, stats):
    for k, v in stats.items():
        wandb.log({dir + "/" + k: v}, T)