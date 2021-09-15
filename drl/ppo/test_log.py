import wandb
import math
import random

def test_func():
    wandb.log({"test_fun": 1})

experiment_index = 5
for i in range(5):
  job_type = "rollout"
  if i == 1: break
  if i == 2:
    job_type = "eval"
  if i == 3:
    job_type = "eval2"
  if i == 4:
    job_type = "optimizer"

  # Set group and job_type to see auto-grouping in the UI
  wandb.init(project="group-demo", 
             group="exp_" + str(experiment_index), 
             job_type=job_type,
             name="do_it_{}".format(experiment_index))

  for j in range(100):
    acc = 0.1 * (math.log(1 + j + .1) + random.random())
    val_acc = 0.1 * (math.log(1+ j + 2) + random.random() + random.random())
    if (j % 10 == 0):
      wandb.log({"acc":acc, "val_acc":val_acc})
  
  # Using this to mark a run complete in a notebook context
  wandb.finish()

# I'm incrementing this so you can re-run this cell and get another experiment
# grouped in the W&B UI
experiment_index += 1