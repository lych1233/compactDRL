## Soft Actor-Critic (SAC)

### Quick Start

Use
```
bash scripts/sac/Pendulum.sh 0 0
bash scripts/sac/Hopper.sh 0 0
bash scripts/sac/Huanmoid.sh 0 0
```
for a quick start

To evaluate existing models, just add "--test_model --load_file 'local path to your model'" at the end of the training script




### Commands and Tips for SAC

Here is an example command to train a sac agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario mujoco --env Hopper-v3 \
    --algo sac \
    --num_T 3000000 --test_interval 20000 \
    --lr 3e-4 --batch_size 128 \ # Learning rates are of the most important
    --alpha 0.2 # the temprature coefficient for the maximum entropy objective function
```




### Some Implementation Lessons

- Currently I do not know any other strategies to solve MountainCarContinuous-v0 by SAC without strong Ornstein Uhlenbeck noise or state-dependent exploration. On some random seeds the agent can explore the goal successfully without OU-noise but it apperantly to be some random events. Currently we do not find a good hyperparameter configuration such that SAC continuously solves MountainCarContinous-v0
- On mujoco SAC is not that sensitive to the hyperparameters and the performance is stable on different hyperparameters
