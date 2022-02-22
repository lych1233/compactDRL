## Twin Delayed Deep Deterministic Policy Gradient (TD3)

### Quick Start

Use
```
bash scripts/td3/Pendulum.sh 0 0
bash scripts/td3/Hopper.sh 0 0
bash scripts/td3/Huanmoid.sh 0 0
```
for a quick start

To evaluate existing models, just add "--test_model --load_file 'local path to your model'" at the end of the training script




### Commands and Tips for TD3

Here is an example command to train a td3 agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario mujoco --env Hopper-v3 \
    --algo td3 \
    --num_T 3000000 --test_interval 20000 \
    --actor_lr 3e-4 --critic_lr 3e-4 --batch_size 128 \ # Learning rates are of the most important
    --online_noise_scale 0.1 --target_noise_scale 0.2 --target_noise_clip 0.5 # Noise for exploration and target critic value smoothing; default setting works well; you may use --OU_noise for a better exploration in certain environments
```




### Some Implementation Lessons

- We do not use gradient clipping here but experiments show that extremely large Q-values still occur occasionally during training, so gradient clipping might be helpful for stable training
- Currently I do not know any other simple strategies to solve MountainCarContinuous-v0 by TD3 without strong Ornstein Uhlenbeck noise
- TD3 is sensitive to the hyperparameters so careful tuning for each environment respectively is essential for a good performance
