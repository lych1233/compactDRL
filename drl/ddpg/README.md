### Quick Start

Use
```
bash scripts/ddpg/Pendulum.sh 0 0
bash scripts/ddpg/Hopper.sh 0 0
bash scripts/ddpg/Huanmoid.sh 0 0
```
for a quick start

To evaluate existing models, just add "--test_model --load_file 'local path to your model'" at the end of the training script




### Commands and Tips for DDPG

Here is an example command to train a ddpg agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario mujoco --env Hopper-v3 \
    --algo ddpg \
    --num_T 3000000 --test_interval 20000 \
    --actor_lr 3e-4 --critic_lr 3e-4 --batch_size 128 \ # Learning rates are of the most important
    --online_noise_scale 0.1 # Noise for exploration and target critic value smoothing; default setting works well; you may use --OU_noise for a better exploration in certain environments
```




### Warning for Using DDPG

- DDPG is a little bit out-dated and is not the state of the art for single agent environments (though MADDPG is still used for MARL or many situations), and it has a serious problem on overestimating the Q-value which often leads to a terrible result
- It is better to use TD3 as an alternative, as TD3 is an improved version of DDPG which alleviates the problem of Q-value overestimation