### Quick Start

Use
```
bash scripts/ppo/CartPole.sh 0 0
bash scripts/ppo/Hopper.sh 0 0
bash scripts/ppo/pong.sh 0 0
```
for a quick start




### Commands and Tips for PPO

Here is an example command to train a ppo agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario control --env CartPole-v1 \
    --algo ppo \
    --num_T 1000000 --test_interval 10000 \
    --sample_steps 1024 --num_env 32 --num_minibatch 4 \
    --buffer_capacity 1024 \ # make it at least the same as sample_steps
    --lr 1e-4 --clip_ratio 0.2 --lam 0.95
```




### Some Implementation Lessons

- Observation normalization is almost always harmless for vectorized input and it is often helpful for better performance, and it is especially critial for some environments like "MountainCar-v0"
- Reward/Value normalization helps fast fitting, but one should keep an eye on the its potential harm to the performance (although usually it's harmless)
- Always use **tanh** instead of **ReLU** for MLP! Otherwise you may fail in "MountainCar-v0". Plus, it is also benenicial as shown by [a study of GoogleBrain](https://arxiv.org/abs/2006.05990)
- Share the CNN feature extractor for the actor and the critic, or you will fail in "pong" unless a extremely tiny learning rate is used. Sharing CNN accelarates training a lot
