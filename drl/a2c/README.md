### Quick Start

Use
```
bash scripts/a2c/CartPole.sh 0 0
bash scripts/a2c/Hopper.sh 0 0
bash scripts/a2c/pong.sh 0 0
```
for a quick start




### Commands and Tips for PPO

Here is an example command to train a a2c agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario mujoco --env Hopper-v3 \
    --algo a2c \
    --num_T 10000000 --test_interval 20000 \
    --sample_steps 256 --num_env 16 --num_minibatch 1 \ # Use the full batch is safe; always use mutiple envs
    --buffer_capacity 256 \ # make it at least the same as sample_steps
    --lr 3e-4 --lam 0.95
```




### Some Implementation Lessons

- See the tips for ppo (in drl/ppo/README.md)
- Use small batch_size for sample efficiency and mutiple paralleled environments (though this implies short chunck length)
