## Deep Q-Network (DQN)

### Quick Start

Use
```
bash scripts/dqn/CartPole.sh 0 0
bash scripts/dqn/pong.sh 0 0
bash scripts/dqn/qbert.sh 0 0
```
for a quick start

To evaluate existing models, just add "--test_model --load_file 'local path to your model'" at the end of the training script




### Commands and Tips for DQN

Here is an example command to train a dqn agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario atari --env pong \
    --algo dqn \
    --num_T 3000000 --test_interval 20000 \
    --eps 0.05 \ # use epsilon-greedy exploration; 0.05 is usually a default option
    --update_frequency 4 \ # skip somes steps between two model updating processes
    --lr 6e-5 --batch_size 32 --hidden_dim 512 # this is a safe configuration; dqn is sensitive to the learning rate, but a small learning rate is usually a reliable starting point
```




### Warning for Using DQN

- Vanilla DQN is usually the first algorithm one will meet in deep reinforcement learning, which is simple and efficient; however, it is no longer the SOTA and there is a lot of improvement of DQN, like Double-Dueling DQN, or later Rainbow
- Since DQN does not use the double Q-network trick, it will overestimate the Q value (sometimes million or billion large), especially in environments with only positive rewards like "CartPole-v1"
