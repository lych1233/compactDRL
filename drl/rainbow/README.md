### Quick Start

Use
```
bash scripts/rainbow/CartPole.sh 0 0
bash scripts/rainbow/pong.sh 0 0
bash scripts/rainbow/qbert.sh 0 0
```
for a quick start

To evaluate existing models, just add "--test_model --load_file 'local path to your model'" at the end of the training script




### Commands and Tips for Rainbow

Here is an example command to train a rainbow agent, containing those hyperparameters of the first prior to consider:

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario atari --env pong \
    --algo rainbow \
    --num_T 3000000 --test_interval 20000 \
    --update_frequency 4 \ # skip somes steps between two model updating processes
    --lr 6e-5 --batch_size 32 --hidden_dim 512 # this is a safe configuration; the learning rate seems to be the most important hyperparameters, while rainbow is usually robust on other hyperparameters
```

Please pay attention to the hyperparameter ``minV, maxV'' if you want to use the distributional parameterization of Q value. The range [minV, maxV] should contain all possible values of the Q value involved in training. For example, in the environment "MountainCar-v0" where the agent will be assigned reward -1 continuously before success, it is better to use
[-100, 0]
rather than use the default value of minV, maxV.



####


For convenient albation studies, our implementation allows fexible adding or removing of all six improvment modules in rainbow. For example, one may use 

```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario atari --env pong \
    --algo rainbow \
    --num_T 3000000 --test_interval 20000 \
    --update_frequency 4 \
    --lr 6e-5 --batch_size 32 --hidden_dim 512 \
    --enhancement double distributional noisy_net multi_step
```
if only "double distributional noisy_net multi_step" are supposed to be activated, and use
```
CUDA_VISIBLE_DEVICES=0 python run.py \
    --exp_name $exp_name --seed $seed \
    --scenario atari --env pong \
    --algo rainbow \
    --num_T 3000000 --test_interval 20000 \
    --update_frequency 4 \
    --lr 6e-5 --batch_size 32 --hidden_dim 512 \
    --enhancement none
```
to disable all improvement modules.

Here is a simple ablation result on three atari games.



### Warning for Using Rainbow

- Currently rainbow can be regarded as SOTA of model-free algorithms in the atari benchmark in the non-distributed training setting

- There is a good work of revisiting rainbow on a set of mini-atari games which provides some useful insight [rivisiting rainbow](https://psc-g.github.io/posts/research/rl/revisiting_rainbow/)