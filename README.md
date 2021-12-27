# *(To Be Continued ...)* Easy-to-read Deep Reinforcement Learning Algorithms in a Compact Implementation

| Algorithms |  |
| :--------: | :---------------: |
| dqn | :x: (coming soon) |
| a2c | :heavy_check_mark: |
| ddpg | :heavy_check_mark: |
| ppo | :heavy_check_mark: |
| rainbow | :x: |
| td3 | :heavy_check_mark: |
| sac | :heavy_check_mark: |

The hyperparameters are not extensively fine-tuned, but we garuantee an acceptable performance with a safe setting of hyperparameters.



## Performance


Different algorithms have varying sample efficiency and training speed, especially between on-policy algorithms and some off-policy algorithms. Therefore we adopt different training steps for different algorithms. [Full Results](pics)


### Pong (Atari)


![pong](pics/pong.png)



### Hopper (MuJoCo)


![Hopper](pics/Hopper.png)



## The Goal of This Implementation:

- Provide easy to read/learn/follow codes, avoid nesting over nesting
- The code compactly presents the core of those algorithms



## Run

#### Requirements



All basic required packages are those commonly used in Deep Reinforcement Leaninrg:

- torch
- gym
- tqdm

but the results may slightly differ from what we've shown as packages in different versions are used during our training

> Additional packages are needed for scenarios like Atari, MuJoCo or your custom environments.

#### Quick Start (An Running Script Example)


- To start an example experiment, say, use PPO to train an agent in the "CartPole-v1'' environment, you can direct use:

```
cd compactDRL
bash scripts/ppo/CartPole.sh 0 1
```

where "0" is the PGU-id, and "1" is the seed of the entire experiment

See README in each drl/algo folder for some tips for a specific algorithm



## Quick Facts you should know about this implementation:


- All the core codes are presented in the drl/"algo name"/ folder, which might be friendly for the reader
- You could use commands in the scripts/ folder to start an experiment directly
- To run an experiment only two additional files are needed: an environment file (which can be regarded as a lightly extended OpenAI Gym API) and a buffer file (which is used **solely** for storage), and all other works can be done by drl/"algo name"/ local files
- In an experiment, "run.py" will first find suitable "env" and "buffer" for the specific configuration. Then "run.py" passes them to "algo/main.py" and "algo/main.py" will use that "env" and "buffer" and all local files to finish the experiment



## Project Feature (Pros & Cons)

#### Spotlight


- Implementation trick usage is pruned, while those tricks that significantly affect the performance are kept
- Elaborate documentation. **Every** specific configuration (hyper-parameter) has a description: the basic configuration explanation is in run.py; the environment (buffer) configuration explanation is in envs/"env name".py (buffers/base.py); the algorithm-specific hyper-parameter explanation is in drl/"algo name"/config.py
- Acceptable performance on the most common benchmark environments in academy: Atari (for discrete action space) and MuJoCo (for continuous actoin space), as other elegant implementation I've found are seldom tested on both of them

#### Limitation


- Some tricks are not implemented for the sake of simplicity and clarity, and thus the performance may be worse than the best implementation
- Hyperparameters are not tuned vary carefully, but we ensure the performance is closed to the best baseline
- Currently it only supports environments with either simple discrete action space or one-dimensional vectorized continuous action space
- RNN-based models are not suportted for an elegant implementation (becasue of my limited coding skills to make it simple)


## Structure

    ├── drl   // Different algorithms
        ├── a2c // Use a2c for illustration, others follow exactly the same structure
            ├── main.py // Basic controller
            ├── agent.py // An a2c agent including both decision making and learning
            ├── config.py // Complete configuration with all hyperparameters
            ├── logger.py // A simple logger for w&b
            └── other a2c stuffs (usually there is a nn_blocks.py for building basic neural networks)
        ├── ddpg
        ├── ppo
        ├── td3
        └── sac
    ├── envs   // Environments
        ├── control.py
        ├── atari.py
        ├── mujoco.py
        └── base.py
    ├── buffers   // (purely for storage) Buffers 
        ├── normal_buffer.py
        ├── image_buffer.py
        ├── vector_buffer
        └── base.py 
    ├── scripts
    ├── results
    └── run.py




## Refenrences

- The best deep reinforcement learning turorial I've found: [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
- Spinning Up's easy-to-follow code: [Spinning Up Code](https://github.com/openai/spinningup)
- The baseline of the highest quality in pytorch that I've found: [Stable-Baselines 3](https://github.com/DLR-RM/stable-baselines3)
