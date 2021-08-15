# *(To Be Continued ...)* Easy-to-read Deep Reinforcement Learning Algorithms in a Compact Implementation


## Two goal of this implementation:

- Provide easy to read/learn code, avoiding nesting over nesting
- The code compactly presents the core of those algorithms



## Quick Facts you should know about this implementation:

- All the core codes are presented in the /drl/"algo name"/ folder, which might be friendly for the reader
- You chould use commands in the /scripts/ folder to start an experiment directly
- To run an experiment only two additional files are needed: an environment file (which can be regarded as a lightly extended OpenAI Gym API) and an buffer file (which is used **solely** for storage)
- In an experiment, "run.py" will first find suitable "env" and "buffer" for the given environment. Then "run.py" passes them to "algo/main" and "algo/main" will use the "env" and "buffer" and all local utils to finish the experiment



## Project Feature (Pros & Cons)

#### Spotlight


- Trick usage is as minimized as possible
- Elaborate documentation. **Every** specific configuration (hyper-parameter) has a description: the basic configuration explanation in run.py; the environment(buffer) configuration explanation is in envs/"env name".py (buffers/base.py); the algorithm-specific hyper-parameter explanation is in drl/"algo name"/main.py

#### Limitation


- Some tricks are not implemented for the sake of simplicity and clarity, and thus the performance may be worse than the best implementation



## Run

#### Quick Start (An Running Script Example)


- To start an example experiment, say, use PPO to train an agent in the "CartPole-v1'' environment, you can direct use:

```
cd compactDRL
bash scripts/ppo/CartPole.sh 0 1
```

where 0 is the gpu id and 1 is the seed of the entire experiment


## Structure

    ├── drl   // Different algorithms             
        ├── ppo
            ├── other ppo stuffs
            └── main.py
        ├── policy_gradient_agents
        └── stochastic_policy_search_agents 
    ├── envs   // Environments
        ├── control.py // Environments
        ├── atari.py
        └── base.py
    ├── buffers   // (purely for storage) Buffers 
        ├── normal_buffer.py
        ├── image_buffer.py
        └── base.py 
    ├── scripts
    ├── results
    └── run.py

