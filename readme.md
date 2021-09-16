# *(To Be Continued ...)* Easy-to-read Deep Reinforcement Learning Algorithms in a Compact Implementation


## The Goal of This Implementation:

- Provide easy to read/learn code, avoiding nesting over nesting
- The code compactly presents the core of those algorithms



## Quick Facts you should know about this implementation:

- All the core codes are presented in the drl/"algo name"/ folder, which might be friendly for the reader
- You could use commands in the scripts/ folder to start an experiment directly
- To run an experiment only two additional files are needed: an environment file (which can be regarded as a lightly extended OpenAI Gym API) and a buffer file (which is used **solely** for storage), and all other works can be done by algorithm local files
- In an experiment, "run.py" will first find suitable "env" and "buffer" for the specific configuration. Then "run.py" passes them to "algo/main.py" and "algo/main.py" will use that "env" and "buffer" and all local files to finish the experiment



## Project Feature (Pros & Cons)

#### Spotlight


- Implementation trick usage is pruned, while keeping those tricks that significantly affect the performance
- Elaborate documentation. **Every** specific configuration (hyper-parameter) has a description: the basic configuration explanation is in run.py; the environment (buffer) configuration explanation is in envs/"env name".py (buffers/base.py); the algorithm-specific hyper-parameter explanation is in drl/"algo name"/config.py

#### Limitation


- Some tricks are not implemented for the sake of simplicity and clarity, and thus the performance may be worse than the best implementation
- Hyperparameters are not tuned carefully, but they are set safely and areuniversal to similar environments
- Currently it only supports environments with either simple discrete action space of one-dimensional vectorized continuous action space
- RNN-based models are not suportted to make an elegant implementation (or maybe due to my limited coding skills)


## Run

#### Quick Start (An Running Script Example)


- To start an example experiment, say, use PPO to train an agent in the "CartPole-v1'' environment, you can direct use:

```
cd compactDRL
bash scripts/ppo/CartPole.sh 0 1
```

where "0" is the PGU-id, and "1" is the seed of the entire experiment

#### Arg/Config One Should Notice for A Complete Experiment

TODO



## Structure

    ├── drl   // Different algorithms
        ├── vpg // Use vpg for illustration, others follow exactly the same structure
            ├── main.py // Basic controller
            ├── agnet.py // An ppo agent including both deciding part and learning part
            ├── config.py // Complete configuration and hyperparameter
            ├── logger.py // A simple logger for w&b
            └── other vpg stuffs
        ├── ppo
        └── sac
    ├── envs   // Environments
        ├── control.py
        ├── atari.py
        └── base.py
    ├── buffers   // (purely for storage) Buffers 
        ├── normal_buffer.py
        ├── image_buffer.py
        └── base.py 
    ├── scripts
    ├── results
    └── run.py

