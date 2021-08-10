## *(Let's Get Started!)* Easy-to-read Deep Reinforcement Learning Algorithms in a Compact Implementation

### Two goal of this implementation:

- Provide easy to read/learn code, avoiding nesting over nesting
- The code compactly presents the core of those algorithms



### Quick Facts you should know about this implementation:

- All the core codes are presented in the /drl/"algo name"/ folder, which might be friendly for the reader
- Trick usage is as minimized as possible
- To run an experiment only two additional files are needed: an environment file (which can be regarded as a lightly extended OpenAI Gym API) and an buffer file (which is used **solely** for storage)
- You chould use commands in the /scripts/ folder to start an experiment directly
- In an experiment, "run.py" will first find suitable "env" and "buffer" for the given environment. Then "run.py" passes them to "algo/main" and "algo/main" will use the "env" and "buffer" and all local utils to finish the experiment
- **Every** specific configuration (hyper-parameter) has a description. The basic configuration explanation in run.py; the environment(buffer) configuration explanation is in envs/"env name".py (buffers/base.py); the algorithm-specific hyper-parameter explanation is in drl/"algo name"/main.py



### Quick Start

- To start an example experiment, say, use PPO to train an agent in the CartPole-v1 environment, you can direct use:

```
cd compactDRL
bash scripts/ppo/CartPole.sh 0 0
```



### Structure

├── drl   // Different algorithms
        ├── ppo
       		 ├── ppo stuffs
      		  └── main.py
├── envs   // Environments
    	├── control.py
   	 ├── atari.py
   	 └── base.py
├── buffers   // (purely for storage) Buffers
   	 ├── normal_buffer.py
  	  ├── image_buffer.py
	    └── base.py
├── scripts   // Running scripts
├── results   // Contain potential future experiment results
└── run.py   // Get Started Here!!!

