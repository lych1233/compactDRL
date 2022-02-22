import argparse


def get_args():
    """Rainbow (a combination of tricks of DQN improvement)
    One can also use Dueling Double DQN by setting "--enhancement double dueling"
    
    core hyper-parameters for general DQN
    ---------------------------------------------------------------------------
    lr                       |   the learning rate of deep Q-network
    batch_size               |   the size of each sampled minibatch
    gamma                    |   discounting factor γ for future reward accumulation
    max_grad_norm            |   clip the gradient for stable training
    hidden_dim               |   number of hidden nodes per mlp layer
    eps                      |   use eps-greedy exploration
    num_T                    |   number of interaction steps to train an agent (may affect the learning rate decay
    start_learning           |   number of steps before updating polices
    update_frequency         |   number of steps between two model updating process
    target_update_interval   |   number of model updating steps between cloing the online network to the target network
    
    hyper-parameters for dqn improvement tricks in rainbow
    ---------------------------------------------------------------------------
    enhancement   |  there are overall six types of improvement of DQN that is used in Rainbow
                     double, dueling, distributional, noisy_net, multi_step, prioritized_replay
    For distributional RL
        atoms            |   number of supporting points in Q-value distribution
        minV             |    the samllest support point
        maxV             |   the largest support point
    For noisy net
        noise_std        |   epsilon std set in the noisy net
    For multi-step bootstrap learning
        n_steps          |   number of steps for bootstraping in the Bellman Equation
    For prioritized replay
        priority_alpha   |   transition sampling probability proportional to δ^α
        priority_beta    |   importance sampling weight w^β where β starts from β_0 (0.4) to 1 to compensate prioritzied sampling for unbiased Q-value equation near convergence

    More parameters for record, test and other stuffs
    ---------------------------------------------------------------------------
    load_file    |   provide files storing pretrained models, or the training will be from scratch
    save_dir     |   the folder where models and training statictis will be saved
    render       |   render the enviroment during test time
    test_model   |   purely evaluate an agent without any training
    test_times   |   number of episodes to do a test
    test_interval         |   number of epochs between two tests
    checkpoint_interval   |   number of interaction steps to save a model backup when it > 0

    And there are some additional hyper-parameters specific for CNN architecture defined in
    the argument_complement function
    ---------------------------------------------------------------------------
    channel_dim       |   basic number of channels in each CNN layer
    channel_divider   |   reduce channels by a divider in each layer from top to bottom
    kernel_size       |   a list of kernel sizes in each layer from top to bottom
    stride            |   a list of stride values in each layer from top to bottom

    Advanced training statistics visualization using wandb
    ---------------------------------------------------------------------------
    wandb_show       |   statistics visualization by plotting the curves of certain statistic target
    wandb_user       |   username of your wandb account
    wandb_project    |   the project of the experiment on wandb
    wandb_group      |   the group of the experiment on wandb, should be env_tpye/env if not specified
    wandb_job_type   |   the job_type of the experiment on wandb
    """
    parser = argparse.ArgumentParser(description="rainbow parser")

    parser.add_argument("--lr", default=3e-4, type=float, \
        help="the learning rate of deep Q-network")
    parser.add_argument("--batch_size", default=32, type=int, \
        help="the size of each sampled minibatch")
    parser.add_argument("--gamma", default=0.99, type=float, \
        help="discounting factor γ for future reward accumulation")
    parser.add_argument("--max_grad_norm", default=10, type=float, \
        help="clip the gradient for stable training")
    parser.add_argument("--hidden_dim", default=256, type=int, \
        help="number of hidden nodes per mlp layer")
    parser.add_argument("--eps", default=0.05, type=float, \
        help="use eps-greedy exploration")
    parser.add_argument("--num_T", default=10000000, type=int, \
        help="number of interaction steps to train an agent (may affect the learning rate decay)")
    parser.add_argument("--start_learning", default=20000, type=int, \
        help="number of steps before updating polices")
    parser.add_argument("--update_frequency", default=1, type=int, \
        help="number of steps between two model updating process")
    parser.add_argument("--target_update_interval", default=2000, type=int, \
        help="number of model updating steps between cloing the online network to the target network")
    
    print("\n---------- general dqn core hyperparameters ----------")
    for key, val in vars(parser.parse_known_args()[0]).items():
        print("{:>25}   |   {}".format(key, val))
    print("------------------------------------------------------\n")
    
    # Now hyperparameters for a bunch of enhancements
    parser.add_argument("--enhancement", default=["double", "dueling", "distributional", "noisy_net", "multi_step", "prioritized_replay"], type=str, nargs="+", \
        choices=["double", "dueling", "distributional", "noisy_net", "multi_step", "prioritized_replay", "none"], \
        help="there are overall six types of improvement of DQN that is used in Rainbow")
    # For distributional RL
    parser.add_argument("--atoms", type=int, default=51, help="number of supporting points in Q-value distribution")
    parser.add_argument("--minV", type=float, default=-10, help="the samllest support point")
    parser.add_argument("--maxV", type=float, default=10, help="the largest support point")
    # For noisy net
    parser.add_argument("--noise_std", type=float, default=0.5, help="epsilon std set in the noisy net")
    # For multi-step bootstrap learning
    parser.add_argument("--n_steps", type=int, default=3, help="number of steps for bootstraping in the Bellman Equation")
    # For prioritized replay
    parser.add_argument("--priority_alpha", type=float, default=0.5, help="transition sampling probability proportional to δ^α")
    parser.add_argument("--priority_beta", type=float, default=0.4, help="importance sampling weight w^β where β starts from β_0 (0.4) to 1 to compensate prioritzied sampling for unbiased Q-value equation near convergence")

    print("\n---------- rainbow enhancement hyperparameters ----------")
    start_printing = False
    for key, val in vars(parser.parse_known_args()[0]).items():
        if key == "enhancement":
            start_printing = True
        if start_printing:
            print("{:>25}   |   {}".format(key, val))
    print("---------------------------------------------------------\n")

    # Other necessary parameters for a complete experiment
    parser.add_argument("--load_file", default=None, type=str, \
        help="provide files storing pretrained models, or the training will be from scratch")
    parser.add_argument("--save_dir", default="results/", type=str, \
        help="the folder where models and training statictis will be saved")
    parser.add_argument("--render", default=False, action="store_true", \
        help="render the enviroment during test time")
    parser.add_argument("--test_model", default=False, action="store_true", \
        help="purely evaluate an agent without any training")
    parser.add_argument("--test_times", default=10, type=int, \
        help="number of episodes to estimate the performance of the agent")
    parser.add_argument("--test_interval", default=10000, type=int, \
        help="number of epochs between two tests")
    parser.add_argument("--checkpoint_interval", default=-1, type=int, \
        help="number of interaction steps to save a model backup (-1 to disable)")

    argument_complement(parser)
    return parser.parse_args()

def argument_complement(parser):
    """The purpose of this function is to complete the argument,
    so that we can use a complete parser to check if there is any typo in the command line
    """
    # Base configuration
    parser.add_argument("--exp_name", default="unnamed", type=str, \
        help="the name of the experiment; the result will be saved at results/exp_name by default")
    parser.add_argument("--seed", default=0, type=int, \
        help="random seed of the whole experiment under which the result should be the same")
    parser.add_argument("--scenario", default="control", choices=["control", "atari", "mujoco"], \
        help="the type/background of the environment")
    parser.add_argument("--env", default="CartPole-v1", type=str, \
        help="environment to interact with")
    parser.add_argument("--num_env", default=1, type=int, \
        help="number of parallel environments")
    parser.add_argument("--algo", default="dqn", \
        choices=["dqn", "a2c", "ddpg", "rainbow", "ppo", "td3", "sac"], \
        help="deep learning algorithm to choose")
    parser.add_argument("--disable_cuda", default=False, action="store_true", \
        help="cpu training even when gpus are available")
    
    # Atari ALE configuration
    parser.add_argument("--screen_size", type=int, default=84, \
        help="clip the image into L*L squares")
    parser.add_argument("--sliding_window", type=int, default=4, \
        help="keep a series of contiguous frames as one observation (represented as a S*L*L tensor)")
    parser.add_argument("--max_episode_length", type=int, default=100000, \
        help="the maximum steps in one episode to enforce an early stop in some case")
    
    # Buffer configuration
    parser.add_argument("--buffer_type", default="dequeue", choices=["dequeue", "reservoir"], \
        help="the way to kick out old data when the buffer is full")
    parser.add_argument("--buffer_capacity", default=1000000, type=int, \
        help="the maximum number of trainsitions the buffer can hold")
    
    # CNN architecture hyper-parameters
    parser.add_argument("--channel_dim", default=64, type=int, \
        help="basic number of channels in each CNN layer")
    parser.add_argument("--channel_divider", default=[2, 1, 1], type=int, nargs="+", \
        help="reduce channels by a divider in each layer from top to bottom")
    parser.add_argument("--kernel_size", default=[8, 4, 3], type=int, nargs="+", \
        help="a list of kernel sizes in each layer from top to bottom")
    parser.add_argument("--stride", default=[4, 2, 1], type=int, nargs="+", \
        help="a list of stride values in each layer from top to bottom")
    
    # Wandb setting for advanced users
    parser.add_argument("--wandb_show", default=False, action="store_true", \
        help="statistics visualization by plotting the curves of certain statistic target")
    parser.add_argument("--wandb_user", default=None, type=str, \
        help="username of your wandb account")
    parser.add_argument("--wandb_project", default="compactDRL", type=str, \
        help="the project of the experiment on wandb")
    parser.add_argument("--wandb_group", default=None, type=str, \
        help="the group of the experiment on wandb, should be env_tpye/env if not specified")
    parser.add_argument("--wandb_job_type", default="unknown agent", type=str, \
        help="the job_type of the experiment on wandb")
