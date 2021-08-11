CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --env_type control --env CartPole-v1 \
    --algo ppo \
    --policy_lr 0.001 \
    --sample_steps 1024 --buffer_capacity 1024 \
    --hidden_dim 64
