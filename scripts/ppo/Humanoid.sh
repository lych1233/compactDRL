CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario mujoco --env Humanoid-v3 \
    --algo ppo \
    --num_T 10000000 --test_interval 20000 --num_env 32 \
    --sample_steps 2048 --reuse_times 10 --num_minibatch 16 \
    --buffer_capacity 2048 \
    --lr 1e-4 --lam 0.95