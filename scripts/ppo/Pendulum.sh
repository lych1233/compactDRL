export OMP_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --env_type control --env Pendulum-v0 \
    --algo ppo \
    --num_T 1000000 --sample_steps 4096 --test_interval 10000 \
    --lam 0.97 --buffer_capacity 4096
