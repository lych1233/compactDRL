CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env Pendulum-v0 \
    --algo a2c \
    --num_T 1000000 --test_interval 2000 \
    --sample_steps 256 --num_env 16 --num_minibatch 4 \
    --buffer_capacity 256 \
    --lr 3e-4 --lam 0.95 --hidden_dim 64