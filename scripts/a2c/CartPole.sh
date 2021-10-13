CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env CartPole-v1 \
    --algo a2c \
    --num_T 1000000 --test_interval 2000 \
    --sample_steps 1024 --num_env 32 --num_minibatch 4 \
    --buffer_capacity 1024 \
    --lr 1e-4 --lam 0.95