CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario atari --env pong \
    --algo a2c \
    --num_T 15000000 --test_interval 30000 \
    --sample_steps 128 --num_env 16 --num_minibatch 1 \
    --buffer_capacity 128 \
    --lr 3e-4 --lam 0.95 --hidden_dim 512 --value_loss_coef 1 --entropy_coef 0.01