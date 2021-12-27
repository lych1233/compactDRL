CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario atari --env pong \
    --algo dqn \
    --num_T 10000000 --test_interval 20000 \
    --update_frequency 4 \
    --lr 6e-5 --hidden_dim 512 