CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario atari --env pong \
    --algo rainbow \
    --num_T 15000000 --test_interval 30000 \
    --update_frequency 4 \
    --lr 6e-5 --hidden_dim 512 