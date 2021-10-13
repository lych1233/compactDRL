CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario atari --env qbert \
    --algo ppo \
    --num_T 15000000 --test_interval 30000 \
    --sample_steps 1024 --num_env 8 --num_minibatch 4 \
    --buffer_capacity 1024 \
    --lr 1e-4 --lam 0.95 --clip_ratio 0.1 --hidden_dim 512 --value_loss_coef 1 --entropy_coef 0.01