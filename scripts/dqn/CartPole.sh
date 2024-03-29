CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env CartPole-v1 \
    --algo dqn \
    --num_T 1000000 --test_interval 2000 \
    --batch_size 512 \
    --update_frequency 4 --target_update_interval 100
