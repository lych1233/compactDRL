CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env Acrobot-v1 \
    --algo rainbow \
    --num_T 1000000 --test_interval 2000 \
    --target_update_interval 100 \
    --minV -100 maxV 0
