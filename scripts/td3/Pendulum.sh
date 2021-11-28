CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env Pendulum-v0 \
    --algo td3 \
    --num_T 1000000 --test_interval 2000