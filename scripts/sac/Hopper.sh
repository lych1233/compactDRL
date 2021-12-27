CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario mujoco --env Hopper-v3 \
    --algo sac \
    --num_T 3000000 --test_interval 20000 \
    --lr 1e-3 --alpha 0.2