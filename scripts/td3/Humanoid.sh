CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario mujoco --env Humanoid-v3 \
    --algo td3 \
    --num_T 3000000 --test_interval 20000 \
    --actor_lr 1e-4 --critic_lr 3e-4 --batch_size 256