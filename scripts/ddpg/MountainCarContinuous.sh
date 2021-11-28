CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env MountainCarContinuous-v0 \
    --algo ddpg \
    --num_T 1000000 --test_interval 2000 \
    --OU_noise --online_noise_scale 0.5