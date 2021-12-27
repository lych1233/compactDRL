CUDA_VISIBLE_DEVICES=$1 python run.py \
    --exp_name $0 --seed $2 \
    --scenario control --env MountainCarContinuous-v0 \
    --algo sac \
    --num_T 1000000 --test_interval 2000 \
    --additional_OU_noise 0.5 \
    --lr 1e-3 --alpha 0.2