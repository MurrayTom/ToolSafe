iter_id=1
model_path_init=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_labeler_v1-rollout32_iter0/qwen2.5_7b_instruct_function_rm/global_step_5

python iter_self-labeling.py \
    --iter_id $iter_id \
    --model_path $model_path_init/actor_hf