root_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints
project_name=verl_grpo_ReAlign_v2/qwen2_7b_function_rm

python legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $root_dir/$project_name/global_step_35/actor \
    --hf_model_path $root_dir/$project_name/global_step_35/actor/huggingface \
    --target_dir $root_dir/$project_name/global_step_35/actor_hf

if [[ $? -ne 0 ]]; then
    echo "ERROR: model merge failed!"
    exit 1
fi

rm -rf $root_dir/$project_name/global_step_35/actor

# /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_ashell_stage1_v2/qwen2.5_7b_instruct_function_rm_no-sft/global_step_40/actor_hf