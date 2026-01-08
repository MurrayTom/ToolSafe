# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

data_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main
# model_dir=/mnt/shared-storage-user/ai4good2-share/models
# model_dir=/mnt/shared-storage-user/mouyutao/Models/saves
iter_model_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints
reward_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/verl/utils/reward_score

total_iter=10

#$model_dir/Qwen2.5-7B-Instruct/merged/Qwen2.5-7B-Instruct-ashell-v2-stage1
#$model_dir/Qwen/Qwen2.5-7B-Instruct

#data.train_files=$data_dir/data/agentsafety/train.parquet \
#data.val_files=$data_dir/data/agentsafety/test.parquet \

# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=[$data_dir/data/agentsafety/train_agentalign-harm-seedset0.parquet] \
#     data.val_files=[$data_dir/data/agentsafety/val_agentalign-harm-seedset0.parquet] \
#     data.train_batch_size=256 \
#     data.max_prompt_length=4096 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=$model_dir/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=32 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     custom_reward_function.path=$reward_dir/agentsafety.py \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","tensorboard"]' \
#     trainer.project_name='verl_grpo_labeler_v1-rollout16' \
#     trainer.experiment_name='qwen2.5_7b_instruct_function_rm' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=5 $@

for iter_id in $(seq 1 $total_iter)
do
  echo "Starting iteration $iter_id of $total_iter"
    model_name_init=verl_grpo_labeler_v1-rollout32_iter$((iter_id-1))/qwen2.5_7b_instruct_function_rm
    model_path_init=$iter_model_dir/$model_name_init
    global_step=$(cat "$model_path_init/latest_checkpointed_iteration.txt")
    model_path_init=$model_path_init/global_step_$global_step
    echo "Using model path: $model_path_init at global step $global_step"

    if [[ -d "$model_path_init/actor_hf" ]]; then
        echo "actor_hf exists in $model_path_init"
    else
        echo "Merging model to Hugging Face format..."
        python /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir $model_path_init/actor \
            --hf_model_path $model_path_init/actor/huggingface \
            --target_dir $model_path_init/actor_hf
        if [[ $? -ne 0 ]]; then
            echo "ERROR: model merge failed!"
            exit 1
        fi
    fi

    rm -rf $model_path_init/actor

    ### Prepare for the next iteration training data
    python /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/data/agentsafety/iter_self-labeling.py \
        --iter_id $iter_id \
        --model_path $model_path_init/actor_hf

    project_name=verl_grpo_labeler_v1-rollout32_iter$iter_id

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=[$data_dir/data/agentsafety/train_agentalign-harm-seedset$iter_id.parquet] \
        data.val_files=[$data_dir/data/agentsafety/val_agentalign-harm-seedset$iter_id.parquet] \
        data.train_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=$model_path_init \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=$reward_dir/agentsafety.py \
        trainer.critic_warmup=0 \
        trainer.logger='["console","tensorboard"]' \
        trainer.project_name=$project_name \
        trainer.experiment_name='qwen2.5_7b_instruct_function_rm' \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=5 \
        trainer.total_epochs=5 $@
done
