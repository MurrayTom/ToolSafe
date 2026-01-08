# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

data_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main
model_dir=/mnt/shared-storage-user/ai4good2-share/models
# model_dir=/mnt/shared-storage-user/mouyutao/Models
# model_dir=/mnt/shared-storage-user/mouyutao/Models/saves
reward_dir=/mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/verl/utils/reward_score

#$model_dir/Qwen2.5-7B-Instruct/merged/Qwen2.5-7B-Instruct-ashell-v2-stage1
#$model_dir/Qwen/Qwen2.5-7B-Instruct

#data.train_files=$data_dir/data/agentsafety/train.parquet \
#data.val_files=$data_dir/data/agentsafety/test.parquet \

# data.train_files=[$data_dir/data/agentsafety/train_agentalign-harm.parquet,$data_dir/data/agentsafety/train_agentalign-benign.parquet,$data_dir/data/agentsafety/train_asb-dpi.parquet,$data_dir/data/agentsafety/train_asb-opi.parquet] \
# data.val_files=[$data_dir/data/agentsafety/val_agentalign-harm.parquet,$data_dir/data/agentsafety/val_agentalign-benign.parquet,$data_dir/data/agentsafety/val_asb-dpi.parquet,$data_dir/data/agentsafety/val_asb-opi.parquet] \

# data.train_files=[$data_dir/data/agentsafety/train_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/train_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/train_asb-dpi.parquet,$data_dir/data/agentsafety/train_asb-opi.parquet] \
# data.val_files=[$data_dir/data/agentsafety/val_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/val_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/val_asb-dpi.parquet,$data_dir/data/agentsafety/val_asb-opi.parquet] \
    
# data.train_files=[$data_dir/data/llmsafety/train_llmsafety_queryonly.parquet,$data_dir/data/llmsafety/train_llmsafety.parquet] \
# data.val_files=[$data_dir/data/llmsafety/val_llmsafety_queryonly.parquet,$data_dir/data/llmsafety/val_llmsafety.parquet] \

# data.train_files=[$data_dir/data/agentsafety/train_prompt_inject_dpi.parquet,$data_dir/data/agentsafety/train_prompt_inject_opi.parquet,$data_dir/data/agentsafety/train_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/train_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/train_asb-dpi.parquet,$data_dir/data/agentsafety/train_asb-opi.parquet] \
# data.val_files=[$data_dir/data/agentsafety/val_prompt_inject_dpi.parquet,$data_dir/data/agentsafety/val_prompt_inject_opi.parquet,$data_dir/data/agentsafety/val_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/val_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/val_asb-dpi.parquet,$data_dir/data/agentsafety/val_asb-opi.parquet] \
    

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=[$data_dir/data/agentsafety/train_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/train_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/train_agentalign-benign-seedset0.parquet,$data_dir/data/agentsafety/train_asb-dpi_v2.parquet,$data_dir/data/agentsafety/train_asb-opi.parquet] \
    data.val_files=[$data_dir/data/agentsafety/val_agentalign-harm-seedset1.parquet,$data_dir/data/agentsafety/val_agentalign-harm-seedset0.parquet,$data_dir/data/agentsafety/val_agentalign-benign-seedset0.parquet,$data_dir/data/agentsafety/val_asb-dpi_v2.parquet,$data_dir/data/agentsafety/val_asb-opi.parquet] \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/mouyutao/Models/lzc/Qwen2.5-7B-Instruct/merged/Qwen2.5-7B-Instruct-TooGuard-sft \
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$reward_dir/agentsafety_v2_uniform.py \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_ashell_guardian_v2.4.0-rollout16_sft_rl' \
    trainer.experiment_name='qwen2.5_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 $@