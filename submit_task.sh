############  Guardrail Model Evaluation ############

# nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/agentharm_traj.yaml > guardian_test.log 2>&1 &

# nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/asb_traj.yaml > guardian_test.log 2>&1 &

# nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/agentdojo_traj.yaml > guardian_test.log 2>&1 &


############  RUN experiments  ############

nohup python src/main_experiment.py --config ./src/config/agentharm.yaml > run_agentharm.log 2>&1 &

# nohup python src/main_experiment.py --config ./src/config/asb.yaml > run_asb.log 2>&1 &

# nohup python src/main_experiment.py --config ./src/config/agentdojo.yaml > run_agentdojo.log 2>&1 &


















##### Model Deployment #####

# ps aux | grep VLLM | grep -v grep | awk '{print $2}' | xargs kill -9
# ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9

# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_ashell_guardian_v2.4.0-rollout16_binary/qwen2.5_7b_function_rm/global_step_80/actor_hf ashell-guardian-binary 8000 0 > vllm_ashell-guardian.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_ashell_guardian_v2.4.0-rollout16_multitask_uniform/qwen2.5_7b_function_rm/global_step_80/actor_hf ashell-guardian 8000 0 > vllm_ashell-guardian.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_ashell_guardian_v2.4.0-rollout16/qwen2.5_7b_instruct_function_rm/global_step_40/actor_hf ashell-guardian 8000 0 > vllm_ashell-guardian.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/Models/Safiron safiron 8000 0 > vllm_safiron.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/Models/ShieldAgent shieldagent 8000 1 > vllm_shieldagent.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/guard/Llama-Guard-3-8B llamaguard3-8b 8200 2 > vllm_llamaguard3-8b.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct 8000 0 > vllm_qwen2.5-7b-instruct.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-30B-A3B-Instruct-2507 Qwen3-30B-A3B-Instruct-2507 8000 0 > vllm_Qwen3-30B-A3B-Instruct-2507.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-14B-Instruct Qwen2.5-14B-Instruct 6000 0 > vllm_Qwen2.5-14B-Instruct.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct 8200 2 > vllm_Llama-3.1-8B-Instruct.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-8B Qwen3-8B 8300 3 > vllm_Qwen3-8B.log 2>&1 &

# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/Models/saves/Qwen2.5-7B-Instruct/full/Qwen2.5-7B-Instruct-ReAlign-stage1SFT-stage2DPO qwen2.5-7b-instruct-ReAlign 8000 0 > vllm_qwen2.5-7b-instruct-ReAlign.log 2>&1 &

# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/Models/Qwen2-7B-Instruct qwen2-7b-instruct 8000 0 > vllm_qwen2-7b-instruct.log 2>&1 &
# nohup bash vllm_server.sh /mnt/shared-storage-user/mouyutao/AShell-ours/verl-main/checkpoints/verl_grpo_ashell_guardian_v2.3.5-rollout16/qwen2.5_7b_instruct_function_rm/global_step_40/actor_hf ashell-guardian-single 8000 0 > vllm_ashell-guardian_st.log 2>&1 &
##### RUN experiments #####

# rm run_agentharm.log
# nohup python src/main_experiment_v2.py --config ./src/config/asb.yaml > run_asb.log 2>&1 &
# nohup python src/main_experiment_v2.py --config ./src/config/agentdojo.yaml > run_agentdojo.log 2>&1 &
# nohup python src/main_experiment_v2.py --config ./src/config/agentharm.yaml > run_agentharm_2.log 2>&1 &
# nohup python src/main_experiment_v2.py --config ./src/config/bfcl.yaml > run_asb.log 2>&1 &
# nohup CUDA_VISIBLE_DEVICES=1 python src/guardian_experiment.py > run_agentsafetybench.log 2>&1 &

# nohup python src/main_experiment_v2.py --config ./src/config/agentharm.yaml > run_agentharm.log 2>&1 &
# nohup python src/main_experiment_v2.py --config ./src/config/asb.yaml > run_asb.log 2>&1 &

# nohup python src/main_experiment_v2.py --config ./src/config/agentdojo.yaml > run_agentdojo.log 2>&1 &

##### RUN guardian test #####
#rm guardian_test.log
#CUDA_VISIBLE_DEVICES=0 
#nohup python src/guardian_experiment.py > guardian_test.log 2>&1 &

# nohup python src/entropy_analysis.py > entropy_analysis.log 2>&1 &