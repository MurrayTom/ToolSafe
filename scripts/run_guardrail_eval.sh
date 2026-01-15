nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/agentharm_traj.yaml > guardian_test.log 2>&1 &

nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/asb_traj.yaml > guardian_test.log 2>&1 &

nohup python src/guardian_experiment.py --config ./src/config_guardrail_eval/agentdojo_traj.yaml > guardian_test.log 2>&1 &
