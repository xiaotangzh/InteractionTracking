source /home/xiaotang/miniconda3/etc/profile.d/conda.sh
conda activate isaac

EXTRA_ARGS="$@"

# Section: Testing
# Tracking
# python main.py --eval --disable_progressbar --env Tracking --agent PPO --num_envs 32 --steps 100000 --robot LAFAN1 --dataset LaFan1_aiming_1 --checkpoint "/home/xiaotang/Downloads/agent_10000.pt" $EXTRA_ARGS 

# Tracking (ADD)
# python main.py --eval --disable_progressbar --env Tracking --agent ADD --num_envs 32 --steps 100000 --dataset "AMASS_run_large" --checkpoint "/home/xiaotang/Downloads/agent_250000.pt" $EXTRA_ARGS 

# T-REX demo
# python main.py --train --env TREX_Demo --agent TREX_DEMO --dataset LaFan1_walk_1 --robot LAFAN1  $EXTRA_ARGS --checkpoint "./logs/Tracking/TREX_CKPT_5000envs_1000w_LaFan1_walk_1_[unfixed-log-std]/checkpoints" #--video --video_length 30 --video_path "/home/xiaotang/Downloads"

# Section: Training
# Tracking
# python main.py --train --env Tracking --agent PPO --dataset LaFan1_walk_1 --robot LAFAN1  $EXTRA_ARGS 

# T-REX gaussian
# python main.py --train --env Tracking_TREX --agent TREX_UNCERTAINTY --dataset LaFan1_walk_1 --robot LAFAN1 --name "[CLSTransformer-hidden128-4head-2layer-dropout01-Snorm][loss-bt][incl-val-data-no-stage0]"  --checkpoint "./logs/Tracking/TREX_CKPT_10000envs_1000w_LaFan1_walk_1_[unfixed-log-std][bin-save-002]/checkpoints"  $EXTRA_ARGS
# python main.py --train --env Tracking_TREX --agent TREX_GAUSSIAN --dataset LaFan1_walk_1 --robot LAFAN1 --name "[CLSTrans-128-Snorm][loss-bt-var+bilip][incl-val-noisy-data]"  --checkpoint "./logs/Tracking/TREX_CKPT_10000envs_1000w_LaFan1_walk_1_[unfixed-log-std][bin-save-002]/checkpoints"  $EXTRA_ARGS

# T-REX ckpt
# python main.py --train --env Tracking --agent TREX_CKPT --dataset LaFan1_run_1 --robot LAFAN1  $EXTRA_ARGS

# T-REX infer
python main.py --train --env TREX --agent PPO --dataset LaFan1_walk_1 --robot LAFAN1  $EXTRA_ARGS

# Tracking (ADD)
# python main.py --train --env Tracking --agent ADD --dataset AMASS_run  $EXTRA_ARGS

# Sync
# python main.py --train --env Sync --agent PPO --dataset LaFan1_aiming_1 --robot LAFAN1 $EXTRA_ARGS

# eg26
# python main.py --train --env Tracking --agent PPO --dataset "AMASS_style/highkick"  $EXTRA_ARGS 

# Section: Deep Learning