source /home/xiaotang/miniconda3/etc/profile.d/conda.sh
conda activate isaac

EXTRA_ARGS="$@"

# Section: Testing

# single mootion
# python main.py --eval --env Tracking --agent PPO --dataset kinect/kinect_record_smpl_skmotion.npz  $EXTRA_ARGS --checkpoint "/home/xiaotang/Files/Dropbox/Projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/Tracking/agentPPO env30000 step1000w full-dataset_[plus-glob-rot-obs]_[phc-dof-reward]/checkpoints/agent_30000.pt"

# mulitple motions
# python main.py --eval --env Tracking --agent PPO --dataset InterHuman_SMPL/132motions  $EXTRA_ARGS --checkpoint "/home/xiaotang/Files/Dropbox/Projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/Tracking/agentPPO env30000 step1000w full-dataset_[plus-glob-rot-obs]_[phc-dof-reward]/checkpoints/agent_50000.pt"

# interaction tracking
# python main.py --eval --env InteractionTracking --agent PPOX2 --dataset "InterHuman_SMPL/2349"  $EXTRA_ARGS  --checkpoint "/home/xiaotang/Files/Dropbox/Projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/InteractionTracking/default/agentPPOX2 env10000 step1000w motion_full-mixture-policy-[keypoint-obs-reward]/checkpoints/agent_100000.pt"

# kinect
# python main.py --eval --env Kinect --agent PPO --dataset InterHuman_SMPL/1880  $EXTRA_ARGS --checkpoint "/home/xiaotang/Files/Dropbox/Projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/Tracking/agentPPO env4000 step1000w motion-kinect-mlp-policy-[keypoint-obs]/checkpoints/agent_40000.pt"

# Skill Transition
# python main.py --train --env SkillTransition --agent PPOX2 --dataset InterHuman_SMPL/skill_transitions/5587_5559  $EXTRA_ARGS --checkpoint "/home/xiaotang/NCC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/SkillTransition/agentPPOX2 env10000 step1000w cat5587_5559-[keypose-obs-reward][PRT-interaction-tracking][lift-height-0.1][random-reset]/checkpoints/agent_40000.pt"

# Distillation
# python main.py --eval --env Distill --agent DAGGER --dataset InterHuman_SMPL/132motions --checkpoint "/home/xiaotang/NCC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/Distill/agentDAGGER env4096 step1000w 132motions-4experts-sample_by_reward-plus_dof_vel_obs_reward-no_energy/checkpoints/agent_20000.pt" $EXTRA_ARGS

# Control
# python main.py --eval --env InteractionTracking --agent HRL --dataset InterHuman_SMPL/1880 --checkpoint "/home/xiaotang/NCC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/PhysicsProject/logs/InteractionTracking/agentHRL env4096 step1000w motion1880/checkpoints/agent_330000.pt" $EXTRA_ARGS


# Section: Training
# python main.py --train --env Tracking --agent PPO --dataset InterHuman_SMPL/1880  $EXTRA_ARGS 
# python main.py --train --env InteractionTracking --agent PPOX2 --dataset InterHuman_SMPL/1880 $EXTRA_ARGS
# python main.py --train --env Distill --agent Dagger --dataset InterHuman_SMPL/132motions $EXTRA_ARGS
# python main.py --train --env InteractionTracking --agent HRL --dataset InterHuman_SMPL/1880  $EXTRA_ARGS

# revision
python main.py --eval --env InteractionTracking --agent PPOX2 --dataset InterHuman/2613  $EXTRA_ARGS --checkpoint "/home/xiaotang/Desktop/agent_10000.pt"


# test: mixture of experts
# python main.py --train --env Tracking --agent MOE --dataset InterHuman_SMPL/1880  $EXTRA_ARGS  --checkpoint "./logs/Tracking/agentMOE env4000 step1000w motion1880+MoE+multi_std_value+copy_all_param+select_by_random/checkpoints/agent_90000.pt"

# test: train on kinect motion
# python main.py --train --env Tracking --agent PPO --dataset kinect/kinect_record_smpl_skmotion.npz  $EXTRA_ARGS 

# Sync
# python main.py --train --env Sync --agent PPO --dataset InterHuman_SMPL/full/6929_1.npz $EXTRA_ARGS

# Deep learning
# python -m deep_learning.train_interaction_encoder  $EXTRA_ARGS

