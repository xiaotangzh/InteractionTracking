import os
from dataclasses import MISSING
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.direct.InteractionTracking.envs.env_cfgs import BaseEnvCfg

from isaaclab_assets import HUMANOID_28_CFG
from assets.robot.smpl import SMPL_CFG, SMPL_Upright_CFG
from isaaclab_tasks.direct.InteractionTracking.assets.robot.lafan1 import LAFAN1_CFG

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"

ROBOT_CONFIGS = {
    "SMPL": dict(
        cfg=lambda: SMPL_CFG.replace(prim_path="/World/envs/env_.*/robot"),
        action_space=69,
        key_body_names=["L_Hand", "R_Hand", "L_Toe", "R_Toe", "Head"],
        reference_body="Pelvis",
        root_height=0.95,
        terminate_root_height=0.8,
    ),
    "LAFAN1": dict(
        cfg=lambda: LAFAN1_CFG.replace(prim_path="/World/envs/env_.*/robot"),
        action_space=63,
        reference_body="Pelvis",
        key_body_names=["L_Wrist", "R_Wrist", "L_Toe", "R_Toe", "Head"],
        root_height=0.95,
        terminate_root_height=0.8,
    ),
    "HUMANOID_28": dict(
        cfg=lambda: HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/robot"),
        action_space=72,
        # etc...
    ),
}

def setup_env_config(cfg, args, env: str, robot: str = "SMPL", dataset_path: str = "InterHuman/1_1.npz"):
    cfg = setup_robot_config(cfg, robot)
    cfg = setup_sizes(cfg, env, robot)
    cfg = setup_dataset(cfg, dataset_path)
    cfg = setup_others(cfg, args)
    return cfg

def setup_sizes(cfg, env: str, robot: str):

    if "Interaction" in env or "SkillTransition" in env:
        if robot == "SMPL":
            cfg.robot1 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/robot1")
            cfg.robot2 = SMPL_CFG.replace(prim_path="/World/envs/env_.*/robot2")

            cfg.single_proprioception_space = 144
            cfg.single_observation_space = 498 + cfg.single_proprioception_space
            cfg.observation_space = 2 * cfg.single_observation_space
            cfg.single_action_space = 69
            cfg.action_space = 2 * cfg.single_action_space

            # temp: eg26, deepmimic
            # cfg.single_proprioception_space = 1+6+3+3+23*3*2+3*5
            # cfg.single_observation_space = 3+6+3+3+23*3*2+3*5 + cfg.single_proprioception_space
            # cfg.observation_space = 2 * cfg.single_observation_space

    elif env == "Distill":
        cfg.proprioception_space = 144
        cfg.tracking_space = cfg.observation_space - cfg.proprioception_space
    
    elif env == "AMP":
        if robot == "LAFAN1":
            cfg.observation_space = 1+6+3+3+21*3*2
            cfg.amp_observation_size = 1+6+3+3+21*3*2

    elif env == "TREX":
        pass

    else:
        if robot == "SMPL":
            # cfg.observation_space = 24 * 3 * 2 + (24 * 3 * 4)
            # cfg.amp_observation_size = 24*3*6 # ADD

            cfg.observation_space = 1+6+3+3+23*3*2+3*5 + 3+6+3+3+23*3*2+3*5

            # temp: eg26, PHC
            # cfg.observation_space = 498 + 144

        elif robot == "LAFAN1":
            cfg.observation_space = 1+6+3+3+21*3*2+3*5 + 3+6+3+3+21*3*2+3*5
            cfg.amp_observation_size = 1+6+3+3+21*3*2

    return cfg

def setup_robot_config(cfg, robot: str):
    if robot not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot}")
    rconf = ROBOT_CONFIGS[robot]

    cfg.robot = rconf["cfg"]()
    cfg.action_space = rconf["action_space"]
    cfg.key_body_names = rconf["key_body_names"]
    cfg.reference_body = rconf["reference_body"]
    cfg.root_height = rconf.get("root_height", None)
    cfg.terminate_root_height = rconf.get("terminate_root_height", 0.0)
    
    return cfg

def setup_dataset(cfg, motion_file: str):
    cfg.motion_file = str(DATASETS_DIR / motion_file)
    return cfg

def setup_others(cfg, args):
    if args.visualize == "False":
        cfg.visualize = False
    return cfg