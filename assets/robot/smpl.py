from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

SMPL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), "assets", "robot", "usd", "smpl_humanoid_small_head.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 1.0, 1.0), metallic=0.5
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),  # Default initial state of SMPL with the upright configuration
        rot = (1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Hip_.",
                "R_Hip_.",
                "L_Knee_.",
                "R_Knee_.",
                "L_Ankle_.",
                "R_Ankle_.",
                "L_Toe_.",
                "R_Toe_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[
                "Torso_.",
                "Spine_.",
                "Chest_.",
                "Neck_.",
                "Head_.",
                "L_Thorax_.",
                "R_Thorax_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Shoulder_.",
                "R_Shoulder_.",
                "L_Elbow_.",
                "R_Elbow_.",
                "L_Wrist_.",
                "R_Wrist_.",
                "L_Hand_.",
                "R_Hand_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 500,
                "R_Elbow_.": 500,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
                "L_Hand_.": 300,
                "R_Hand_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 50,
                "R_Elbow_.": 50,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
    },
)

SMPL_Upright_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), "assets", "robot", "usd", "smpl_humanoid_upright.usda"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95), # Default initial state of SMPL with the upright configuration
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["L_Hip_.", "R_Hip_.", "L_Knee_.", "R_Knee_.", "L_Ankle_.", "R_Ankle_.", "L_Toe_.", "R_Toe_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
                
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["Torso_.", "Spine_.", "Chest_.", "Neck_.", "Head_.", "L_Thorax_.", "R_Thorax_."],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_Shoulder_.", "R_Shoulder_.", "L_Elbow_.", "R_Elbow_.", "L_Wrist_.", "R_Wrist_.", "L_Hand_.", "R_Hand_."],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
                "L_Hand_.": 300,
                "R_Hand_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
    },
)