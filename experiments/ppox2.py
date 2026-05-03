from agents.ppox2 import PPOX2, PPOX2_DEFAULT_CONFIG
from models.ppo import *

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from utils.run import setup_learning_rate

def setup(env, args, experiment_config, device):
    agent_cfg = PPOX2_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "experiment": experiment_config,
        
        "memory_1": RandomMemory(memory_size=16, num_envs=env.num_envs, device=device),
        "memory_2": RandomMemory(memory_size=16, num_envs=env.num_envs, device=device),

        "learning_rate": args.lr,

        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.single_observation_size},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1},
        
        "time_limit_bootstrap": True, # prevent miscoupling terminate and truncate signals between two robots
    })

    models = {}

    models["policy"] = MixturePolicy(
        env.single_observation_size, 
        env.single_action_size, 
        initial_log_std=-2.9,
        fixed_log_std=True,
        hard_weights=False,
        num_experts=4,
        is_train=args.train,
        device=device)

    models["value"] = Value(
        env.single_observation_size, 
        env.single_action_size, 
        device=device)
    
    agent = PPOX2(models=models,
        memory=None,  
        cfg=agent_cfg,
        observation_space=env.single_observation_size,
        action_space=env.single_action_size,
        device=device)
    
    return agent