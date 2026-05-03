from agents.ppo import PPO, PPO_DEFAULT_CONFIG
from models.ppo import *
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

def setup(env, args, experiment_cfg, device):

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "experiment": experiment_cfg,

        "rollouts": 16,  
        "learning_epochs": 8,  
        "mini_batches": 2,  
        "learning_rate": args.lr,   

        "state_preprocessor": RunningStandardScaler,       
        "state_preprocessor_kwargs": {"size": env.observation_size},    
        "value_preprocessor": RunningStandardScaler,  
        "value_preprocessor_kwargs": {"size": 1},  

        "time_limit_bootstrap": True, # bootstrap at timeout termination (episode truncation)
    })

    rollout_memory = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device  
    )

    models = {}

    models["policy"] = Policy(
        env.observation_size, 
        env.action_size, 
        initial_log_std=-2.9,
        fixed_log_std=True,
        device=device,
        num_layers=4,
        is_train=args.train
    )

    models["value"] = Value(
        env.observation_size,
        env.action_size, 
        num_layers=4,
        device=device
    )
    
    agent = PPO(
        models=models,
        memory=rollout_memory,  
        cfg=agent_cfg,
        observation_space=env.observation_size,
        action_space=env.action_size,
        device=device
    )
    
    return agent