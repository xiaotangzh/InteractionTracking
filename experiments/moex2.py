from agents.moex2 import MOEX2, MOEX2_DEFAULT_CONFIG
from models.ppo import Value
from models.moe_unfreeze import MoE_Unfreeze, MoV

from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from utils.run import setup_learning_rate

def setup(env, args, experiment_config, device):
    agent_cfg = MOEX2_DEFAULT_CONFIG.copy()
    agent_cfg["experiment"] = experiment_config

    agent_cfg["memory_1"] = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device)
    agent_cfg["memory_2"] = RandomMemory(
        memory_size=agent_cfg["rollouts"], 
        num_envs=env.num_envs, 
        device=device)

    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.single_observation_size}
    agent_cfg["value_preprocessor"] = RunningStandardScaler 
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1}
    agent_cfg["time_limit_bootstrap"] = True # prevent miscoupling terminate and truncate signals between two robots

    agent_cfg = setup_learning_rate(agent_cfg, args.lr, KLAdaptiveLR)

    models = {}

    models["policy"] = MoE_Unfreeze(
        env.single_observation_size, 
        env.single_action_size, 
        hidden_dim=args.params,
        device=device,

        eval_mode=args.eval, 

        num_experts=4,
        init_expert=1, 

        print_log=False,

        min_reward_growth=0.002,
        reward_check_interval=20000,
        double_reward_check_interval=False,

        use_multiple_log_std=True,
        copy_parameters_to_new_expert=True,
        add_noise_when_copy_parameters=False,
        reinit_std_when_activate_new_expert=True,

        # zero_init_new_expert=False,
        init_adapters_as="random",
        adapter_usage_loss=False,

        # selection_strategy='inverse',
    )

    # note: cannot solve conflicting sampled_experts in two-robot env
    # models["value"] = MoV(
    #     env.observation_space.shape[0], 
    #     env.action_space.shape[0], 
    #     hidden_dim=args.params,
    #     num_experts=4,
    #     device=device)
    
    models["value"] = Value(
        env.single_observation_size, 
        env.single_action_size,  
        hidden_dim=args.params,
        device=device)
    
    agent = MOEX2(models=models,
        memory=None,  
        cfg=agent_cfg,
        observation_space=env.single_observation_size,
        action_space=env.single_action_size,
        device=device)
    
    return agent