import sys
import torch
from typing import List, Optional, Union, Type
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.torch import Wrapper
from skrl.agents.torch import Agent
import tqdm
from agents.base_agent import BaseAgent
from utils.func import *
from isaaclab.utils.dict import print_dict as isaaclab_print_dict
from typing import List, Optional, Union, Type
import datetime
import os
import argparse
import re

def check_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    assert args.train ^ args.eval, "Exactly one of --train or --eval must be specified."
    assert not (args.eval and args.checkpoint is None), "When --eval is set, --checkpoint must not be None."
    assert args.env is not None, "Environment name must be specified with --env."

    if args.num_envs > 100 and not args.wandb:
        print(f"Enable Weight & Bias? (y/n): ", end="")
        
        choice = input().strip().lower()
        if choice == "y": 
            args.wandb = True
    
    if args.num_envs != parser.get_default("num_envs") and args.dataset == parser.get_default("dataset"):
        default_motion = parser.get_default("dataset")
        print(f"Use default motion file ({default_motion})? (y/n): ", end="")

        choice = input().strip().lower()
        if choice == "n": 
            print(f"Please specify the motion file: ", end="")
            args.dataset = input().strip()

    if args.agent is None:
        print(f"Please specify agent ", end="")
        args.agent = input().strip()

    # Check if running in a remote/headless environment and auto-enable headless mode
    if not hasattr(args, 'headless') or not args.headless:
        if is_remote():
            if hasattr(args, 'headless'):
                args.headless = True
                print("[INFO] Remote environment detected, automatically enabling headless mode.")
            else:
                print("[WARNING] Remote environment detected, but 'headless' argument not available. Please add --headless True to avoid display errors.")

    # if args.video and args.num_envs == parser.get_default("num_envs"): 
    #     args.num_envs = 64

    return args

def enable_debugging(args: argparse.Namespace):
    if args.debug:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("[DEBUG] CUDA_LAUNCH_BLOCKING is set to 1 for debugging purposes.")
        if not is_remote():
            import debugpy
            debugpy.listen(("localhost", 2333))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
    else:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def setup_learning_rate(cfg: dict, lr: Union[float, int, str], KLAdaptiveLR: Type | None=None):
    if isinstance(lr, str):
        if lr.lower() == "default":
            # use default learning rate in agent config
            cfg["learning_rate_scheduler"] = None

        elif lr.lower().startswith("kl"):
            match = re.match(r"kl([0-9.eE+-]+)", lr, re.IGNORECASE)
            if match:
                user_lr = float(match.group(1))
                cfg["learning_rate"] = user_lr

            cfg["learning_rate_scheduler"] = KLAdaptiveLR
            cfg["learning_rate_scheduler_kwargs"] = {
                "min_lr": cfg["learning_rate"]/100.0, 
                "max_lr": cfg["learning_rate"]*100.0,

                # if KL > threshold * factor, lr = max(lr / lr_factor, min_lr)
                # if KL < threshold / factor, lr = min(lr * lr_factor, max_lr)
                "kl_threshold": 0.2,    # Threshold for KL divergence (default: 0.008)
                "kl_factor": 2,         # The number used to modify the KL divergence threshold (default: 2),
                "lr_factor": 1.2,       # The number used to modify the learning rate (default: 1.5)
            }

    elif isinstance(lr, (float, int)):
        cfg["learning_rate_scheduler"] = None
        cfg["learning_rate"] = lr

    return cfg

def parse_lr(value):
    try:
        return float(value)
    except ValueError:
        return value

def set_viewer(cfg, eye: tuple | None=None, lookat: tuple | None=None, resolution: tuple | None=None):
    if resolution:  cfg.viewer.resolution = resolution
    if eye:         cfg.viewer.eye = eye
    if lookat:      cfg.viewer.lookat = lookat
    return cfg

def wrap_video(args, env, gym: Type, path: str):
    args.enable_cameras = True
    video_kwargs = {
        "video_folder": path,
        "video_length": args.video_length,
        "disable_logger": True,
        "name_prefix": 'video',
    }
    if args.video_interval > 0:
        video_kwargs.update({"step_trigger": lambda step: step % args.video_interval == 0})
    print("[INFO] Recording videos during training.")
    isaaclab_print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    return args, env

def setup_experiment_name(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, str]:
    dataset = str(args.dataset).replace("/", "_").replace("\\", "_").replace(".","_")
    experiment_name = f"{args.agent}_{args.num_envs}envs_{int(args.steps/10000)}w_{dataset}"

    if args.name:
        experiment_name = f"{experiment_name}_{args.name}"
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        experiment_name = f"{experiment_name}_{timestamp}"
    
    full_name = f"{args.env}_{experiment_name}"
    return experiment_name, full_name

class Trainer(SequentialTrainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)


    def train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # reset env
        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states


def evaluate(agent: BaseAgent, env, args):
    agent.set_running_mode("eval")
    disable_agent_grads(agent)
    timestep, timesteps = 0, 100000
    # reset env
    states, infos = env.reset()
    while(True):
        # pre-interaction
        agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            # compute actions
            actions = agent.act(states, timestep=timestep, timesteps=100000)[0]

            # step the environments
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # render scene
            if not args.headless:
                env.render()
            
            agent.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )

        # reset environments
        states = next_states