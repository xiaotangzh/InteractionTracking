from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
import torch
import argparse
import os
from isaaclab.app import AppLauncher
import gymnasium as gym
import sys

# do not import isaaclab_tasks before launching the app
from utils.func import *
from utils.run import *

# parse the arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=1000000, help="Number of training steps.")
parser.add_argument("--env", type=str, default=None, help="Name of the environment.")
parser.add_argument("--name", type=str, default="", help="Name of the experiment.")
parser.add_argument("--agent", type=str, default=None, help="Agent to use for training (e.g., PPO, AMP).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval to save model checkpoints (in steps).")
parser.add_argument("--write_interval", type=int, default=100, help="Interval to write tracking data.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=0, help="Interval to record videos (in steps).")
parser.add_argument("--video_path", type=str, default=None, help="Path to save videos.")
parser.add_argument("--viewer", action="store_true", default=False, help="Set the viewer.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log training results to Weight and Bias.")
parser.add_argument("--lr", type=parse_lr, default="5e-5", help="Learning rate.")
parser.add_argument("--disable_progressbar", action="store_true", default=False, help="Disable progress bar of tqdm.")
parser.add_argument("--train", action="store_true", default=False, help="Training mode.")
parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the models and disable require_grad.")
parser.add_argument("--debug", action="store_true", default=False, help="Attach debugger to VSCode.")
parser.add_argument("--robot", type=str, default="SMPL", help="Robot type to use in the environment.")
parser.add_argument("--dataset", type=str, default="InterHuman_SMPL/1_1.npz", help="Path to motion files to use.")
parser.add_argument("--visualize", choices=["True", "False"], default="True", help="Path to motion files to use.")

# load and wrap the Isaac Lab environment
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
enable_debugging(args) # connect debugpy if --debug is set
args = check_args(args, parser) # check if the arguments are valid

# experiment configuration
experiment_name, full_name = setup_experiment_name(args, parser)
directory = os.path.join("logs", args.env)
experiment_cfg = {
    "directory": directory, 
    "experiment_name": experiment_name,   
    "checkpoint_interval": args.checkpoint_interval if args.num_envs > 100 else 0, # only create folder for large-scale training 
    "write_interval": args.write_interval,
    "wandb": args.wandb,      
    "wandb_kwargs": {
        "entity": "",
        "project": "",
        "name": full_name
    }
}
trainer_cfg = {
    "timesteps": args.steps, 
    "disable_progressbar": args.disable_progressbar
}

# start the app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaacsim.core must be imported after the app is started
from isaaclab_tasks.utils import parse_env_cfg
from envs.setup_cfgs import *
cfg = parse_env_cfg(args.env, num_envs=args.num_envs)
cfg = setup_env_config(cfg, args, args.env, args.robot, args.dataset)

# wrap environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.viewer: set_viewer(cfg, eye=(0.0, 6.0, 2.0), lookat=(0.0, 0.0, 1.0))

# prepare video path
video_path = args.video_path
video_interval = 0
if args.video:
    if video_path is None:
        video_path = os.path.join(directory, experiment_name, "videos")
    video_interval = args.video_interval
    os.makedirs(video_path, exist_ok=True)
env = gym.make(args.env, cfg=cfg, render_mode="rgb_array" if args.video else None, is_train=args.train,
               video_path=video_path, video_interval=video_interval)
if args.video and video_path: 
    args, env = wrap_video(args, env, gym, path=video_path)
env = wrap_env(env, wrapper="isaaclab-single-agent")

# setup experiment
import experiments 
set_seed(42)
agent_mappings = {
    "PPO": experiments.ppo,
    "PPOX2": experiments.ppox2,
    "AMP": experiments.amp,
    "ADD": experiments.add,
    "TREX": experiments.trex,
    "TREX_CKPT": experiments.trex_ckpt,
    "TREX_GAUSSIAN": experiments.trex_gaussian,
    "TREX_DEMO": experiments.trex_demo,
    "TREX_UNCERTAINTY": experiments.trex_uncertainty,
    "TREX_VARIANCE": experiments.trex_variance,
}
agent_name = args.agent.upper()
if agent_name in agent_mappings:
    agent = agent_mappings[agent_name].setup(env, args, experiment_cfg, device)
else:
    raise ValueError(f"Unknown agent: {args.agent}.")

# configure and instantiate the RL trainer
trainer = Trainer(cfg=trainer_cfg, env=env, agents=agent)

# resume checkpoint (if specified)
if args.checkpoint and not getattr(agent, "skip_main_checkpoint_load", False):
    from isaaclab.utils.assets import retrieve_file_path
    resume_path = retrieve_file_path(args.checkpoint)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path, is_train=args.train, skip_modules=['optimizer', 'optimizer_disc']) 

if args.eval:
    agent.set_mode("eval")
    agent.set_running_mode("eval")
    evaluate(agent, env, args)
else: 
    agent.set_mode("train")
    agent.set_running_mode("train")
    trainer.train()

# close the simulator
env.close()

# close sim app
simulation_app.close()




