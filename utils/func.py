import sys
import torch
from skrl.models.torch import Model
import os

def disable_agent_grads(agent):
    for k, v in vars(agent).items():
        if isinstance(v, torch.nn.Module):
            disable_grads(v)

def disable_grads(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def enable_grads(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True

def set_grads(model: torch.nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad

def find_nan(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1: return torch.isnan(tensor)
    return torch.isnan(tensor).any(dim=-1)

def has_nan(tensor: torch.Tensor) -> torch.Tensor:
    return torch.any(find_nan(tensor))

def all_zeros(x: torch.Tensor) -> bool:
    return bool(torch.all(x == 0))

def printl(msg, value: torch.Tensor | float | int | None=None):
    if value is None:
        print(f"\r{msg}", end="")
    else:
        print(f"\r{msg}: {value}", end="")

import time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{func.__name__}] Time: {end_time - start_time:.6f} sec")
        return result
    return wrapper


import pynvml
def get_gpu_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        total = mem_info.total / 1024**3  # GB
        used = mem_info.used / 1024**3
        free = mem_info.free / 1024**3

        print(f"GPU {i}: Total: {total:.2f} GB | Used: {used:.2f} GB | Free: {free:.2f} GB")

    pynvml.nvmlShutdown()
    return {"total": total, "used": used, "free": free}

def print_dict(data):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
            
def print_state_dict(modules: dict, title: str = "Dictionary Structure"):
    print(f"[{title}]")
    for model_name, state_dict in modules.items():
        if hasattr(state_dict, "keys"):
            print(f"  - {model_name}: {len(state_dict)} parameters")
            for key in list(state_dict.keys()): 
                print(f"      • {key}")
        else:
            print(f"  - {model_name}: non-dict module")

def format_dict_to_string(dictionary):
    if not dictionary:
        return None
    
    lines = []
    for key, value in dictionary.items():
        # Handle different value types appropriately
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                # Single value tensor
                lines.append(f"{key}: {value.item():.4f}")
            else:
                # Multi-value tensor - format as list
                if value.dim() == 1 and len(value) <= 10:  # Only format short tensors
                    formatted_values = [f"{v:.3f}" for v in value]
                    lines.append(f"{key}: [{', '.join(formatted_values)}]")
                else:
                    lines.append(f"{key}: {value.shape}")
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        elif isinstance(value, str):
            lines.append(f"{key}: {value}")
        else:
            # For other types, use string representation
            lines.append(f"{key}: {str(value)}")
    
    return "\n".join(lines)


def anneal(start: float, end: float, progress: float) -> float:
    if progress < 0.0 or progress > 1.0:
        raise ValueError("Progress must be between 0 and 1.")
    return start + (end - start) * progress

def is_remote():
    return (
            os.environ.get('SSH_CLIENT') is not None or  # SSH connection
            os.environ.get('SSH_TTY') is not None or     # SSH TTY
            os.environ.get('DISPLAY') is None or         # No display
            os.environ.get('DISPLAY') == '' or           # Empty display
            'TMUX' in os.environ or                      # Running in tmux
            'STY' in os.environ                          # Running in screen
        )

