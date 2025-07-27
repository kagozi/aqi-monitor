# File: dqn/train.py
import sys
import os
import random
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulated.environment import  FanControlGymEnv
from simulated.callbacks import EnhancedLoggingCallback
from simulated.physics import FanControlEnv

from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Set all random seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
CHECKPOINT_DIR = './a2c/train/' # 
LOG_DIR = './a2c/logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# Create and wrap the environment
fan_env = FanControlEnv()

env = FanControlGymEnv(fan_env=fan_env)
# Initialize A2C model with similar configuration where applicable
model = A2C(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    gamma=0.99,             # Discount factor
    n_steps=5,              # Default for A2C, controls update frequency
    gae_lambda=1.0,         # Generalized Advantage Estimation lambda
    ent_coef=0.0,           # Entropy coefficient (can encourage exploration)
    vf_coef=0.5,            # Value function loss coefficient
    max_grad_norm=0.5,      # Gradient clipping
    use_rms_prop=True,      # Default optimizer setting for A2C
    verbose=1,
    seed=SEED,          # Seed for reproducibility
)
# Set up and attach custom logger
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000, callback=callback)

