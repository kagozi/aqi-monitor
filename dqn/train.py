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

from stable_baselines3 import DQN
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


CHECKPOINT_DIR = './dqn/train/' # 
LOG_DIR = './dqn/logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# Create and wrap the environment
fan_env = FanControlEnv()

env = FanControlGymEnv(fan_env=fan_env)
env.seed(SEED) 
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=128,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
    seed=SEED 
)
# Set up and attach custom logger
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000, callback=callback)

