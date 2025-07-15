import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulated.environment import  AirQualityDiscreteEnv
from simulated.callbacks import EnhancedLoggingCallback
from simulated.physics import GaussianPlumeModel
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


CHECKPOINT_DIR = './ppo/train/'
LOG_DIR = './ppo/logs/'
callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# Create and wrap the environment
# Initialize DQN model with more reasonable parameters

# Create Plume Model and Environment
plume_model = GaussianPlumeModel(
    min_voltage=12.0,
    max_voltage=24.0,
    voltage_step=1.0
)

env = AirQualityDiscreteEnv(plume_model=plume_model)
model = PPO(
    'MlpPolicy', 
    env, 
    tensorboard_log=LOG_DIR, 
    verbose=1,
    n_steps=2048,        # Number of steps to run for each environment per update
    batch_size=64,       # Minibatch size
    n_epochs=10,         # Number of epoch when optimizing the surrogate loss
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # Factor for trade-off of bias vs variance for GAE
    ent_coef=0.01,       # Entropy coefficient
    learning_rate=3e-4,  # Learning rate
    clip_range=0.2,      # Clipping parameter
)
# Set up and attach custom logger
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000, callback=callback)

