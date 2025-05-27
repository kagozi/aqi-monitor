from aqi import AirQualityDiscreteEnv, TrainAndLoggingCallback, EnhancedLoggingCallback
from stable_baselines3 import A2C
import os
from stable_baselines3.common.logger import configure

CHECKPOINT_DIR = './a2c/train/'
LOG_DIR = './a2c/logs/'
callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# Create and wrap the environment
env = AirQualityDiscreteEnv()

# Initialize A2C model with similar configuration where applicable
model = A2C(
    'MlpPolicy',
    env,
    learning_rate=1e-4,     # Matching DQN's learning rate
    gamma=0.99,             # Discount factor
    n_steps=5,              # Default for A2C, controls update frequency
    gae_lambda=1.0,         # Generalized Advantage Estimation lambda
    ent_coef=0.0,           # Entropy coefficient (can encourage exploration)
    vf_coef=0.5,            # Value function loss coefficient
    max_grad_norm=0.5,      # Gradient clipping
    use_rms_prop=True,      # Default optimizer setting for A2C
    verbose=1,
)

# Set up and attach custom logger
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000, callback=callback)