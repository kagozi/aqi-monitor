from aqi import  AirQualityDiscreteEnv, TrainAndLoggingCallback, EnhancedLoggingCallback
from stable_baselines3 import DQN
import os
from stable_baselines3.common.logger import configure


CHECKPOINT_DIR = './dqn/train/'
LOG_DIR = './dqn/logs/'
callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# Create and wrap the environment
# Initialize DQN model with more reasonable parameters
env = AirQualityDiscreteEnv()
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    verbose=1,
)

# Set up and attach custom logger
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=100000, callback=callback)