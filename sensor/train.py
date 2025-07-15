import sys
import os
module_dir = os.path.abspath('./simulated') 
sys.path.append(module_dir)
from ppo_environment import MQTTFanEnv, EnhancedLoggingCallback
from stable_baselines3 import PPO
import os
from stable_baselines3.common.logger import configure

# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ppo', 'train')
LOG_DIR = os.path.join(BASE_DIR, 'ppo', 'logs')

# Ensure the directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

callback = EnhancedLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# Create and wrap the environment
# Initialize DQN model with more reasonable parameters
env = MQTTFanEnv()
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

'''
Train the model
24hrs x 60 minutes x 60 seconds / 5 seconds = 17280 steps per day. 
I will use about 17280 episodes to ensure the model has enough time to learn for about 1 day of data.
''' 
model.learn(total_timesteps=17280, callback=callback)  # 5s steps × 17280 = 24hrs