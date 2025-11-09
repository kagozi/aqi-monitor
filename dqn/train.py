# dqn/train.py
import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulated.environment import FanControlGymEnv

CHECKPOINT_DIR = './dqn/train/'
LOG_DIR = './dqn/logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
TOTAL_STEPS = 500_000

env = make_vec_env(FanControlGymEnv, n_envs=8, seed=SEED)
eval_env = make_vec_env(FanControlGymEnv, n_envs=1, seed=SEED+1)

checkpoint_cb = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix='dqn')
eval_cb = EvalCallback(eval_env, best_model_save_path=CHECKPOINT_DIR, log_path=LOG_DIR,
                       eval_freq=10_000, n_eval_episodes=5, deterministic=True)

model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3,           # ← Higher LR for faster learning
    buffer_size=50_000,           # ← Smaller buffer (less old data)
    learning_starts=1_000,        # ← Start learning earlier
    batch_size=256,               # ← Larger batches (more stable)
    gamma=0.98,                   # ← Lower gamma (emphasize immediate rewards)
    train_freq=1,                 # ← Update every step
    target_update_interval=500,   # ← Faster target network updates
    exploration_fraction=0.5,     # ← Longer exploration (250k steps)
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,    # ← Higher final ε (keep exploring)
    tensorboard_log=LOG_DIR,
    seed=SEED,
    verbose=1,
    # Add these:
    gradient_steps=1,             # ← One grad step per env step
    # optimize_memory_usage=True,   # ← Efficiency
)

print("Training DQN...")
model.learn(total_timesteps=TOTAL_STEPS, callback=[checkpoint_cb, eval_cb])
model.save(os.path.join(CHECKPOINT_DIR, "dqn_final"))
print("Training complete!")