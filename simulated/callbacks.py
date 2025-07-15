# File: simulated/callbacks.py (updated)
import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(EnhancedLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.metrics = {
            'aqi': [],
            'fan_speed': [],
            'power': [],
            'reward': []
        }
        self.df = pd.DataFrame(columns=['timestep', 'aqi', 'fan_speed', 'power', 'reward'])

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            self.metrics['aqi'].append(info.get('aqi', 0))
            self.metrics['fan_speed'].append(info.get('fan_speed', 0))
            self.metrics['power'].append(info.get('power', 0))
            self.metrics['reward'].append(self.locals['rewards'][0])

        # Log every 100 steps
        if self.n_calls % 100 == 0:
            self._log_metrics()

        # Save model and full metrics every check_freq steps
        if self.n_calls % self.check_freq == 0:
            self._save_model()
            self._log_to_tensorboard()
            self._reset_metrics()

        return True

    def _log_metrics(self):
        new_row = {
            'timestep': self.num_timesteps,
            'aqi': np.mean(self.metrics['aqi']),
            'fan_speed': np.mean(self.metrics['fan_speed']),
            'power': np.mean(self.metrics['power']),
            'reward': np.mean(self.metrics['reward'])
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        self.df.to_csv(os.path.join(self.save_path, 'training_metrics.csv'), index=False)

    def _save_model(self):
        model_path = os.path.join(self.save_path, f'model_{self.num_timesteps}')
        self.model.save(model_path)

    def _log_to_tensorboard(self):
        self.logger.record('env/aqi', np.mean(self.metrics['aqi']))
        self.logger.record('env/fan_speed', np.mean(self.metrics['fan_speed']))
        self.logger.record('env/power', np.mean(self.metrics['power']))
        self.logger.record('env/reward', np.mean(self.metrics['reward']))

    def _reset_metrics(self):
        self.metrics = {k: [] for k in self.metrics}