# File: simulated/callbacks.py
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
# File: simulated/callbacks.py
# import os
# import numpy as np
# import pandas as pd
# from stable_baselines3.common.callbacks import BaseCallback

# class EnhancedLoggingCallback(BaseCallback):
#     def __init__(self, check_freq, save_path, verbose=1):
#         super(EnhancedLoggingCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.save_path = save_path
#         self.metrics = {
#             'aqi': [],
#             'fan_speed': [],
#             'power': [],
#             'reward': []
#         }
#         self.df = pd.DataFrame(columns=['timestep', 'aqi', 'fan_speed', 'power', 'reward'])
#         self.last_info = {}  # Store last info for algorithms that don't provide it every step

#     def _init_callback(self):
#         os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self):
#         # Try to get info from different possible locations in locals
#         info = self._extract_info_from_locals()
#         reward = self._extract_reward_from_locals()
        
#         # Update last info with whatever we found
#         if info:
#             self.last_info.update(info)
        
#         # Store metrics
#         self.metrics['aqi'].append(self.last_info.get('aqi', 0))
#         self.metrics['fan_speed'].append(self.last_info.get('fan_speed', 0))
#         self.metrics['power'].append(self.last_info.get('power', 0))
#         self.metrics['reward'].append(reward)

#         # Log every 100 steps
#         if self.n_calls % 100 == 0:
#             self._log_metrics()

#         # Save model and full metrics every check_freq steps
#         if self.n_calls % self.check_freq == 0:
#             self._save_model()
#             self._log_to_tensorboard()
#             self._reset_metrics()

#         return True

#     def _extract_info_from_locals(self):
#         """Extract info from various possible locations in locals dict"""
#         info = {}
        
#         # Try different locations where info might be stored
#         possible_info_locations = [
#             self.locals.get('infos', []),
#             self.locals.get('info', {}),
#             self.locals.get('new_obs', {}),  # Sometimes info is in new_obs
#             self.last_info  # Fallback to last known info
#         ]
        
#         for location in possible_info_locations:
#             if isinstance(location, list) and len(location) > 0:
#                 if isinstance(location[0], dict):
#                     info.update(location[0])
#                     break
#             elif isinstance(location, dict):
#                 info.update(location)
#                 break
                
#         return info

#     def _extract_reward_from_locals(self):
#         """Extract reward from various possible locations in locals dict"""
#         # Try different reward locations
#         reward_locations = [
#             self.locals.get('rewards', [0]),
#             self.locals.get('reward', 0),
#             self.locals.get('values', 0),  # Sometimes reward is in values
#             0  # Fallback
#         ]
        
#         for location in reward_locations:
#             if isinstance(location, list) and len(location) > 0:
#                 return float(location[0])
#             elif isinstance(location, (int, float)):
#                 return float(location)
                
#         return 0.0

#     def _log_metrics(self):
#         if len(self.metrics['aqi']) > 0:  # Only log if we have data
#             new_row = {
#                 'timestep': self.num_timesteps,
#                 'aqi': np.mean(self.metrics['aqi'][-100:]),  # Last 100 steps
#                 'fan_speed': np.mean(self.metrics['fan_speed'][-100:]),
#                 'power': np.mean(self.metrics['power'][-100:]),
#                 'reward': np.mean(self.metrics['reward'][-100:])
#             }
#             new_df = pd.DataFrame([new_row])
#             self.df = pd.concat([self.df, new_df], ignore_index=True)
#             self.df.to_csv(os.path.join(self.save_path, 'training_metrics.csv'), index=False)

#     def _save_model(self):
#         model_path = os.path.join(self.save_path, f'model_{self.num_timesteps}')
#         self.model.save(model_path)

#     def _log_to_tensorboard(self):
#         if len(self.metrics['aqi']) > 0:
#             self.logger.record('env/aqi', np.mean(self.metrics['aqi'][-100:]))
#             self.logger.record('env/fan_speed', np.mean(self.metrics['fan_speed'][-100:]))
#             self.logger.record('env/power', np.mean(self.metrics['power'][-100:]))
#             self.logger.record('env/reward', np.mean(self.metrics['reward'][-100:]))

#     def _reset_metrics(self):
#         # Don't reset completely, just keep recent history
#         for key in self.metrics:
#             if len(self.metrics[key]) > 1000:  # Keep last 1000 steps
#                 self.metrics[key] = self.metrics[key][-500:]