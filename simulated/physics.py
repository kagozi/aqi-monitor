# simulated/physics.py
"""
Realistic physics engine for an RL-controlled indoor AQI monitor using a
small motorized fan (6x4 propeller, 80 W mechanical, 12–24 V).

Key features:
1. **Dilution ventilation** – fan mixes indoor/outdoor air (no filter).
2. **Mass-balance equation** with realistic airflow from propeller.
3. **Gravitational settling** via Stokes' law.
4. **AQI (1–5)** derived from PM2.5 concentration (EPA breakpoints).
5. **Motor power model** – 80 W mechanical, 12 V dead-zone, 24 V supply.
6. **RL-friendly dynamics**: smooth decay over minutes–hours.
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Tuple, Dict, Any


class Action(Enum):
    DECREASE = 1
    DO_NOTHING = 1
    INCREASE = 1


class FanControlEnv:
    # --------------------------------------------------------------------- #
    #                         INITIALISATION
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        room_volume: float = 50.0,          # m³
        floor_area: float = 25.0,           # m²
        outdoor_conc: float = 80.0,         # μg/m³ PM2.5 (polluted city)
        init_conc: float = 300.0,           # μg/m³ initial indoor
        target_aqi: int = 1,                # desired AQI level (1-5)
        max_steps: int = 1000,
        energy_weight: float = 0.3,         # AQI vs. energy trade-off
        particle_diameter: float = 2.5e-6,  # m  (PM2.5)
        particle_density: float = 1400.0,   # kg/m³ (typical PM)
        # ---- motor / propeller (80 W, 12–24 V, 20 000 RPM) -------------
        prop_diameter: float = 0.1524,      # 6" in meters
        prop_pitch: float = 0.1016,         # 4" in meters
        prop_efficiency: float = 0.65,      # typical
        voltage: float = 24.0,              # operating voltage
        min_voltage: float = 12.0,          # dead-zone threshold
        max_rpm: float = 20000.0,           # datasheet
    ) -> None:

        # ----------------------------------------------------------------- #
        #  Geometry & external conditions
        # ----------------------------------------------------------------- #
        self.room_volume = room_volume
        self.floor_area = floor_area
        self.outdoor_conc = outdoor_conc
        self.target_aqi = int(np.clip(target_aqi, 1, 5))

        # ----------------------------------------------------------------- #
        #  Simulation control
        # ----------------------------------------------------------------- #
        self.max_steps = max_steps
        self.energy_weight = energy_weight
        self.time_step = 5.0                     # seconds per step

        # ----------------------------------------------------------------- #
        #  Particle physics
        # ----------------------------------------------------------------- #
        self.particle_diameter = particle_diameter
        self.particle_density = particle_density

        # ----------------------------------------------------------------- #
        #  Motor / propeller (80 W mechanical, 12–24 V)
        # ----------------------------------------------------------------- #
        self.prop_diameter = prop_diameter
        self.prop_pitch = prop_pitch
        self.prop_efficiency = prop_efficiency
        self.voltage = voltage
        self.min_voltage = min_voltage
        self.min_speed = self.min_voltage / self.voltage   # = 0.5
        self.max_rpm = max_rpm

        # ----------------------------------------------------------------- #
        #  Dynamic state
        # ----------------------------------------------------------------- #
        self.current_conc: float = init_conc
        self.current_fan_speed: float = 0.5
        self.current_step: int = 0

        # ----------------------------------------------------------------- #
        #  Action / observation space
        # ----------------------------------------------------------------- #
        self.action_space = [a for a in Action]
        self.observation_space = np.array([0.0, 0.0], dtype=np.float32)

    # --------------------------------------------------------------------- #
    #                         PUBLIC API
    # --------------------------------------------------------------------- #
    def reset(self) -> np.ndarray:
        """Reset to a realistic polluted indoor state."""
        self.current_conc = float(np.clip(np.random.normal(300, 80), 100, 600))
        self.current_fan_speed = float(np.clip(np.random.normal(0.5, 0.1), 0.0, 1.0))
        self.current_step = 0
        return self._get_obs()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # ---- 1. Apply action with dead-zone ----------------------------- #
        speed_change = 0.1 * action.value
        new_speed = self.current_fan_speed + speed_change

        if 0.0 < new_speed < self.min_speed:
            new_speed = 0.0
        else:
            new_speed = np.clip(new_speed, self.min_speed, 1.0)

        self.current_fan_speed = float(new_speed)

        # ---- 2. Update concentration (dilution + settling) ------------- #
        self._update_concentration()

        # ---- 3. Reward --------------------------------------------------- #
        reward = self._calculate_reward()

        # ---- 4. Termination --------------------------------------------- #
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        # ---- 5. Info ---------------------------------------------------- #
        aqi = self.pm25_to_aqi(self.current_conc)
        rpm = self._get_actual_rpm()
        airflow = self._calculate_airflow(rpm)
        info = {
            "aqi": int(aqi),
            "concentration": round(self.current_conc, 2),
            "fan_speed": self.current_fan_speed,
            "power": round(self._calculate_power(), 2),
            "rpm": int(rpm),
            "airflow_m3s": round(airflow, 4),
            "airflow_m3h": round(airflow * 3600, 1),
            "ach": round(airflow * 3600 / self.room_volume, 3),
        }

        return self._get_obs(), reward, terminated, info

    def render(self, mode: str = "human") -> None:
        """Console visualization."""
        aqi = self.pm25_to_aqi(self.current_conc)
        rpm = self._get_actual_rpm()
        power = self._calculate_power()
        airflow = self._calculate_airflow(rpm) * 3600
        print(
            f"Step {self.current_step:4d} | AQI {aqi} | "
            f"Conc {self.current_conc:6.1f}µg/m³ | "
            f"Fan {self.current_fan_speed:.2f} ({int(rpm):5d} RPM) | "
            f"Power {power:5.1f}W | Flow {airflow:5.1f} m³/h"
        )

    # --------------------------------------------------------------------- #
    #                         INTERNAL HELPERS
    # --------------------------------------------------------------------- #
    @staticmethod
    def pm25_to_aqi(pm25: float) -> int:
        """Convert PM2.5 (μg/m³) → AQI level 1–5 (simplified EPA)."""
        if pm25 <= 12.0:
            return 1
        if pm25 <= 35.4:
            return 2
        if pm25 <= 55.4:
            return 3
        if pm25 <= 150.4:
            return 4
        return 5

    @staticmethod
    def aqi_to_pm25_midpoint(aqi_level: int) -> float:
        """Mid-point concentration for AQI level."""
        breakpoints = [0.0, 12.1, 35.5, 55.5, 150.5, 500.0]
        i = int(np.clip(aqi_level, 1, 5)) - 1
        return (breakpoints[i] + breakpoints[i + 1]) / 2.0

    def _get_obs(self) -> np.ndarray:
        """Normalized observation: [norm_aqi, fan_speed]."""
        aqi = self.pm25_to_aqi(self.current_conc)
        norm_aqi = (aqi - 1) / 4.0
        return np.array([norm_aqi, self.current_fan_speed], dtype=np.float32)

    # ----------------------------------------------------------------- #
    #  AIRFLOW FROM PROPELLER (6x4)
    # ----------------------------------------------------------------- #
    def _calculate_airflow(self, rpm: float) -> float:
        """Estimate airflow (m³/s) from 6x4 propeller."""
        if rpm <= 0:
            return 0.0
        pitch_speed = (rpm / 60.0) * self.prop_pitch
        area = np.pi * (self.prop_diameter / 2.0) ** 2
        theoretical = pitch_speed * area
        # Efficiency: 65%, derate 70% for room mixing
        return theoretical * self.prop_efficiency * 0.7

    # ----------------------------------------------------------------- #
    #  GRAVITATIONAL SETTLING
    # ----------------------------------------------------------------- #
    def _calculate_settling_rate(self) -> float:
        """Settling removal rate (1/s) via Stokes' law."""
        rho_p = self.particle_density
        rho_a = 1.225
        d = self.particle_diameter
        mu = 1.81e-5
        g = 9.81
        v_s = (rho_p - rho_a) * g * d ** 2 / (18.0 * mu)
        return (v_s * self.floor_area) / self.room_volume

    # ----------------------------------------------------------------- #
    #  MASS BALANCE: DILUTION VENTILATION + SETTLING
    # ----------------------------------------------------------------- #
    def _update_concentration(self) -> None:
        """Mass balance: fan brings in outdoor air, gravity removes particles."""
        rpm = self._get_actual_rpm()
        Q = self._calculate_airflow(rpm)           # m³/s
        aer = Q / self.room_volume                 # 1/s (ventilation)
        settling = self._calculate_settling_rate() # 1/s
        total_rate = aer + settling

        decay = np.exp(-total_rate * self.time_step)
        new_conc = self.outdoor_conc * (1.0 - decay) + self.current_conc * decay

        noise = np.random.normal(0.0, 0.5)
        self.current_conc = float(np.clip(new_conc + noise, 0.0, 1000.0))

    # ----------------------------------------------------------------- #
    #  REWARD
    # ----------------------------------------------------------------- #
    def _calculate_reward(self) -> float:
        cur_aqi = self.pm25_to_aqi(self.current_conc)
        aqi_error = abs(cur_aqi - self.target_aqi)
        aqi_reward = -aqi_error
        power_w = self._calculate_power()
        energy_cost = self.energy_weight * (power_w / 100.0)
        return aqi_reward - energy_cost

    # ----------------------------------------------------------------- #
    #  MOTOR POWER – 80 W MECHANICAL, 12 V DEAD-ZONE, 24 V SUPPLY
    # ----------------------------------------------------------------- #
    def _get_actual_rpm(self) -> float:
        """RPM after the 12 V dead-zone."""
        if self.current_fan_speed < self.min_speed:
            return 0.0
        effective = (self.current_fan_speed - self.min_speed) / (1.0 - self.min_speed)
        return effective * self.max_rpm

    def _calculate_power(self) -> float:
        """
        Electrical power for a 24 V, 80 W (mechanical) fan motor.

        * No-load  → ~6 W  (≈ 0.25 A × 24 V)
        * Full load → ≈ 114 W electrical (80 W mech / 0.7 eff)
        * Power ∝ (speed_norm)²   (air-drag dominates)
        """
        if self.current_fan_speed < self.min_speed:
            return 0.0

        speed_norm = (self.current_fan_speed - self.min_speed) / (1.0 - self.min_speed)
        speed_norm = max(speed_norm, 0.0)

        P_no_load   = 6.0                     # ~0.25 A × 24 V
        P_full_load = 80.0 / 0.70             # ≈ 114 W electrical
        power = P_no_load + (P_full_load - P_no_load) * (speed_norm ** 2)

        return float(power)


# ------------------------------------------------------------------------- #
#  QUICK TEST
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    env = FanControlEnv(outdoor_conc=35.0)  # clean outdoor air
    env.current_conc = 300.0
    env.current_fan_speed = 1.0
    obs = env.reset()
    print("Initial obs:", obs)

    print("\nFan ON → AQI approaches outdoor level (35 µg/m³):")
    for i in range(1000):
        obs, reward, terminated, info = env.step(Action.DO_NOTHING)
        if i % 20 == 0:
            env.render()
        if terminated:
            break

    print("\nPower check at different speeds:")
    for s in [0.0, 0.5, 0.62, 0.8, 1.0]:
        env.current_fan_speed = s
        rpm = env._get_actual_rpm()
        p = env._calculate_power()
        print(f"Speed {s:.2f} → {int(rpm):5d} RPM → {p:5.1f} W")