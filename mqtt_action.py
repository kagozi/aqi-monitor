# import time
# import json
# import sqlite3
# import numpy as np
# import paho.mqtt.client as mqtt
# from stable_baselines3 import PPO, A2C, DQN
# from simulated.environment import FanControlGymEnv
# from simulated.physics import FanControlEnv

# # MQTT setup
# BROKER_IP = "10.0.0.193"
# DATA_TOPIC = "sensor/data"
# ACTION_TOPIC = "agent/action"

# # 1. Load Environment and Model
# fan_env = FanControlEnv()
# env = FanControlGymEnv(fan_env=fan_env)

# # Load PPO model (same as training)
# CHECKPOINT_DIR_PPO = "ppo/train"
# CHECKPOINT_DIR_A2C = "a2c/train"
# model_a2c = A2C.load(f"{CHECKPOINT_DIR_PPO}/model_100000")

# # SQLite database setup
# conn = sqlite3.connect("fan_metrics.db", check_same_thread=False)
# cursor = conn.cursor()
# cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
#     timestamp TEXT,
#     aqi INTEGER,
#     tvoc INTEGER,
#     eco2 INTEGER,
#     voltage REAL,
#     fan_speed REAL,
#     action TEXT,
#     reward REAL,
#     energy REAL,
#     algorithm TEXT,
# )''')
# conn.commit()

# # Energy tracking
# last_energy = 0.0
# TIMESTEP_S = 1.0

# def on_connect(client, userdata, flags, rc):
#     print("Connected to MQTT Broker with result code", rc)
#     client.subscribe(DATA_TOPIC)

# def on_message(client, userdata, msg):
#     global last_energy
#     try:
#         payload = json.loads(msg.payload.decode())
#         print("Received state:", payload)

#         aqi = payload.get("aqi", 0)
#         tvoc = payload.get("tvoc", 0)
#         eco2 = payload.get("eco2", 0)
#         voltage = payload.get("voltage", 0.0)
#         fan_speed = payload.get("fan_speed", 0.0)

#         # Build observation as in training
#         obs = np.array([aqi, fan_speed], dtype=np.float32)
#         print(f"Observation: {obs}")

#         # Get action from PPO model
#         action, _ = model_a2c.predict(obs)

#         # Step environment locally to get reward and info (voltage, energy)
#         _, reward, _, _, info = env.step(action)

#         # Update energy
#         last_energy += info.get("energy", 0.0)

#         # Map numeric action to string
#         actions = ["decrease", "nothing", "increase"]
#         action_str = actions[action]

#         # Publish action
#         action_payload = {"action": action_str}
#         client.publish(ACTION_TOPIC, json.dumps(action_payload))
#         print(f"Published action: {action_payload}")

#         # Log metrics to SQLite
#         cursor.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
#             time.strftime("%Y-%m-%d %H:%M:%S"),
#             aqi,
#             tvoc,
#             eco2,
#             info.get("voltage", voltage),
#             fan_speed,
#             action_str,
#             reward,
#             last_energy,
#             "A2C"  # Algorithm name
#         ))
#         conn.commit()

#     except Exception as e:
#         print("Error decoding or logging:", e)

# # MQTT client setup
# client = mqtt.Client()
# client.on_connect = on_connect
# client.on_message = on_message
# client.connect(BROKER_IP)

# print("Starting MQTT listener with PPO agent...")
# client.loop_forever()
import time
import json
import sqlite3
import numpy as np
import paho.mqtt.client as mqtt
from stable_baselines3 import A2C, PPO, DQN
from simulated.environment import FanControlGymEnv
from simulated.physics import FanControlEnv

# MQTT setup
BROKER_IP = "10.0.0.189"
DATA_TOPIC = "sensor/data"
ACTION_TOPIC = "agent/action"

# 1. Load Environment and Model
fan_env = FanControlEnv()
env = FanControlGymEnv(fan_env=fan_env)

# Load A2C model
CHECKPOINT_DIR_A2C = "a2c/train"
CHECKPOINT_DIR_DQN = "dqn/train"
CHECKPOINT_DIR_PPO = "ppo/train"
model_a2c = A2C.load(f"{CHECKPOINT_DIR_A2C}/model_100000")
model_ppo = PPO.load(f"{CHECKPOINT_DIR_PPO}/model_100000")
model_dqn = DQN.load(f"{CHECKPOINT_DIR_DQN}/model_100000")

# SQLite database setup
conn = sqlite3.connect("fan_metrics.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS metrics (
    timestamp TEXT,
    aqi INTEGER,
    tvoc INTEGER,
    eco2 INTEGER,
    voltage REAL,
    fan_speed REAL,
    action TEXT,
    reward REAL,
    energy REAL,
    algorithm TEXT
)''')
conn.commit()

# Energy tracking
last_energy = 0.0
TIMESTEP_S = 1.0

# Timer setup (30 minutes = 1800 seconds)
START_TIME = time.time()
RUN_DURATION = 1800  # seconds

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code", rc)
    client.subscribe(DATA_TOPIC)

def on_message(client, userdata, msg):
    global last_energy
    try:
        payload = json.loads(msg.payload.decode())
        print("Received state:", payload)

        aqi = payload.get("aqi", 0)
        tvoc = payload.get("tvoc", 0)
        eco2 = payload.get("eco2", 0)
        voltage = payload.get("voltage", 0.0)
        fan_speed = payload.get("fan_speed", 0.0)

        # Build observation
        obs = np.array([aqi, fan_speed], dtype=np.float32)
        print(f"Observation: {obs}")

        # Get action from A2C model
        action, _ = model_dqn.predict(obs)

        # Step environment locally to get reward and info
        _, reward, _, _, info = env.step(action)

        # Update energy
        last_energy += info.get("energy", 0.0)

        # Map numeric action to string
        actions = ["decrease", "nothing", "increase"]
        action_str = actions[action]

        # Publish action
        action_payload = {"action": action_str}
        client.publish(ACTION_TOPIC, json.dumps(action_payload))
        print(f"Published action: {action_payload}")

        # Log metrics
        cursor.execute("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            aqi,
            tvoc,
            eco2,
            info.get("voltage", voltage),
            fan_speed,
            action_str,
            reward,
            last_energy,
            "DQN"
        ))
        conn.commit()

        # Stop after 2 hours
        if time.time() - START_TIME > RUN_DURATION:
            print("Test duration completed (2 hours). Exiting...")
            client.loop_stop()
            client.disconnect()
            conn.close()
            exit(0)

    except Exception as e:
        print("Error decoding or logging:", e)

# MQTT client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_IP)

print("Starting MQTT listener for 2-hour test...")
client.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    client.loop_stop()
    client.disconnect()
    conn.close()
    print("Clean exit.")
