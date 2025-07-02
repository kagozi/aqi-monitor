# mqtt_action_simulator.py (trigger action on data receive)
import time
import json
import random
import paho.mqtt.client as mqtt

# MQTT setup
BROKER_IP = "10.0.0.194"
DATA_TOPIC = "sensor/data"
ACTION_TOPIC = "agent/action"

ACTIONS = ["increase", "decrease", "nothing"]

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with result code", rc)
    client.subscribe(DATA_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print("Received state:", payload)

        # Choose a random action on message receive
        action = random.choice(ACTIONS)
        action_payload = {"action": action}
        client.publish(ACTION_TOPIC, json.dumps(action_payload))
        print(f"Published action: {action_payload}")

    except Exception as e:
        print("Error decoding or publishing:", e)

# MQTT client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_IP)

client.loop_forever()
