import numpy as np
import time
from pythonosc import udp_client
from scipy.spatial.transform import Rotation

OSC_IP   = "127.0.0.1"
OSC_PORT = 9000

CV_TO_UNITY = np.array([
    [0,  0, 1],
    [0,  1, 0],
    [-1, 0, 0]
], dtype=np.float64)

class OSCSender:
    def __init__(self):
        self.client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    def send_pose(self, marker_id, position, quaternion, timestamp):
        t_unity    = CV_TO_UNITY @ position
        R_unity    = CV_TO_UNITY @ Rotation.from_quat(quaternion).as_matrix() @ CV_TO_UNITY.T
        quat_unity = Rotation.from_matrix(R_unity).as_quat()

        msg = [
            marker_id,
            float(t_unity[0]), float(t_unity[1]), float(t_unity[2]),
            float(quat_unity[0]), float(quat_unity[1]), float(quat_unity[2]), float(quat_unity[3]),
            timestamp
        ]
        self.client.send_message("/markers", msg)
        print(f"[osc_sender] Sent — ID: {marker_id}, pos: {t_unity}, quat: {quat_unity}")