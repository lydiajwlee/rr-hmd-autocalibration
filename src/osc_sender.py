from pythonosc import udp_client
import time

# ── CONFIG ──────────────────────────────────────────────────────────────────
OSC_IP      = "127.0.0.1" #local
OSC_PORT    = 9000
OSC_ADDRESS = "/markers"
# ────────────────────────────────────────────────────────────────────────────

class OSCSender:
    def __init__(self):
        self.client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        print(f"[osc_sender] Ready → {OSC_IP}:{OSC_PORT}")

    def send_pose(self, marker_id, position, quaternion):
        """
        marker_id:  int
        position:   np.array [x, y, z] in meters (room space)
        quaternion: np.array [qx, qy, qz, qw]
        """
        msg = [
            int(marker_id),
            float(position[0]), float(position[1]), float(position[2]),
            float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3]),
            time.time()
        ]

        self.client.send_message(OSC_ADDRESS, msg)