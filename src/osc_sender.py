from pythonosc import udp_client
import time

# ── CONFIG ──────────────────────────────────────────────────────────────────
OSC_IP      = "127.0.0.1" #local
OSC_PORT    = 9001
OSC_ADDRESS = "/markers"
# ────────────────────────────────────────────────────────────────────────────

class OSCSender:
    def __init__(self):
        self.client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        print(f"[osc_sender] Ready at {OSC_IP}:{OSC_PORT}")

    def send_pose(self, marker_id, position, quaternion, frame_time=None, frame_position=None):
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

        if frame_time is not None and frame_position is not None:
            # Keep the legacy cube packet prefix. Two OSC ints preserve millisecond
            # precision, which a float32 Unix timestamp cannot represent.
            seconds, milliseconds = divmod(round(frame_time * 1000), 1000)
            msg.extend([seconds, milliseconds, *map(float, frame_position)])
        self.client.send_message(OSC_ADDRESS, msg)
