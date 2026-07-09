from pythonosc import udp_client

OSC_IP = "127.0.0.1"
OSC_PORT = 9000

class OSCSender:
    def __init__(self):
        self.client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

    def send_pose(self, marker_id, position, quaternion, timestamp):
        """
        position: (x, y, z)
        quaternion: (qx, qy, qz, qw)
        """
        msg = [
            marker_id,
            position[0], position[1], position[2], 
            quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            timestamp
        ]

        self.client.send_message("/markers", msg)