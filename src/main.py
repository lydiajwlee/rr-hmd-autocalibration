from aruco_detector import run
from osc_sender import OSCSender

sender = OSCSender()

def on_pose_detected(rvec, tvec):
    sender.send_pose(rvec, tvec)

run(on_pose_detected)