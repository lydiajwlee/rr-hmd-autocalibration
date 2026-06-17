import pyzed.sl as sl

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000

init_params = sl.InitParameters()
init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

zed = sl.Camera()
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to connect: {status}")
    exit()

print("ZED connected successfully!")

calib = zed.get_camera_information().camera_configuration.calibration_parameters
print(f"fx: {calib.left_cam.fx}")
print(f"fy: {calib.left_cam.fy}")
print(f"cx: {calib.left_cam.cx}")
print(f"cy: {calib.left_cam.cy}")

zed.close()
print("ZED closed.")