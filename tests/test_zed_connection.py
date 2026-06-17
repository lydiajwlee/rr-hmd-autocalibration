import pyzed.sl as sl
import cv2

CAMERA_IP   = "192.168.50.3"
CAMERA_PORT = 30000

init_params = sl.InitParameters()
init_params.set_from_stream(CAMERA_IP, CAMERA_PORT)

zed = sl.Camera()
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to connect: {status}")
    exit()

print("ZED connected — press Q to quit")

image = sl.Mat()
runtime_params = sl.RuntimeParameters()

while True:
    err = zed.grab(runtime_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Grab error: {err}")
        continue

    zed.retrieve_image(image, sl.VIEW.LEFT)
    frame = image.get_data()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow("ZED Live", frame_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
zed.close()