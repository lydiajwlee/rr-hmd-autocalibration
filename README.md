# Reality Roost HMD Autocalibration

## Current status

Automatic room alignment works in Unity's `AutocalibrationTest` scene. The cube
tracks the headset marker, and Roost Rails locks into approximate alignment
with the physical railing. Controllers placed on the physical corners still
show a small discrepancy in VR. Its cause and magnitude are unconfirmed.

**Keep `main` as the working baseline. Develop and test further rail-alignment
fixes only on `rail-alignment-accuracy` (`git switch rail-alignment-accuracy`).**

## Current pipeline

1. The webcam detects wall markers `100` and `101` and headset marker `0` in
   the same frame.
2. Python uses the eight wall-marker corners to estimate the camera pose.
   It refines that pose and a shared printed-square width together, keeping
   the surveyed anchor centers fixed. This reduced depth-dependent height drift.
3. Python transforms the headset marker into room coordinates and streams
   accepted poses to Unity at `/markers` on `127.0.0.1:9001`. The default
   continuous test accepts anchor reprojection error up to `5.0 px`.
4. Unity pairs each timestamped marker position with the XR eye pose for that
   moment. It fits the room position, yaw, and sticker-to-eye offset, then
   places Roost Rails once the fit passes. The cube continues tracking.

## Coordinate and marker configuration

Unity room coordinates:

- Origin `(0, 0, 0)`: center of the railing-enclosed space
- `+X`: right while facing the window
- `+Y`: upward
- `+Z`: toward the window

ArUco configuration:

- Dictionary: `DICT_6X6_250`
- Marker `0`: HMD marker, black-square side length `32 mm`
- Markers `100` and `101`: wall anchors, black-square side length `179 mm`

Current surveyed anchor assumptions:

| Marker | Position in inches | Position in meters | Orientation |
| --- | --- | --- | --- |
| 100 | `(+37, 95.5, -97)` | `(+0.9398, 2.4257, -2.4638)` | Upright; printed face toward Unity `+Z` |
| 101 | `(-37, 95.5, -97)` | `(-0.9398, 2.4257, -2.4638)` | Upright; printed face toward Unity `+Z` |

All positions refer to the center of the detected black square. If marker 100
is physically rotated or mirrored rather than merely positioned at positive
X, update its world-corner orientation in `src/pose_calculator.py`.

## Why joint solvePnP

For each anchor, the code generates four known world-space corners from its
surveyed center, orientation, and physical size. ArUco supplies the matching
four image-space corners. OpenCV then receives eight 3D-to-2D correspondences:

```python
success, rvec, tvec = cv2.solvePnP(
    eight_world_corners,
    eight_image_corners,
    K,
    dist,
)
```

OpenCV returns the transform from world coordinates into camera coordinates:

```text
camera_T_world
```

The application needs the opposite direction, so it inverts the matrix:

```python
world_T_camera = np.linalg.inv(camera_T_world)
world_T_hmd = world_T_camera @ camera_T_hmd
```

Joint solving uses the full distance between the two anchors as a geometric
baseline and produces one internally consistent camera pose. This is generally
more stable than calculating two independent HMD poses and averaging them.

A low reprojection error confirms that the detected image corners agree with
the configured anchor geometry. It does not prove that the surveyed world
positions themselves are correct; a consistent survey offset can still shift
the final HMD pose.

## Installation

```bash
git clone https://github.com/lydiajwlee/rr-hmd-autocalibration
cd rr-hmd-autocalibration
python3 -m pip install -r requirements.txt
```

Core dependencies are OpenCV, NumPy, SciPy, and `python-osc`. ZED support also
requires the Python API supplied by the installed ZED SDK.

## Running the current pipeline

From the repository root:

```bash
python3 src/main.py
```

The webcam preview and OSC stream stay active during testing. Press `Q` to stop.

The overlay contains:

```text
HMD world XYZ
HMD world XYZW
Anchor reprojection error
```

Before testing, verify that:

- Both anchors are fixed at their configured center positions.
- Both anchors are upright and face Unity `+Z`.
- Markers `0`, `100`, and `101` are simultaneously visible.
- The terminal reports the same resolution stored in the intrinsics file.

## Unity integration

In `AutocalibrationTest`, the `CalibrationManager` GameObject hosts
`OSCPoseReceiver`. Its **Roost Rails** reference identifies the prefab to place,
and **Pose Target** identifies the marker cube. The receiver creates
`RailRoomAlignment` in code.

Start the Python sender and enter Play mode with XR floor tracking active.
Keep all three markers visible while stepping sideways and turning your head.
The solver starts fitting at eight distinct poses and retains up to 48. Look
for `Roost Rails aligned. Pose fit RMS … m.` in the Unity Console. The
`0.025 m` RMS threshold measures agreement between the webcam and XR headset
poses. Physical corner accuracy is checked separately against the railing.

## Project structure

```text
src/
  aruco_detector.py       Camera input, ArUco detection, per-marker solvePnP
  pose_calculator.py      Joint extrinsics and HMD world-pose calculations
  osc_sender.py           OSC pose serialization and UDP transmission
  main.py                 Current pipeline entry point

webcam_calibration/
  capture.py              ChArUco image capture at the requested camera mode
  calibrate.py            Intrinsic calibration and resolution metadata
  intrinsics.npz          Current saved webcam calibration

unityscripts/
  OSCPoseReceiver.cs        OSC receiver and XR pose synchronization
  RailRoomAlignment.cs      Rigid room alignment solver

tests/
  test_pose_calculator.py Synthetic pose/extrinsics validation
  test_relative_pose.py   Historical single-anchor manual test
  test_roomspace_hmd.py   Historical room-space manual test
  test_webcam_heart.py    Basic webcam ArUco visualization
```
