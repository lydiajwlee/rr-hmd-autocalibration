# Reality Roost HMD Autocalibration

Reality Roost HMD Autocalibration aligns a physical Meta Quest headset with a
shared Unity room. The project began as a continuous external-camera tracking
experiment and evolved into a one-time entrance calibration: the external
camera establishes the HMD's initial world pose, Unity places the XR Rig, and
Quest inside-out tracking takes over.

This README documents both the current pipeline and the earlier experimental
pipelines preserved in Git history.

## Current pipeline

```text
User starts in Quest passthrough
        ↓
External camera detects wall anchors 100 and 101
        ↓
Joint solvePnP uses all eight anchor corners
        ↓
Invert camera_T_world to obtain world_T_camera
        ↓
Camera detects HMD marker 0 and solves camera_T_hmd
        ↓
world_T_hmd = world_T_camera @ camera_T_hmd
        ↓
Python sends one OSC world pose to Unity
        ↓
Unity places the XR Rig and exits passthrough
        ↓
Quest inside-out tracking takes over
```

The two anchors are not solved independently and averaged in the current
pipeline. Their eight corners jointly constrain one camera-to-world extrinsic
transform. The camera window reports the resulting HMD world pose and the
joint anchor reprojection error. A pose is accepted only when that error is at
most `2.0 px`.

## Coordinate and marker configuration

Unity room coordinates:

- Origin `(0, 0, 0)`: center of the railing-enclosed space
- `+X`: right while facing the window
- `+Y`: upward
- `+Z`: toward the window

ArUco configuration:

- Dictionary: `DICT_6X6_250`
- Marker `0`: HMD marker, black-square side length `100 mm`
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

## User experience

1. The user puts on the Quest while Unity displays passthrough.
2. The user walks toward the entrance.
3. The external camera waits until markers `0`, `100`, and `101` are visible in
   the same frame.
4. Python solves the camera extrinsics from anchors `100` and `101`.
5. Frames with anchor reprojection error above `2.0 px` are rejected.
6. Python transforms marker `0` into Unity world coordinates and sends the
   accepted pose through OSC.
7. Unity applies the pose immediately, with no lerp or slerp.
8. Unity enables the VR scene, disables passthrough, and ignores later
   calibration messages.
9. Quest inside-out tracking handles all subsequent headset motion.

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

The current test configuration keeps the camera window open and continuously
updates the pose overlay, while OSC is sent only once. Press `Q` to stop.

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

TBD

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

unity/Assets/Scripts/
  OneShotCalibrationReceiver.cs
  CalibrationSceneTransition.cs

tests/
  test_pose_calculator.py Synthetic pose/extrinsics validation
  test_relative_pose.py   Historical single-anchor manual test
  test_roomspace_hmd.py   Historical room-space manual test
  test_webcam_heart.py    Basic webcam ArUco visualization

---

README archive updated: August 4, 2026.
