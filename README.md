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

## Camera intrinsic calibration

Intrinsics convert detected pixels into camera-space rays and distances. They
must be generated at the same resolution and camera mode used during HMD
detection.

Capture a new ChArUco image set:

```bash
python3 webcam_calibration/capture.py
```

- Press `Space` to capture an image.
- Capture 20–30 sharp images across the center, edges, and corners.
- Include several board distances and angles.
- Press `Q` to finish.

Calculate and save the intrinsics:

```bash
python3 webcam_calibration/calibrate.py
```

The result is written to `webcam_calibration/intrinsics.npz` with `K`, `dist`,
and `image_size`. The current checked-in calibration is `1920×1080`. Detection
rejects legacy files without resolution metadata and files whose calibration
resolution differs from the live camera resolution.

A reprojection error below `1.0 px` is a useful target. It measures intrinsic
fit quality, not wall-anchor survey accuracy.

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

The repository includes two package-agnostic Unity components:

- `unity/Assets/Scripts/OneShotCalibrationReceiver.cs`
- `unity/Assets/Scripts/CalibrationSceneTransition.cs`

Setup:

1. Copy `unity/Assets/Scripts` into the Unity project.
2. Add `OneShotCalibrationReceiver` to a scene object.
3. Assign the XR Rig and tracked headset camera.
4. Bind the OSC `/markers` payload to `ReceivePose`. The payload is marker ID,
   position XYZ, quaternion XYZW, and timestamp.
5. Add `CalibrationSceneTransition` and assign the passthrough component and
   VR scene root.
6. Connect the receiver's calibration-complete event to `TransitionToVr`.

The receiver compensates for the Quest camera's current offset inside the XR
Rig, applies the calibration once, and rejects subsequent messages.

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
```

## Evolution archive

The project history intentionally preserves earlier approaches. The commits
below are practical checkpoints, not claims that every older experiment is
production-ready. Historical versions can contain known coordinate, scaling,
or stability limitations that motivated the next iteration.

| Stage | Date | Checkpoint | Pipeline represented | Why it changed |
| --- | --- | --- | --- | --- |
| Webcam ArUco proof of concept | 2026-06-12 | [`9f80e16`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/9f80e16) | Detect a webcam marker and render a test overlay | Established basic 2D detection only |
| Single-anchor relative pose | 2026-06-17 | [`75bceca`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/75bceca) | Separate solvePnP poses and `inv(anchor_T) @ hmd_T` | Needed a room/world reference |
| ZED streaming experiment | 2026-06-17 | [`053dec8`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/053dec8) | Connect to and read the ZED stream | Camera connection preceded integrated pose transport |
| Room-space HMD prototype | 2026-06-23 | [`7b2dfb6`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/7b2dfb6) | Apply a hardcoded anchor world pose to the relative HMD transform | Exposed coordinate-convention issues |
| OSC transport | 2026-07-09 | [`7b9d445`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/7b9d445) | Serialize marker pose and transmit it to Unity | Enabled end-to-end integration |
| Modular webcam pipeline | 2026-07-13 | [`d33bcec`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/d33bcec) | Detector, pose calculator, OSC sender, and main entry point | Prepared continuous integrated testing |
| Continuous detection loop | 2026-07-24 | [`b1f11f7`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/b1f11f7) | Continuously detect and update the HMD pose | Replaced by one-time calibration to avoid competing with Quest tracking |
| One-time calibration and Unity transition | 2026-07-30 | [`562a6da`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/562a6da) | Stop after calibration; direct XR Rig placement; passthrough-to-VR event; independent multi-anchor pose fusion | Independent planar estimates could disagree or carry separate biases |
| Correct physical marker sizes and zoom | 2026-08-03 | [`5b02995`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/5b02995) | 179 mm anchors, 96 mm HMD marker, zoomed detection frame | Correct scale and improve distant-marker detection |
| Anchor-relative conversion checkpoint | 2026-08-03 | [`24bb482`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/24bb482) | Revised single-anchor coordinate conversion with an origin-recovery test | Coordinate handedness still depended on physical marker orientation |
| Resolution-aware calibration | 2026-08-03 | [`e6027cd`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/e6027cd) | High-resolution capture, intrinsic resolution metadata, and matching detection mode | Prevented use of 640×480 intrinsics on higher-resolution frames |
| Coordinate alignment checkpoint | 2026-08-03 | [`aa094f2`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/aa094f2) | Align marker-relative axes with the measured Unity room convention | Retained as the latest single-anchor conversion experiment |
| Current 1080p intrinsics | 2026-08-04 | [`3cf275a`](https://github.com/lydiajwlee/rr-hmd-autocalibration/commit/3cf275a) | Thirty-two 1920×1080 ChArUco images and matching intrinsics | Corrected the stale 640×480 calibration used during early testing |
| Joint two-anchor camera extrinsics | Current `main` | `main` | One joint solvePnP for `world_T_camera`, followed by `world_T_camera @ camera_T_hmd` | Current recommended architecture |

## Trying an older pipeline

### Recommended: create a separate worktree

This keeps the current checkout untouched. Replace the name and commit with a
checkpoint from the archive table:

```bash
git worktree add ../rr-hmd-single-anchor 24bb482
cd ../rr-hmd-single-anchor
```

Remove the worktree when finished:

```bash
cd ../rr-hmd-autocalibration
git worktree remove ../rr-hmd-single-anchor
```

### Inspect a commit directly

Only do this with a clean working tree:

```bash
git switch --detach 562a6da
# Test the historical one-time/averaging pipeline.
git switch main
```

### Create a named local archive branch

```bash
git branch archive/single-anchor 24bb482
git branch archive/one-shot-averaging 562a6da
git branch archive/high-resolution-calibration e6027cd
```

These commands create local pointers only. To share an archive branch through
GitHub, push it explicitly after reviewing the historical code:

```bash
git push -u origin archive/single-anchor
```

Commit hashes are the canonical archive references because they are immutable;
branch names can move over time.

## Tests

Run the automated pose tests:

```bash
python3 -m unittest discover -s tests -p 'test_pose_calculator.py' -v
```

The current suite covers single-anchor origin recovery, quaternion-aware
averaging retained for historical compatibility, and synthetic end-to-end
joint extrinsics recovery.

---

README archive updated: August 4, 2026.
