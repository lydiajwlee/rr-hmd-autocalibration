# Reality Roost HMD Autocalibration

One-time automatic HMD calibration pipeline for Reality Roost, a VR co-location system. An external camera detects the entrance anchor and headset ArUco markers once, computes the headset pose in room space, and transmits that pose to Unity via OSC. The camera pipeline then stops and Quest inside-out tracking takes over.

---

## Pipeline Overview
ZED Camera
→ ArUco detection: HMD 0 and the configured wall anchor(s)
→ Independent HMD pose estimation from each detected wall anchor
→ Position and quaternion-aware rotation averaging when using multiple anchors
→ Coordinate conversion: OpenCV → Unity
→ OSC transmission (osc_sender.py)
→ One-time Unity XR Rig placement
→ Passthrough disabled and VR scene enabled
→ Quest inside-out tracking

---

## Marker Setup

**Dictionary:** DICT_6X6_250

**Marker Sizes**

- Wall anchors 100 and 101: 179 mm (`0.179 m`)
- HMD marker 0: 96 mm (`0.096 m`)

**ID Convention**
- ID 0-99 → HMD markers
- IDs 100 and 101 → Fixed entrance wall markers

**Room Coordinate System (Unity convention)**
- Origin (0,0,0): Center of the railing-enclosed space
- +Z: Toward the window
- +Y: Up (toward the ceiling)
- +X: Right (when facing the window)

**Coordinate Conversion**
- OpenCV → Unity rotation matrix:
```python
[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
```
- Rotation displayed in Euler angles for readability; converted to quaternion in `osc_sender.py` via scipy

---

## Project Structure
**src/**
- `aruco_detector.py` — Camera connection and simultaneous two-anchor/HMD detection
- `pose_calculator.py` — Per-anchor world pose calculation and pose fusion
- `osc_sender.py` — Converts pose to quaternion, sends via OSC to Unity
- `main.py` — Entry point

**unity/Assets/Scripts/**
- `OneShotCalibrationReceiver.cs` — Applies the first OSC pose directly, with no lerp/slerp
- `CalibrationSceneTransition.cs` — Switches from passthrough to the VR scene when calibration completes

**tests/**
- `test_zed_connection.py` — Verify ZED camera connection and K matrix
- `test_heart.py` — ArUco detection test, renders heart above detected marker
- `test_relative_pose.py` — Relative pose between anchor (ID 100) and HMD (ID 0)
- `test_roomspace_hmd.py` - Real-world pose of HMD marker relative to a fixed anchor marker
- `captures/` — Screenshots captured during testing

---

## Dependencies
opencv-python

numpy

scipy

python-osc

pyzed (installed via ZED SDK)

---

## Setup

```bash
git clone https://github.com/lydiajwlee/rr-hmd-autocalibration
cd rr-hmd-autocalibration
pip install -r requirements.txt
```

Note: `pyzed` must be installed separately via ZED SDK:
```bash
cd "C:\Program Files (x86)\ZED SDK"
python get_python_api.py
```

---

## How to Run

**Test ZED connection:**
```bash
python tests/test_zed_connection.py
```

**Test ArUco detection (heart overlay):**
```bash
python tests/test_heart.py
```

**Test relative pose between anchor and HMD marker:**
```bash
python tests/test_relative_pose.py
```

**Test real-world pose of HMD marker relative to a fixed anchor marker:**
```bash
python test/test_roomspace_hmd.py
```

Press `SPACE` to capture screenshot. Press `Q` to quit.

## Unity Setup

1. Copy `unity/Assets/Scripts` into the Unity project.
2. Add `OneShotCalibrationReceiver` to a scene object and assign the XR Rig and its tracked headset camera.
3. Bind the OSC `/markers` message fields to `ReceivePose`. The expected payload is marker ID, position XYZ, quaternion XYZW, and timestamp.
4. Add `CalibrationSceneTransition`, assign the passthrough component and VR scene root, then connect the receiver's **On Calibration Complete** event to `TransitionToVr`.

The receiver solves the XR Rig origin from the current tracked-head offset, applies the pose immediately, and rejects all later calibration messages.

The hardcoded world positions and orientations for anchors 100 and 101 live in
`src/pose_calculator.py`. The current test configuration in
`src/aruco_detector.py` enables only marker 101, at surveyed position
`(-37, 95.5, -97)` inches and facing Unity `+Z` (the window). These values must
match the marker's measured installation pose.

---
*README last updated: June 18, 2026*
