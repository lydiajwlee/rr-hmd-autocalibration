# Reality Roost HMD Autocalibration

Automatic HMD calibration pipeline for Reality Roost, a VR co-location system. Uses external ZED cameras to detect ArUco markers on headsets and compute real-time pose in room space, transmitted to Unity via OSC.

---

## Pipeline Overview
ZED Camera
→ ArUco marker detection (aruco_detector.py)
→ Relative pose estimation (anchor marker as origin)
→ Coordinate conversion: OpenCV → Unity
→ OSC transmission (osc_sender.py)
→ Unity XR Rig update

---

## Marker Setup

**Dictionary:** DICT_6X6_250

**Marker Size:** 0.1m (TBD)

**ID Convention**
- ID 0-99 → HMD markers
- ID 100+ → Fixed anchor markers

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
- `aruco_detector.py` — ZED camera connection, ArUco detection, pose estimation
- `osc_sender.py` — Converts pose to quaternion, sends via OSC to Unity
- `main.py` — Entry point

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

---
*README last updated: June 18, 2026*