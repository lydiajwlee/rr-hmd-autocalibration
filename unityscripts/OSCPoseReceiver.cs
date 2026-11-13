using UnityEngine;
using extOSC;
using Unity.XR.CoreUtils;
using UnityEngine.XR;

public class OSCPoseReceiver : MonoBehaviour
{
    [Header("OSC Settings")]
    public int port = 9001;
    public string address = "/markers";
    public int expectedMarkerId = 0;

    [Header("References")]
    public Transform poseTarget;
    public Transform calibrationOffset;
    public Transform xrCamera; // Assign XR Origin/Camera Offset/Main Camera
    public Transform roostRails;
    private RailRoomAlignment railAlignment = new RailRoomAlignment();
    private bool railsAligned;
    public bool hasCalibrated;
    [SerializeField] private string railStatus = "Waiting for marker OSC packets";
    private XROrigin rig;
    private Transform floor;
    private Pose lastFloor;
    private float nextReport;
    private struct EyeSample { public long time; public Pose pose; }
    private readonly EyeSample[] history = new EyeSample[256];
    private int nextEye, eyeCount;

    [Header("Smoothing")]
    [Range(0f, 1f)]
    public float positionSmoothSpeed = 0.1f;

    [Range(0f, 1f)]
    public float rotationSmoothSpeed = 0.1f;

    private OSCReceiver receiver;
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    private bool hasTarget;

    void Start()
    {
        if (roostRails == null && poseTarget == null && calibrationOffset == null)
        {
            Debug.LogError(
                "[OSCPoseReceiver] Assign a Pose Target or both Calibration Offset and XR Camera."
            );
            enabled = false;
            return;
        }

        receiver = gameObject.AddComponent<OSCReceiver>();
        receiver.LocalPort = port;
        receiver.Bind(address, OnMarkerReceived);

        Debug.Log(
            $"[OSCPoseReceiver] Listening on port {port}, address {address}"
        );
    }

    void OnMarkerReceived(OSCMessage message)
    {
        if (message.Values.Count < 9)
        {
            Debug.LogWarning("[OSCPoseReceiver] Incomplete message received");
            return;
        }

        int id = message.Values[0].IntValue;
        if (id != expectedMarkerId)
            return;

        Vector3 detectedHmdPosition = new Vector3(
            message.Values[1].FloatValue,
            message.Values[2].FloatValue,
            message.Values[3].FloatValue
        );

        Quaternion detectedHmdRotation = new Quaternion(
            message.Values[4].FloatValue,
            message.Values[5].FloatValue,
            message.Values[6].FloatValue,
            message.Values[7].FloatValue
        ).normalized;

        if (roostRails != null)
        {
            if (!railsAligned && floor != null && eyeCount > 0)
                TryAlignRails(message, detectedHmdPosition);
            if (poseTarget != null)
            {
                poseTarget.SetPositionAndRotation(roostRails.position + roostRails.rotation * detectedHmdPosition,
                                                  roostRails.rotation * detectedHmdRotation);
            }
            return;
        }

        if (poseTarget != null)
        {
            poseTarget.SetPositionAndRotation(detectedHmdPosition, detectedHmdRotation);
            return;
        }

        /*
         * Find the camera's tracking pose relative to CalibrationOffset.
         *
         * cameraWorld = calibrationOffset * cameraLocal
         *
         * Therefore:
         * calibrationOffset = detectedHmdWorld * inverse(cameraLocal)
         */
        ResolveCamera();
        if (xrCamera == null)
            return;
        Vector3 cameraLocalPosition =
            calibrationOffset.InverseTransformPoint(xrCamera.position);

        Quaternion cameraLocalRotation =
            Quaternion.Inverse(calibrationOffset.rotation) *
            xrCamera.rotation;

        targetRotation =
            detectedHmdRotation *
            Quaternion.Inverse(cameraLocalRotation);

        targetPosition =
            detectedHmdPosition -
            targetRotation * cameraLocalPosition;

        hasTarget = true;
        Debug.Log($"[OSCPoseReceiver] recv pos={detectedHmdPosition:F3}");
    }

    void TryAlignRails(OSCMessage message, Vector3 roomMarker)
    {
        bool timed = message.Values.Count >= 14;
        long capture = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        Pose eye = history[(nextEye + history.Length - 1) % history.Length].pose;
        if (timed)
        {
            capture = (long)message.Values[9].IntValue * 1000 + message.Values[10].IntValue;
            if (!TryGetEye(capture, out eye))
            {
                railStatus = "Waiting for XR history covering the webcam frame";
                return;
            }
            roomMarker = new Vector3(message.Values[11].FloatValue, message.Values[12].FloatValue,
                                     message.Values[13].FloatValue);
        }
        railsAligned = railAlignment.TryAdd(roomMarker, eye.position, eye.rotation, 0,
                                            out Pose candidate, out float rms);
        railStatus = $"{railAlignment.Status} ({railAlignment.SampleCount} poses)";
        if (railsAligned)
        {
            roostRails.SetPositionAndRotation(floor.TransformPoint(candidate.position),
                                              floor.rotation * candidate.rotation);
            hasCalibrated = true;
            Debug.Log($"[OSCPoseReceiver] Roost Rails aligned. Pose fit RMS {rms:F3} m.");
        }
    }

    bool TryGetEye(long capture, out Pose eye)
    {
        EyeSample after = history[(nextEye + history.Length - 1) % history.Length];
        eye = after.pose;
        if (capture >= after.time)
            return capture - after.time <= 40;
        for (int i = 1; i < eyeCount; i++)
        {
            EyeSample before = history[(nextEye + history.Length - 1 - i) % history.Length];
            if (before.time <= capture)
            {
                if (after.time - before.time > 150) return false;
                float t = (float)(capture - before.time) / (after.time - before.time);
                eye = new Pose(Vector3.Lerp(before.pose.position, after.pose.position, t),
                               Quaternion.Slerp(before.pose.rotation, after.pose.rotation, t));
                return true;
            }
            after = before;
        }
        return false;
    }

    void LateUpdate()
    {
        if (roostRails == null) return;
        if (railsAligned)
        {
            return;
        }
        ResolveCamera();
        if (rig == null && xrCamera != null) rig = xrCamera.GetComponentInParent<XROrigin>();
        if (rig == null || rig.CameraFloorOffsetObject == null)
            railStatus = "Waiting for the XR rig camera";
        else if ((rig.CurrentTrackingOriginMode & TrackingOriginModeFlags.Floor) == 0)
            railStatus = $"Waiting for XR floor tracking (current mode {rig.CurrentTrackingOriginMode})";
        else
        {
            floor = rig.CameraFloorOffsetObject.transform;
            if (eyeCount > 0 && (Vector3.Distance(floor.position, lastFloor.position) > 0.01f ||
                                 Quaternion.Angle(floor.rotation, lastFloor.rotation) > 1f))
            {
                eyeCount = 0;
                railAlignment = new RailRoomAlignment();
            }
            lastFloor = new Pose(floor.position, floor.rotation);
            var head = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (head.isValid && (!head.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked) || tracked))
            {
                history[nextEye] = new EyeSample {
                    time = System.DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    pose = new Pose(floor.InverseTransformPoint(xrCamera.position),
                                    Quaternion.Inverse(floor.rotation) * xrCamera.rotation)
                };
                nextEye = (nextEye + 1) % history.Length;
                eyeCount = Mathf.Min(eyeCount + 1, history.Length);
                if (eyeCount == 1) railStatus = "Waiting for marker OSC packets";
            }
            else
            {
                if (eyeCount > 0) railAlignment = new RailRoomAlignment();
                eyeCount = 0;
                railStatus = "Waiting for headset tracking";
            }
        }
        if (rig == null || (rig.CurrentTrackingOriginMode & TrackingOriginModeFlags.Floor) == 0)
        {
            if (eyeCount > 0) railAlignment = new RailRoomAlignment();
            eyeCount = 0;
        }
        if (Time.unscaledTime >= nextReport)
        {
            Debug.Log($"[OSCPoseReceiver] Rails: {railStatus}");
            nextReport = Time.unscaledTime + 5f;
        }
    }

    void ResolveCamera()
    {
        if (xrCamera == null && Camera.main != null)
            xrCamera = Camera.main.transform;
    }

    void Update()
    {
        if (!hasTarget)
            return;

        calibrationOffset.position = Vector3.Lerp(
            calibrationOffset.position,
            targetPosition,
            positionSmoothSpeed
        );

        calibrationOffset.rotation = Quaternion.Slerp(
            calibrationOffset.rotation,
            targetRotation,
            rotationSmoothSpeed
        );
    }
}
