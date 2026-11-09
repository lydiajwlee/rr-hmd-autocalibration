using UnityEngine;
using extOSC;

/// Receives live HMD-marker poses from Python (/markers) and moves the rig every frame
/// so that the tracked head camera ends up at the external-camera pose.
///
/// OSC message (from osc_sender.py):
///   [int id, float px, py, pz, float qx, qy, qz, qw, float timestamp]
///   Position/rotation are already in Unity room coordinates (left-handed, meters).
public class LiveMarkerPoseReceiver : MonoBehaviour
{
    [Header("OSC")]
    public OSCReceiver receiver;          // OSCReceiver component, Local Port = 9000
    public string address = "/markers";
    public int hmdMarkerId = 0;

    [Header("Rig")]
    public Transform rigRoot;             // CalibrationOffset (parent of XR Origin)
    public Transform head;                // Main Camera under Camera Offset
    [Tooltip("Rotate the rig only around world Y. Keeps Quest's own gravity/tilt estimate.")]
    public bool yawOnly = true;

    [Header("Marker -> head offset (in marker-local axes)")]
    public Vector3 headOffsetPosition = Vector3.zero;   // meters
    public Vector3 headOffsetEuler = Vector3.zero;      // degrees

    [Header("Smoothing")]
    [Range(0f, 0.95f)] public float smoothing = 0f;      // 0 = raw, higher = smoother but laggier

    [Header("Debug")]
    public bool logPoses = false;
    public float staleAfterSeconds = 0.5f;

    Vector3 targetPos;
    Quaternion targetRot = Quaternion.identity;
    bool hasPose;
    float lastReceiveTime;

    void Start()
    {
        if (receiver == null) receiver = GetComponent<OSCReceiver>();
        receiver.Bind(address, OnMarkers);
    }

    void OnMarkers(OSCMessage message)
    {
        var v = message.Values;
        if (v.Count < 8) return;
        if (v[0].IntValue != hmdMarkerId) return;

        var markerPos = new Vector3(v[1].FloatValue, v[2].FloatValue, v[3].FloatValue);
        var markerRot = Normalize(new Quaternion(
            v[4].FloatValue, v[5].FloatValue, v[6].FloatValue, v[7].FloatValue));

        // marker pose -> head pose (fixed mounting offset)
        targetPos = markerPos + markerRot * headOffsetPosition;
        targetRot = markerRot * Quaternion.Euler(headOffsetEuler);

        hasPose = true;
        lastReceiveTime = Time.time;

        if (logPoses)
            Debug.Log($"[OSC] pos={targetPos:F3} euler={targetRot.eulerAngles:F1}");
    }

    void LateUpdate()
    {
        if (!hasPose || rigRoot == null || head == null) return;
        if (Time.time - lastReceiveTime > staleAfterSeconds) return;   // marker lost: hold last rig pose

        // Head pose relative to the rig root (what Quest tracking currently reports)
        Vector3 headLocalPos = rigRoot.InverseTransformPoint(head.position);
        Quaternion headLocalRot = Quaternion.Inverse(rigRoot.rotation) * head.rotation;

        Quaternion newRigRot = yawOnly
            ? Quaternion.Euler(0f, Yaw(targetRot) - Yaw(headLocalRot), 0f)
            : targetRot * Quaternion.Inverse(headLocalRot);

        // Solve rigRoot so that rigRoot * headLocal == target
        Vector3 newRigPos = targetPos - newRigRot * headLocalPos;

        float t = 1f - smoothing;
        rigRoot.SetPositionAndRotation(
            Vector3.Lerp(rigRoot.position, newRigPos, t),
            Quaternion.Slerp(rigRoot.rotation, newRigRot, t));
    }

    static float Yaw(Quaternion q)
    {
        Vector3 f = q * Vector3.forward;
        return Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
    }

    static Quaternion Normalize(Quaternion q)
    {
        float m = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return m > 1e-6f ? new Quaternion(q.x / m, q.y / m, q.z / m, q.w / m) : Quaternion.identity;
    }
}