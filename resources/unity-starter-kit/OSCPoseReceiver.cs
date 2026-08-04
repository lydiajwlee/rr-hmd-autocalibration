using UnityEngine;
using extOSC;


public class OSCPoseReceiver : MonoBehaviour
{
    [Header("OSC Settings")]
    public int port = 9000;
    public string address = "/markers";
    public int expectedMarkerId = 0;

    [Header("References")]
    public Transform calibrationOffset;
    public Transform xrCamera; // Assign XR Origin/Camera Offset/Main Camera

    [Header("Marker-to-Head Alignment")]
    [Tooltip("Fixed Euler rotation from marker 0 to the headset tracking frame.")]
    public Vector3 markerToHeadEulerOffset = Vector3.zero;

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
        if (calibrationOffset == null || xrCamera == null)
        {
            Debug.LogError(
                "[OSCPoseReceiver] Calibration Offset and XR Camera must be assigned."
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

        Quaternion markerToHeadRotation =
            Quaternion.Euler(markerToHeadEulerOffset);

        Quaternion detectedHeadRotation =
            detectedHmdRotation * markerToHeadRotation;

        /*
         * Find the camera's tracking pose relative to CalibrationOffset.
         *
         * cameraWorld = calibrationOffset * cameraLocal
         *
         * Therefore:
         * calibrationOffset = detectedHmdWorld * inverse(cameraLocal)
         */
        Vector3 cameraLocalPosition =
            calibrationOffset.InverseTransformPoint(xrCamera.position);

        Quaternion cameraLocalRotation =
            Quaternion.Inverse(calibrationOffset.rotation) *
            xrCamera.rotation;

        targetRotation =
            detectedHeadRotation *
            Quaternion.Inverse(cameraLocalRotation);

        targetPosition =
            detectedHmdPosition -
            targetRotation * cameraLocalPosition;

        hasTarget = true;
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
