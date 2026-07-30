using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// Applies the first calibration pose received from OSC and ignores every
/// subsequent message. Bind ReceivePose to the /markers handler in the
/// project's OSC package.
/// </summary>
public sealed class OneShotCalibrationReceiver : MonoBehaviour
{
    [SerializeField] private Transform xrRig;
    [SerializeField] private Transform trackedHead;
    [SerializeField] private int hmdMarkerId = 0;
    [SerializeField] private UnityEvent onCalibrationComplete;

    public bool IsCalibrated { get; private set; }

    public void ReceivePose(
        int markerId,
        float positionX,
        float positionY,
        float positionZ,
        float rotationX,
        float rotationY,
        float rotationZ,
        float rotationW,
        double timestamp)
    {
        if (IsCalibrated || markerId != hmdMarkerId)
            return;

        ApplyHeadPose(
            new Vector3(positionX, positionY, positionZ),
            new Quaternion(rotationX, rotationY, rotationZ, rotationW));
    }

    private void ApplyHeadPose(Vector3 worldPosition, Quaternion worldRotation)
    {
        if (xrRig == null || trackedHead == null)
        {
            Debug.LogError(
                "OneShotCalibrationReceiver requires both XR Rig and Tracked Head.",
                this);
            return;
        }

        // Compute the head offset relative to the rig (the camera can be nested
        // under an XR Origin camera offset), then remove that offset so the
        // tracked head lands on the measured pose.
        Vector3 headPositionInRigSpace =
            xrRig.InverseTransformPoint(trackedHead.position);
        Quaternion headRotationInRigSpace =
            Quaternion.Inverse(xrRig.rotation) * trackedHead.rotation;
        Quaternion rigRotation =
            worldRotation * Quaternion.Inverse(headRotationInRigSpace);
        Vector3 rigPosition =
            worldPosition - rigRotation * headPositionInRigSpace;

        xrRig.SetPositionAndRotation(rigPosition, rigRotation);
        IsCalibrated = true;

        onCalibrationComplete?.Invoke();
        Debug.Log("One-time HMD calibration complete.", this);
    }
}
