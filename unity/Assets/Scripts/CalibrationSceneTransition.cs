using UnityEngine;

/// <summary>
/// Switches from entrance passthrough to the VR scene. Connect TransitionToVr
/// to OneShotCalibrationReceiver.onCalibrationComplete in the Inspector.
/// </summary>
public sealed class CalibrationSceneTransition : MonoBehaviour
{
    [Tooltip("The Meta passthrough layer, or another component controlling passthrough.")]
    [SerializeField] private Behaviour passthrough;
    [SerializeField] private GameObject vrSceneRoot;

    private void Awake()
    {
        if (passthrough != null)
            passthrough.enabled = true;

        if (vrSceneRoot != null)
            vrSceneRoot.SetActive(false);
    }

    public void TransitionToVr()
    {
        if (vrSceneRoot != null)
            vrSceneRoot.SetActive(true);

        if (passthrough != null)
            passthrough.enabled = false;
    }
}
