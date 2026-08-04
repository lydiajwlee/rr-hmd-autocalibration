// Reality Roost Unity starter-kit placeholder
//
// Replace this entire file with the working OSCPoseReceiver.cs from the
// Unity development computer. This comment-only file is valid C# and will not
// create a Unity component until the real script is pasted here.
//
// Before copying the real receiver back into this repository, verify that it:
//
// 1. Listens on UDP port 9000.
// 2. Handles the OSC address "/markers".
// 3. Reads the payload in this order:
//      int markerId
//      float positionX, positionY, positionZ
//      float quaternionX, quaternionY, quaternionZ, quaternionW
//      float/double timestamp
// 4. Accepts HMD marker ID 0.
// 5. Applies the incoming pose to the intended XR Rig transform.
// 6. Logs the received pose during initial integration testing.
//
// The current Python pipeline sends only the first valid pose. It requires
// HMD marker 0 and wall anchors 100 and 101 to be visible in the same frame.
// Restart Python between one-shot transmission tests.
//
// Optional future production behavior (not required for the transport test):
// - Ignore subsequent poses after calibration.
// - Remove lerp/slerp from one-shot placement.
// - Apply a measured marker-to-headset transform.
// - Notify Unity that calibration completed.

