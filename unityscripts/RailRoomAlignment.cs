using System.Collections.Generic;
using UnityEngine;

// Fit worldMarker = worldEye + eyeRotation * mountOffset
//                 = floorOrigin + horizontalTranslation + yaw * roomMarker.
// The XR floor supplies the vertical origin. No marker rotation is required.
public sealed class RailRoomAlignment
{
    struct Sample
    {
        public Vector3 room, eye;
        public Quaternion rotation;
    }

    readonly List<Sample> samples = new List<Sample>();
    public string Status { get; private set; } = "Waiting for headset movement";
    public int SampleCount => samples.Count;

    public bool TryAdd(Vector3 room, Vector3 eye, Quaternion rotation, float floorY,
                       out Pose rails, out float rms)
    {
        rails = Pose.identity;
        rms = float.PositiveInfinity;
        if (samples.Count > 0)
        {
            Sample last = samples[samples.Count - 1];
            if (Vector3.Distance(last.eye, eye) < 0.05f &&
                Quaternion.Angle(last.rotation, rotation) < 6f)
                return false;
        }
        if (samples.Count == 48)
            samples.RemoveAt(0);
        samples.Add(new Sample { room = room, eye = eye, rotation = rotation });
        Status = "Collecting distinct headset poses";
        if (samples.Count < 8)
            return false;

        // Eliminate translation X/Z and mount X/Y/Z first. The final two
        // columns then describe the residual cost for cos(yaw) and sin(yaw).
        // Subtract a nearby origin before fitting for numerical stability.
        Vector3 roomBase = samples[0].room;
        roomBase.y = 0f;
        Vector3 worldBase = samples[0].eye;
        worldBase.y = floorY;
        var normal = new double[7, 8];
        foreach (Sample sample in samples)
        {
            Vector3 p = sample.room - roomBase;
            Vector3 e = sample.eye - worldBase;
            Matrix4x4 r = Matrix4x4.Rotate(sample.rotation);
            AddEquation(normal, new double[] { 1, 0, -r.m00, -r.m01, -r.m02, p.x, p.z }, e.x);
            AddEquation(normal, new double[] { 0, 1, -r.m20, -r.m21, -r.m22, p.z, -p.x }, e.z);
            AddEquation(normal, new double[] { 0, 0, r.m10, r.m11, r.m12, 0, 0 }, p.y - e.y);
        }
        if (!EliminateOffsets(normal, samples.Count))
        {
            Status = "Need a change in both head position and heading";
            return false;
        }
        double a = normal[5, 5], b = normal[5, 6], d = normal[6, 6];
        double smallestEigenvalue = (a + d - System.Math.Sqrt((a-d)*(a-d) + 4*b*b)) / 2;
        if (smallestEigenvalue < samples.Count * 0.005)
        {
            Status = "Need more independent movement to determine room heading";
            return false;
        }
        // Evaluate only rigid rotations. Offsets have already been minimized
        // for each heading, so this scan is independent of sample count.
        double bestCost = double.PositiveInfinity, c = 1, s = 0;
        for (int step = 0; step < 720; step++)
        {
            double angle = step * System.Math.PI / 360;
            double x = System.Math.Cos(angle), z = System.Math.Sin(angle);
            double cost = a*x*x + 2*b*x*z + d*z*z - 2*(normal[5,7]*x + normal[6,7]*z);
            if (cost >= bestCost) continue;
            bestCost = cost;
            c = x;
            s = z;
        }
        float halfYaw = (float)(System.Math.Atan2(s, c) * 0.5);
        Quaternion yaw = new Quaternion(0, Mathf.Sin(halfYaw), 0, Mathf.Cos(halfYaw));
        Vector3 mount = Vector3.zero;
        for (int j = 0; j < 3; j++)
            mount[j] = (float)(normal[j+2,7] - normal[j+2,5]*c - normal[j+2,6]*s);
        if (mount.magnitude > 0.4f)
        {
            Status = $"Estimated sticker offset {mount.magnitude:F2} m exceeds 0.40 m";
            return false;
        }

        Vector3 origin = Vector3.zero;
        foreach (Sample sample in samples)
            origin += sample.eye + sample.rotation * mount - yaw * sample.room;
        origin /= samples.Count;
        origin.y = floorY;
        float squaredError = 0f;
        foreach (Sample sample in samples)
        {
            Vector3 error = origin + yaw * sample.room - (sample.eye + sample.rotation * mount);
            squaredError += error.sqrMagnitude;
        }

        rms = Mathf.Sqrt(squaredError / samples.Count);
        rails = new Pose(origin, yaw);
        Status = $"Pose fit RMS {rms:F3} m (limit 0.025 m)";
        return rms <= 0.025f;
    }

    static void AddEquation(double[,] normal, double[] row, double value)
    {
        for (int i = 0; i < 7; i++)
        {
            for (int j = 0; j < 7; j++)
                normal[i, j] += row[i] * row[j];
            normal[i, 7] += row[i] * value;
        }
    }

    static bool EliminateOffsets(double[,] matrix, int count)
    {
        for (int i = 0; i < 5; i++)
        {
            int pivot = i;
            for (int j = i + 1; j < 5; j++)
                if (System.Math.Abs(matrix[j, i]) > System.Math.Abs(matrix[pivot, i]))
                    pivot = j;
            // Little translation or head rotation leaves some parameters ambiguous.
            if (System.Math.Abs(matrix[pivot, i]) < count * 0.005)
                return false;
            for (int k = i; k <= 7; k++)
            {
                double temp = matrix[i, k];
                matrix[i, k] = matrix[pivot, k];
                matrix[pivot, k] = temp;
            }
            double divisor = matrix[i, i];
            for (int k = i; k <= 7; k++)
                matrix[i, k] /= divisor;
            for (int j = 0; j < 7; j++)
            {
                if (j == i) continue;
                double factor = matrix[j, i];
                for (int k = i; k <= 7; k++)
                    matrix[j, k] -= factor * matrix[i, k];
            }
        }
        return true;
    }
}
