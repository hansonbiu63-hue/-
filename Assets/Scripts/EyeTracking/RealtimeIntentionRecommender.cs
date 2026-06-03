using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class RealtimeIntentionRecommender : MonoBehaviour
{
    [Serializable]
    public class WindowDecision
    {
        public long windowStartTsMs;
        public long windowEndTsMs;
        public int windowTotalFrames;
        public int windowValidFrames;
        public int windowValidAoiFrames;
        public bool decisionReady;
        public string skipReason;

        public int ruleTop1AoiId = InvalidAoiConst;
        public int ruleTop2AoiId = InvalidAoiConst;
        public int ruleTop3AoiId = InvalidAoiConst;
        public int ruleTop1Hits;
        public int ruleTop2Hits;
        public int ruleTop3Hits;

        public int bayesTop1AoiId = InvalidAoiConst;
        public int bayesTop2AoiId = InvalidAoiConst;
        public int bayesTop3AoiId = InvalidAoiConst;
        public float bayesTop1Prob;
        public float bayesTop2Prob;
        public float bayesTop3Prob;
        public float bayesConfidence;
        public float bayesThreshold;
        public bool bayesAccepted;

        private const int InvalidAoiConst = -1;
    }

    public event Action<WindowDecision> OnWindowFinalized;
    public WindowDecision LastDecision { get; private set; }

    [Header("Dependencies")]
    public GazeDataLogger dataLogger;
    public EyeTrackingManager eyeTrackingManager;
    public Text ruleResultText;
    public Text bayesResultText;

    [Header("Debug")]
    public Text debugText;
    public bool verboseDebug = false;

    [Header("Window")]
    public float windowSeconds = 3f;
    [Tooltip("Window too sparse => waiting message.")]
    public int minValidFramesForDecision = 5;

    [Header("Rule Method")]
    public float ruleMinContinuousDwellSeconds = 0.8f;
    public float ruleMinTotalDwellSeconds = 1.0f;
    public float ruleMinHitRatio = 0.2f;
    public float ruleShortGapToleranceSeconds = 0.1f;
    public bool ruleEnablePupilGate = false;
    public float ruleMinMeanPupilZ = 0.2f;
    public float ruleMaxPupilDeltaForValid = 0.1f;

    [Header("Bayes Method")]
    [Tooltip("Hit indicator likelihood weight.")]
    public float wHit = 1.20f;
    [Tooltip("Distance likelihood weight.")]
    public float wDist = 1.00f;
    [Tooltip("Continuous dwell likelihood weight.")]
    public float wDwell = 0.80f;
    [Tooltip("Eye speed likelihood weight.")]
    public float wSpeed = 0.60f;
    [Tooltip("Pupil likelihood weight.")]
    public float wPupil = 0.35f;
    [Tooltip("Local stability likelihood weight.")]
    public float wStability = 0.55f;

    public float bayesSeenPenalty = 1.1f;
    public float bayesUnseenPenalty = 0.1f;
    public float bayesBaseThreshold = 0.8f;
    public float bayesConfidenceAlpha = 0.5f;

    [Header("Bayes Feature Params")]
    public float bayesDistanceSigma = 0.18f;
    public float bayesSpeedScale = 1.2f;
    public float bayesProximitySigma = 0.22f;
    public float bayesStabilityDispersionScale = 0.05f;
    public float bayesHitLikelihoodIfHit = 0.92f;
    public float bayesHitLikelihoodIfMiss = 0.18f;
    public int stabilityHistorySize = 12;

    private const int InvalidAoi = -1;
    private const float Eps = 1e-6f;
    private const string RecommendationPrefix = "\u63a8\u8350\u6ce8\u89c6\uff1a";
    private const string ControlSuffix = "\u63a7\u4ef6";
    private const string ChineseComma = "\uff0c";
    private const string WaitingText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";
    private const string NoTargetText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";

    private readonly Dictionary<int, int> aoiIndexById = new Dictionary<int, int>(16);
    private int aoiCount;
    private int[] aoiIds;
    private string[] aoiLabels;
    private Vector2[] aoiCenters01;
    private bool[] aoiCenterValid;

    private float[] ruleTotalDwell;
    private float[] ruleMaxContinuous;
    private float[] ruleCurrentContinuous;
    private float[] rulePendingGap;
    private int[] ruleHitFrames;
    private float[] rulePupilZSum;
    private int[] rulePupilZCount;
    private bool[] seenInWindow;

    private float[] bayesPosterior;
    private float[] bayesPenalty;
    private float[] bayesLogWork;

    private Vector2[] gazeHistory;
    private int gazeHistoryCount;
    private int gazeHistoryWrite;
    private bool hasPrevGaze;
    private Vector2 prevGaze01;
    private long prevGazeTsMs = long.MinValue;

    private float[] pupilValues;
    private int[] pupilAoiIndices;
    private int pupilSampleCount;
    private bool hasAcceptedPupil;
    private float lastAcceptedPupil;
    private int onlinePupilCount;
    private float onlinePupilMean;
    private float onlinePupilM2;

    private long windowStartTsMs = long.MinValue;
    private long lastFrameTsMs = long.MinValue;
    private long fallbackTsMs = 0;
    private int windowTotalFrames;
    private int windowValidFrames;
    private int windowValidAoiFrames;

    private bool initialized;
    private bool subscribed;

    private void Start()
    {
        InitializeIfNeeded();
        SetWaitingTexts();
    }

    private void OnEnable()
    {
        InitializeIfNeeded();
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
    }

    private void OnDestroy()
    {
        Unsubscribe();
    }

    private void InitializeIfNeeded()
    {
        if (initialized)
        {
            return;
        }

        if (dataLogger == null)
        {
            dataLogger = FindObjectOfType<GazeDataLogger>();
        }
        if (eyeTrackingManager == null)
        {
            eyeTrackingManager = FindObjectOfType<EyeTrackingManager>();
        }

        if (dataLogger == null)
        {
            Debug.LogError("[RealtimeIntentionRecommender] dataLogger is null.");
            return;
        }
        if (eyeTrackingManager == null)
        {
            Debug.LogError("[RealtimeIntentionRecommender] eyeTrackingManager is null.");
            return;
        }
        if (eyeTrackingManager.controlItems == null || eyeTrackingManager.controlItems.Count == 0)
        {
            Debug.LogError("[RealtimeIntentionRecommender] eyeTrackingManager.controlItems is empty.");
            return;
        }

        ControlIdUtility.NormalizeLegacyOneBasedIds(eyeTrackingManager.controlItems, "RealtimeIntentionRecommender");
        RebuildAoiCache();
        AllocateRuntimeBuffers();
        ResetWindowStats();

        initialized = true;
    }

    private void Subscribe()
    {
        if (!initialized || subscribed || dataLogger == null)
        {
            return;
        }

        dataLogger.OnFrameRecorded += OnFrameRecorded;
        subscribed = true;
    }

    private void Unsubscribe()
    {
        if (!subscribed || dataLogger == null)
        {
            return;
        }

        dataLogger.OnFrameRecorded -= OnFrameRecorded;
        subscribed = false;
    }

    private void RebuildAoiCache()
    {
        aoiIndexById.Clear();

        List<ControlItem> controls = eyeTrackingManager != null ? eyeTrackingManager.controlItems : null;
        if (controls == null || controls.Count == 0)
        {
            aoiCount = 0;
            return;
        }

        List<int> ids = new List<int>(controls.Count);
        List<ControlItem> uniqueControls = new List<ControlItem>(controls.Count);

        for (int i = 0; i < controls.Count; i++)
        {
            ControlItem item = controls[i];
            if (item == null)
            {
                continue;
            }

            int id = item.controlID;
            if (id < 0)
            {
                Debug.LogWarning($"[RealtimeIntentionRecommender] Skip invalid controlID={id} on {item.name}.");
                continue;
            }
            if (aoiIndexById.ContainsKey(id))
            {
                Debug.LogWarning($"[RealtimeIntentionRecommender] Duplicate controlID={id} on {item.name}, keep first occurrence.");
                continue;
            }

            aoiIndexById[id] = ids.Count;
            ids.Add(id);
            uniqueControls.Add(item);
        }

        aoiCount = ids.Count;
        aoiIds = new int[aoiCount];
        aoiLabels = new string[aoiCount];
        aoiCenters01 = new Vector2[aoiCount];
        aoiCenterValid = new bool[aoiCount];

        for (int i = 0; i < aoiCount; i++)
        {
            ControlItem item = uniqueControls[i];
            aoiIds[i] = ids[i];
            aoiLabels[i] = BuildLabel(item, ids[i]);

            Vector2 center01;
            bool ok = TryComputeAoiCenter01(item, out center01);
            aoiCenters01[i] = center01;
            aoiCenterValid[i] = ok;

            if (!ok)
            {
                Debug.LogWarning($"[RealtimeIntentionRecommender] AOI center unavailable for {aoiLabels[i]} (controlID={aoiIds[i]}).");
            }
        }
    }

    public void RefreshAoiCache()
    {
        if (!initialized || eyeTrackingManager == null || eyeTrackingManager.controlItems == null)
        {
            return;
        }

        RebuildAoiCache();
    }

    private string BuildLabel(ControlItem item, int id)
    {
        if (item != null && !string.IsNullOrEmpty(item.name))
        {
            return item.name;
        }
        return $"Control_{id}";
    }

    private bool TryComputeAoiCenter01(ControlItem item, out Vector2 center01)
    {
        center01 = new Vector2(0.5f, 0.5f);

        if (eyeTrackingManager == null || eyeTrackingManager.uiPlaneRect == null || item == null)
        {
            return false;
        }

        RectTransform targetRect = null;
        if (item.button != null)
        {
            targetRect = item.button.GetComponent<RectTransform>();
        }
        if (targetRect == null)
        {
            targetRect = item.GetComponent<RectTransform>();
        }
        if (targetRect == null)
        {
            return false;
        }

        RectTransform uiPlane = eyeTrackingManager.uiPlaneRect;
        Rect planeRect = uiPlane.rect;
        if (Mathf.Abs(planeRect.width) < Eps || Mathf.Abs(planeRect.height) < Eps)
        {
            return false;
        }

        Vector3 worldCenter = targetRect.TransformPoint(targetRect.rect.center);
        Vector3 localInPlane = uiPlane.InverseTransformPoint(worldCenter);

        float x01 = Mathf.InverseLerp(planeRect.xMin, planeRect.xMax, localInPlane.x);
        float y01 = Mathf.InverseLerp(planeRect.yMin, planeRect.yMax, localInPlane.y);

        if (eyeTrackingManager.rotateScreen180)
        {
            x01 = 1f - x01;
            y01 = 1f - y01;
        }

        if (!IsFinite(x01) || !IsFinite(y01))
        {
            return false;
        }

        if (x01 < 0f || x01 > 1f || y01 < 0f || y01 > 1f)
        {
            Debug.LogWarning($"[RealtimeIntentionRecommender] AOI center outside uiPlaneRect: {item.name} => ({x01:F3},{y01:F3}), clamped.");
        }

        center01 = new Vector2(Mathf.Clamp01(x01), Mathf.Clamp01(y01));
        return true;
    }

    private void AllocateRuntimeBuffers()
    {
        if (aoiCount <= 0)
        {
            return;
        }

        ruleTotalDwell = new float[aoiCount];
        ruleMaxContinuous = new float[aoiCount];
        ruleCurrentContinuous = new float[aoiCount];
        rulePendingGap = new float[aoiCount];
        ruleHitFrames = new int[aoiCount];
        rulePupilZSum = new float[aoiCount];
        rulePupilZCount = new int[aoiCount];
        seenInWindow = new bool[aoiCount];

        bayesPosterior = new float[aoiCount];
        bayesPenalty = new float[aoiCount];
        bayesLogWork = new float[aoiCount];

        float uniform = 1f / aoiCount;
        for (int i = 0; i < aoiCount; i++)
        {
            bayesPosterior[i] = uniform;
            bayesPenalty[i] = 1f;
        }

        int gazeHistoryLen = Mathf.Max(4, stabilityHistorySize);
        gazeHistory = new Vector2[gazeHistoryLen];

        int pupilCap = Mathf.Max(256, Mathf.CeilToInt(Mathf.Max(1f, windowSeconds) * 240f) + 16);
        pupilValues = new float[pupilCap];
        pupilAoiIndices = new int[pupilCap];
    }

    private void ResetWindowStats()
    {
        if (aoiCount <= 0)
        {
            return;
        }

        for (int i = 0; i < aoiCount; i++)
        {
            ruleTotalDwell[i] = 0f;
            ruleMaxContinuous[i] = 0f;
            ruleCurrentContinuous[i] = 0f;
            rulePendingGap[i] = 0f;
            ruleHitFrames[i] = 0;
            rulePupilZSum[i] = 0f;
            rulePupilZCount[i] = 0;
            seenInWindow[i] = false;
        }

        pupilSampleCount = 0;
        hasAcceptedPupil = false;
        lastAcceptedPupil = 0f;
        onlinePupilCount = 0;
        onlinePupilMean = 0f;
        onlinePupilM2 = 0f;

        gazeHistoryCount = 0;
        gazeHistoryWrite = 0;
        hasPrevGaze = false;
        prevGazeTsMs = long.MinValue;
        prevGaze01 = Vector2.zero;

        windowTotalFrames = 0;
        windowValidFrames = 0;
        windowValidAoiFrames = 0;
    }

    private void SetWaitingTexts()
    {
        if (ruleResultText != null)
        {
            ruleResultText.text = WaitingText;
        }
        if (bayesResultText != null)
        {
            bayesResultText.text = WaitingText;
        }
    }

    private void OnFrameRecorded(GazeDataLogger.FrameRecord rec)
    {
        if (!initialized || rec == null || aoiCount <= 0)
        {
            return;
        }

        long tsMs = ResolveTimestampMs(rec.timestampMs);
        if (windowStartTsMs == long.MinValue)
        {
            windowStartTsMs = tsMs;
        }

        long windowMs = Mathf.Max(1, Mathf.RoundToInt(Mathf.Max(0.2f, windowSeconds) * 1000f));
        while (tsMs - windowStartTsMs >= windowMs)
        {
            FinalizeWindow();
            ResetWindowStats();
            windowStartTsMs += windowMs;
        }

        float dt = ComputeDeltaSec(tsMs);
        ProcessFrame(rec, tsMs, dt);
    }

    private long ResolveTimestampMs(long srcTsMs)
    {
        long resolved;
        if (srcTsMs > 0)
        {
            resolved = srcTsMs;
        }
        else
        {
            resolved = (long)(Time.realtimeSinceStartupAsDouble * 1000.0);
            if (fallbackTsMs > 0 && resolved <= fallbackTsMs)
            {
                resolved = fallbackTsMs + 1;
            }
            fallbackTsMs = resolved;
        }

        if (lastFrameTsMs != long.MinValue && resolved <= lastFrameTsMs)
        {
            resolved = lastFrameTsMs + 1;
        }

        return resolved;
    }

    private float ComputeDeltaSec(long tsMs)
    {
        float dt = GetDefaultFrameDt();
        if (lastFrameTsMs != long.MinValue)
        {
            long deltaMs = tsMs - lastFrameTsMs;
            if (deltaMs > 0 && deltaMs < 1000)
            {
                dt = deltaMs / 1000f;
            }
        }
        lastFrameTsMs = tsMs;
        return Mathf.Clamp(dt, 0.001f, 0.2f);
    }

    private float GetDefaultFrameDt()
    {
        if (eyeTrackingManager != null && eyeTrackingManager.samplingInterval > 0.001f && eyeTrackingManager.samplingInterval < 0.2f)
        {
            return eyeTrackingManager.samplingInterval;
        }
        return 0.02f;
    }

    private void ProcessFrame(GazeDataLogger.FrameRecord rec, long tsMs, float dt)
    {
        windowTotalFrames++;

        bool validFrame = rec.trackingState == 1
                          && rec.hitUIPlane == 1
                          && IsNormalized01(rec.screenXNorm)
                          && IsNormalized01(rec.screenYNorm);

        int chosenAoiId = rec.aoiId;
        int chosenIndex = GetAoiIndex(chosenAoiId);

        if (!validFrame)
        {
            UpdateContinuity(InvalidAoi, dt);
            return;
        }

        windowValidFrames++;

        Vector2 gaze01 = new Vector2(rec.screenXNorm, rec.screenYNorm);
        float gazeSpeed = UpdateGazeState(gaze01, tsMs, dt);
        float dispersion = ComputeCurrentDispersion();

        UpdateContinuity(chosenIndex, dt);
        if (chosenIndex >= 0)
        {
            windowValidAoiFrames++;
            ruleHitFrames[chosenIndex]++;
            ruleTotalDwell[chosenIndex] += dt;
            seenInWindow[chosenIndex] = true;
        }

        float pupilZForBayes = 0f;
        TryProcessPupil(rec.pupilDiameter, chosenIndex, out pupilZForBayes);
        UpdateBayesPosterior(chosenIndex, gaze01, gazeSpeed, dispersion, pupilZForBayes);
    }

    private int GetAoiIndex(int aoiId)
    {
        int idx;
        if (aoiId >= 0 && aoiIndexById.TryGetValue(aoiId, out idx))
        {
            return idx;
        }
        return InvalidAoi;
    }

    private float UpdateGazeState(Vector2 gaze01, long tsMs, float dt)
    {
        gazeHistory[gazeHistoryWrite] = gaze01;
        gazeHistoryWrite = (gazeHistoryWrite + 1) % gazeHistory.Length;
        if (gazeHistoryCount < gazeHistory.Length)
        {
            gazeHistoryCount++;
        }

        float speed = 0f;
        if (hasPrevGaze)
        {
            float d = Vector2.Distance(gaze01, prevGaze01);
            float denom = dt;
            if (prevGazeTsMs != long.MinValue)
            {
                long dms = tsMs - prevGazeTsMs;
                if (dms > 0)
                {
                    denom = Mathf.Max(0.001f, dms / 1000f);
                }
            }
            speed = d / Mathf.Max(0.001f, denom);
        }

        prevGaze01 = gaze01;
        prevGazeTsMs = tsMs;
        hasPrevGaze = true;
        return speed;
    }

    private float ComputeCurrentDispersion()
    {
        if (gazeHistoryCount <= 1)
        {
            return 0f;
        }

        Vector2 sum = Vector2.zero;
        for (int i = 0; i < gazeHistoryCount; i++)
        {
            sum += gazeHistory[i];
        }
        Vector2 mean = sum / gazeHistoryCount;

        float distSum = 0f;
        for (int i = 0; i < gazeHistoryCount; i++)
        {
            distSum += Vector2.Distance(gazeHistory[i], mean);
        }
        return distSum / gazeHistoryCount;
    }

    private void UpdateContinuity(int hitAoiIndex, float dt)
    {
        float gapTolerance = Mathf.Max(0f, ruleShortGapToleranceSeconds);
        for (int i = 0; i < aoiCount; i++)
        {
            if (i == hitAoiIndex)
            {
                if (ruleCurrentContinuous[i] > 0f && rulePendingGap[i] > 0f && rulePendingGap[i] <= gapTolerance)
                {
                    ruleCurrentContinuous[i] += rulePendingGap[i] + dt;
                }
                else if (ruleCurrentContinuous[i] > 0f && rulePendingGap[i] <= 0f)
                {
                    ruleCurrentContinuous[i] += dt;
                }
                else
                {
                    ruleCurrentContinuous[i] = dt;
                }
                rulePendingGap[i] = 0f;
            }
            else
            {
                if (ruleCurrentContinuous[i] <= 0f)
                {
                    continue;
                }

                rulePendingGap[i] += dt;
                if (rulePendingGap[i] > gapTolerance)
                {
                    if (ruleCurrentContinuous[i] > ruleMaxContinuous[i])
                    {
                        ruleMaxContinuous[i] = ruleCurrentContinuous[i];
                    }
                    ruleCurrentContinuous[i] = 0f;
                    rulePendingGap[i] = 0f;
                }
            }
        }
    }

    private bool TryProcessPupil(float pupilDiameter, int chosenAoiIndex, out float pupilZForBayes)
    {
        pupilZForBayes = 0f;
        if (!IsFinite(pupilDiameter) || pupilDiameter <= 0f)
        {
            return false;
        }

        if (hasAcceptedPupil && Mathf.Abs(pupilDiameter - lastAcceptedPupil) > Mathf.Max(0f, ruleMaxPupilDeltaForValid))
        {
            return false;
        }

        if (onlinePupilCount >= 2)
        {
            float std = Mathf.Sqrt(onlinePupilM2 / Mathf.Max(1, onlinePupilCount - 1));
            if (std > 1e-4f)
            {
                pupilZForBayes = (pupilDiameter - onlinePupilMean) / std;
                if (!IsFinite(pupilZForBayes))
                {
                    pupilZForBayes = 0f;
                }
            }
        }

        onlinePupilCount++;
        float delta = pupilDiameter - onlinePupilMean;
        onlinePupilMean += delta / onlinePupilCount;
        float delta2 = pupilDiameter - onlinePupilMean;
        onlinePupilM2 += delta * delta2;

        if (chosenAoiIndex >= 0)
        {
            EnsurePupilCapacity(pupilSampleCount + 1);
            pupilValues[pupilSampleCount] = pupilDiameter;
            pupilAoiIndices[pupilSampleCount] = chosenAoiIndex;
            pupilSampleCount++;
        }

        hasAcceptedPupil = true;
        lastAcceptedPupil = pupilDiameter;
        return true;
    }

    private void EnsurePupilCapacity(int desired)
    {
        if (pupilValues != null && desired <= pupilValues.Length)
        {
            return;
        }

        int oldLen = pupilValues != null ? pupilValues.Length : 0;
        int newLen = Mathf.Max(desired, Mathf.Max(256, oldLen * 2));

        float[] newValues = new float[newLen];
        int[] newAoiIndices = new int[newLen];

        if (oldLen > 0)
        {
            Array.Copy(pupilValues, newValues, pupilSampleCount);
            Array.Copy(pupilAoiIndices, newAoiIndices, pupilSampleCount);
        }

        pupilValues = newValues;
        pupilAoiIndices = newAoiIndices;
    }

    private void UpdateBayesPosterior(int hitAoiIndex, Vector2 gaze01, float gazeSpeed, float dispersion, float pupilZ)
    {
        float sigmaDist = Mathf.Max(0.01f, bayesDistanceSigma);
        float sigmaProx = Mathf.Max(0.01f, bayesProximitySigma);
        float speedScale = Mathf.Max(0.01f, bayesSpeedScale);
        float stabilityScale = Mathf.Max(0.005f, bayesStabilityDispersionScale);
        float stabilityBase = Mathf.Exp(-dispersion / stabilityScale);

        float maxLog = float.NegativeInfinity;
        for (int i = 0; i < aoiCount; i++)
        {
            float dist = 1.5f;
            if (aoiCenterValid[i])
            {
                dist = Vector2.Distance(gaze01, aoiCenters01[i]);
            }

            float lHit;
            if (hitAoiIndex >= 0)
            {
                lHit = (i == hitAoiIndex) ? bayesHitLikelihoodIfHit : bayesHitLikelihoodIfMiss;
            }
            else
            {
                lHit = 0.5f;
            }
            lHit = Mathf.Clamp(lHit, Eps, 1f);

            float lDist = Mathf.Exp(-(dist * dist) / (2f * sigmaDist * sigmaDist));
            lDist = Mathf.Clamp(lDist, Eps, 1f);

            float dwellNorm = Mathf.Clamp01(ruleCurrentContinuous[i] / Mathf.Max(0.05f, ruleMinContinuousDwellSeconds));
            float lDwell = 0.5f + 0.5f * dwellNorm;
            lDwell = Mathf.Clamp(lDwell, Eps, 1f);

            float proximity = Mathf.Exp(-(dist * dist) / (2f * sigmaProx * sigmaProx));
            float speedFactor = Mathf.Exp(-gazeSpeed / speedScale);
            float lSpeed = Mathf.Lerp(1f, speedFactor, proximity);
            lSpeed = Mathf.Clamp(lSpeed, Eps, 1f);

            float lPupil = 1f;
            if (i == hitAoiIndex)
            {
                lPupil = 1f + 0.15f * Mathf.Clamp(pupilZ, 0f, 3f);
            }
            lPupil = Mathf.Clamp(lPupil, Eps, 2f);

            float lStability = 0.5f + 0.5f * (stabilityBase * proximity);
            lStability = Mathf.Clamp(lStability, Eps, 1f);

            float prior = Mathf.Max(Eps, bayesPosterior[i]);
            float penalty = Mathf.Max(Eps, bayesPenalty[i]);

            float logValue = Mathf.Log(prior)
                             + Mathf.Log(penalty)
                             + wHit * Mathf.Log(lHit)
                             + wDist * Mathf.Log(lDist)
                             + wDwell * Mathf.Log(lDwell)
                             + wSpeed * Mathf.Log(lSpeed)
                             + wPupil * Mathf.Log(lPupil)
                             + wStability * Mathf.Log(lStability);

            bayesLogWork[i] = logValue;
            if (logValue > maxLog)
            {
                maxLog = logValue;
            }
        }

        float sum = 0f;
        for (int i = 0; i < aoiCount; i++)
        {
            float v = Mathf.Exp(bayesLogWork[i] - maxLog);
            bayesPosterior[i] = v;
            sum += v;
        }

        if (sum <= Eps || float.IsNaN(sum) || float.IsInfinity(sum))
        {
            float uniform = 1f / aoiCount;
            for (int i = 0; i < aoiCount; i++)
            {
                bayesPosterior[i] = uniform;
            }
            return;
        }

        float inv = 1f / sum;
        for (int i = 0; i < aoiCount; i++)
        {
            bayesPosterior[i] *= inv;
        }
    }

    private void FinalizeWindow()
    {
        long windowMs = Mathf.Max(1, Mathf.RoundToInt(Mathf.Max(0.2f, windowSeconds) * 1000f));
        WindowDecision decision = new WindowDecision
        {
            windowStartTsMs = windowStartTsMs,
            windowEndTsMs = windowStartTsMs == long.MinValue ? long.MinValue : windowStartTsMs + windowMs,
            windowTotalFrames = windowTotalFrames,
            windowValidFrames = windowValidFrames,
            windowValidAoiFrames = windowValidAoiFrames,
            decisionReady = false,
            skipReason = ""
        };

        for (int i = 0; i < aoiCount; i++)
        {
            if (ruleCurrentContinuous[i] > ruleMaxContinuous[i])
            {
                ruleMaxContinuous[i] = ruleCurrentContinuous[i];
            }
        }

        if (windowValidFrames < Mathf.Max(1, minValidFramesForDecision))
        {
            decision.skipReason = "insufficient_valid_frames";
            if (ruleResultText != null)
            {
                ruleResultText.text = WaitingText;
            }
            if (bayesResultText != null)
            {
                bayesResultText.text = WaitingText;
            }
            if (debugText != null && verboseDebug)
            {
                debugText.text = $"Window valid={windowValidFrames}, total={windowTotalFrames}";
            }
            LastDecision = decision;
            OnWindowFinalized?.Invoke(decision);
            return;
        }

        ComputeRulePupilZStats();

        RuleTopHits ruleTop = GetTop3RuleHits();
        decision.ruleTop1AoiId = MapAoiId(ruleTop.idx0);
        decision.ruleTop2AoiId = MapAoiId(ruleTop.idx1);
        decision.ruleTop3AoiId = MapAoiId(ruleTop.idx2);
        decision.ruleTop1Hits = Mathf.Max(0, ruleTop.hits0);
        decision.ruleTop2Hits = Mathf.Max(0, ruleTop.hits1);
        decision.ruleTop3Hits = Mathf.Max(0, ruleTop.hits2);
        string ruleOutput = BuildRuleOutput(ruleTop);
        if (ruleResultText != null)
        {
            ruleResultText.text = ruleOutput;
        }

        TopK bayesTop = GetTopKPosterior(3);
        float top1 = bayesTop.prob0;
        float top2 = bayesTop.prob1;
        float confidence = Mathf.Clamp01((top1 - top2) + 0.5f * top1);
        float threshold = bayesBaseThreshold * (1f - bayesConfidenceAlpha * (1f - confidence));
        threshold = Mathf.Clamp01(threshold);

        bool bayesAccept = bayesTop.idx0 >= 0 && top1 >= threshold;
        decision.bayesTop1AoiId = MapAoiId(bayesTop.idx0);
        decision.bayesTop2AoiId = MapAoiId(bayesTop.idx1);
        decision.bayesTop3AoiId = MapAoiId(bayesTop.idx2);
        decision.bayesTop1Prob = bayesTop.prob0;
        decision.bayesTop2Prob = bayesTop.prob1;
        decision.bayesTop3Prob = bayesTop.prob2;
        decision.bayesConfidence = confidence;
        decision.bayesThreshold = threshold;
        decision.bayesAccepted = bayesAccept;
        decision.decisionReady = true;

        string bayesOutput = bayesAccept ? BuildBayesOutput(bayesTop, threshold) : NoTargetText;

        if (bayesResultText != null)
        {
            bayesResultText.text = bayesOutput;
        }

        for (int i = 0; i < aoiCount; i++)
        {
            bayesPenalty[i] = seenInWindow[i] ? bayesSeenPenalty : bayesUnseenPenalty;
        }

        if (debugText != null && verboseDebug)
        {
            debugText.text = $"RuleTop={GetAoiDebugLabel(ruleTop.idx0)}\nBayesTop={GetAoiDebugLabel(bayesTop.idx0)} P={top1:F3}\nvalid={windowValidFrames}, total={windowTotalFrames}";
        }

        LastDecision = decision;
        OnWindowFinalized?.Invoke(decision);
    }

    private void ComputeRulePupilZStats()
    {
        for (int i = 0; i < aoiCount; i++)
        {
            rulePupilZSum[i] = 0f;
            rulePupilZCount[i] = 0;
        }

        if (pupilSampleCount < 2)
        {
            return;
        }

        float sum = 0f;
        for (int i = 0; i < pupilSampleCount; i++)
        {
            sum += pupilValues[i];
        }
        float mean = sum / pupilSampleCount;

        float sq = 0f;
        for (int i = 0; i < pupilSampleCount; i++)
        {
            float d = pupilValues[i] - mean;
            sq += d * d;
        }
        float variance = sq / Mathf.Max(1, pupilSampleCount - 1);
        float std = Mathf.Sqrt(Mathf.Max(0f, variance));
        bool validStd = std > 1e-4f;

        for (int i = 0; i < pupilSampleCount; i++)
        {
            int aoiIdx = pupilAoiIndices[i];
            if (aoiIdx < 0 || aoiIdx >= aoiCount)
            {
                continue;
            }

            float z = validStd ? (pupilValues[i] - mean) / std : 0f;
            if (!IsFinite(z))
            {
                z = 0f;
            }
            rulePupilZSum[aoiIdx] += z;
            rulePupilZCount[aoiIdx]++;
        }
    }

    private RuleTopHits GetTop3RuleHits()
    {
        RuleTopHits top = new RuleTopHits
        {
            idx0 = InvalidAoi,
            idx1 = InvalidAoi,
            idx2 = InvalidAoi,
            hits0 = -1,
            hits1 = -1,
            hits2 = -1
        };

        for (int i = 0; i < aoiCount; i++)
        {
            int hits = ruleHitFrames[i];
            if (hits > top.hits0)
            {
                top.hits2 = top.hits1;
                top.idx2 = top.idx1;
                top.hits1 = top.hits0;
                top.idx1 = top.idx0;
                top.hits0 = hits;
                top.idx0 = i;
            }
            else if (hits > top.hits1)
            {
                top.hits2 = top.hits1;
                top.idx2 = top.idx1;
                top.hits1 = hits;
                top.idx1 = i;
            }
            else if (hits > top.hits2)
            {
                top.hits2 = hits;
                top.idx2 = i;
            }
        }

        return top;
    }

    private string BuildRuleOutput(RuleTopHits top)
    {
        if (windowValidFrames <= 0 || windowValidAoiFrames <= 0 || top.idx0 < 0 || top.hits0 <= 0)
        {
            return NoTargetText;
        }

        List<int> recommendedIds = new List<int>(3);
        AddRuleRecommendation(recommendedIds, top.idx0, top.hits0);
        AddRuleRecommendation(recommendedIds, top.idx1, top.hits1);
        AddRuleRecommendation(recommendedIds, top.idx2, top.hits2);
        return BuildRecommendationText(recommendedIds);
    }

    private string BuildBayesOutput(TopK top, float threshold)
    {
        List<int> recommendedIds = new List<int>(3);
        AddBayesRecommendation(recommendedIds, top.idx0, top.prob0, threshold);
        AddBayesRecommendation(recommendedIds, top.idx1, top.prob1, threshold);
        AddBayesRecommendation(recommendedIds, top.idx2, top.prob2, threshold);
        return BuildRecommendationText(recommendedIds);
    }

    private void AddRuleRecommendation(List<int> recommendedIds, int idx, int hits)
    {
        if (recommendedIds == null || idx < 0 || idx >= aoiCount || hits <= 0)
        {
            return;
        }

        recommendedIds.Add(aoiIds[idx]);
    }

    private void AddBayesRecommendation(List<int> recommendedIds, int idx, float probability, float threshold)
    {
        if (recommendedIds == null || idx < 0 || idx >= aoiCount || probability < threshold)
        {
            return;
        }

        recommendedIds.Add(aoiIds[idx]);
    }

    private string BuildRecommendationText(List<int> recommendedIds)
    {
        if (recommendedIds == null || recommendedIds.Count == 0)
        {
            return NoTargetText;
        }

        string[] labels = new string[recommendedIds.Count];
        for (int i = 0; i < recommendedIds.Count; i++)
        {
            labels[i] = recommendedIds[i].ToString();
        }

        return RecommendationPrefix + "[" + string.Join(",", labels) + "] " + ControlSuffix;
    }

    private string BuildTop3Debug(TopK topk)
    {
        string s0 = topk.idx0 >= 0 ? $"{aoiIds[topk.idx0]}({topk.prob0:F2})" : "-";
        string s1 = topk.idx1 >= 0 ? $"{aoiIds[topk.idx1]}({topk.prob1:F2})" : "-";
        string s2 = topk.idx2 >= 0 ? $"{aoiIds[topk.idx2]}({topk.prob2:F2})" : "-";
        return $"{s0}, {s1}, {s2}";
    }

    private string GetAoiDebugLabel(int idx)
    {
        if (idx < 0 || idx >= aoiCount)
        {
            return "NONE";
        }
        return $"{aoiLabels[idx]}({aoiIds[idx]})";
    }

    private int MapAoiId(int idx)
    {
        if (idx < 0 || idx >= aoiCount)
        {
            return InvalidAoi;
        }

        return aoiIds[idx];
    }

    private struct TopK
    {
        public int idx0;
        public int idx1;
        public int idx2;
        public float prob0;
        public float prob1;
        public float prob2;
    }

    private struct RuleTopHits
    {
        public int idx0;
        public int idx1;
        public int idx2;
        public int hits0;
        public int hits1;
        public int hits2;
    }

    private TopK GetTopKPosterior(int k)
    {
        TopK top = new TopK
        {
            idx0 = InvalidAoi,
            idx1 = InvalidAoi,
            idx2 = InvalidAoi,
            prob0 = 0f,
            prob1 = 0f,
            prob2 = 0f
        };

        for (int i = 0; i < aoiCount; i++)
        {
            float p = bayesPosterior[i];
            if (p > top.prob0)
            {
                top.prob2 = top.prob1;
                top.idx2 = top.idx1;
                top.prob1 = top.prob0;
                top.idx1 = top.idx0;
                top.prob0 = p;
                top.idx0 = i;
            }
            else if (p > top.prob1)
            {
                top.prob2 = top.prob1;
                top.idx2 = top.idx1;
                top.prob1 = p;
                top.idx1 = i;
            }
            else if (p > top.prob2)
            {
                top.prob2 = p;
                top.idx2 = i;
            }
        }

        if (k <= 1)
        {
            top.idx1 = InvalidAoi;
            top.idx2 = InvalidAoi;
            top.prob1 = 0f;
            top.prob2 = 0f;
        }
        else if (k == 2)
        {
            top.idx2 = InvalidAoi;
            top.prob2 = 0f;
        }

        return top;
    }

    private bool IsNormalized01(float v)
    {
        return IsFinite(v) && v >= 0f && v <= 1f;
    }

    private bool IsFinite(float v)
    {
        return !float.IsNaN(v) && !float.IsInfinity(v);
    }
}
