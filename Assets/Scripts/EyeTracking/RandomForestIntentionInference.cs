using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class RandomForestIntentionInference : MonoBehaviour
{
    [Serializable]
    public class WindowDecision
    {
        public long windowStartTsMs;
        public long windowEndTsMs;
        public int windowTotalFrames;
        public int windowValidFrames;
        public bool decisionReady;
        public string skipReason;
        public int top1AoiId = -1;
        public int top2AoiId = -1;
        public int top3AoiId = -1;
        public float top1Prob;
        public float top2Prob;
        public float top3Prob;
        public float threshold;
        public List<int> predictedAoiIds = new List<int>();
    }

    [Serializable]
    private class RandomForestModelJson
    {
        public string model_type;
        public string[] feature_names;
        public int window_frames = 150;
        public int step_frames = 15;
        public int min_valid_frames = 20;
        public int early_min_valid_frames = 40;
        public int aoi_count = 10;
        public float threshold = 0.3f;
        public ForestJson[] forests;
    }

    [Serializable]
    private class ForestJson
    {
        public TreeJson[] trees;
    }

    [Serializable]
    private class TreeJson
    {
        public int[] feature;
        public float[] threshold;
        public int[] left;
        public int[] right;
        public float[] value;
    }

    private class RuntimeFrameSample
    {
        public long timestampMs;
        public int trackingState;
        public int hitUiPlane;
        public int aoiId;
        public float screenXNorm;
        public float screenYNorm;
        public float leftOpenness;
        public float rightOpenness;
    }

    public event Action<WindowDecision> OnWindowFinalized;
    public WindowDecision LastDecision { get; private set; }

    [Header("Dependencies")]
    public GazeDataLogger dataLogger;
    public Text resultText;

    [Header("Model")]
    public TextAsset randomForestAsset;
    public string randomForestResourcePath = "eye_random_forest";

    [Header("Runtime")]
    public int fallbackWindowFrames = 150;
    public int fallbackStepFrames = 15;
    public int fallbackMinValidFrames = 20;
    public int fallbackEarlyMinValidFrames = 40;
    public int topK = 3;

    private const int AoiCount = 10;
    private const string RecommendationPrefix = "\u63a8\u8350\u6ce8\u89c6\uff1a";
    private const string ControlSuffix = "\u63a7\u4ef6";
    private const string ChineseComma = "\uff0c";
    private const string WaitingText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";
    private const string NoTargetText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";

    private readonly Queue<RuntimeFrameSample> sampleQueue = new Queue<RuntimeFrameSample>();
    private RandomForestModelJson model;
    private bool initialized;
    private bool subscribed;
    private int framesSinceInference;

    private void Start()
    {
        InitializeIfNeeded();
        Subscribe();
        RefreshWaitingText();
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

        LoadModel();
        initialized = dataLogger != null && model != null && model.forests != null && model.forests.Length > 0;
        if (!initialized)
        {
            SetStatus("随机森林配置错误\n缺少数据记录器或模型");
        }
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

    private void LoadModel()
    {
        TextAsset asset = randomForestAsset;
        if (asset == null && !string.IsNullOrEmpty(randomForestResourcePath))
        {
            asset = Resources.Load<TextAsset>(randomForestResourcePath);
        }

        if (asset == null || string.IsNullOrEmpty(asset.text))
        {
            Debug.LogWarning("[RandomForestIntentionInference] eye_random_forest.json not found.");
            return;
        }

        try
        {
            model = JsonUtility.FromJson<RandomForestModelJson>(asset.text);
        }
        catch (Exception e)
        {
            Debug.LogWarning("[RandomForestIntentionInference] Failed to parse random forest model. " + e.Message);
            model = null;
        }
    }

    private void OnFrameRecorded(GazeDataLogger.FrameRecord rec)
    {
        if (!initialized || rec == null)
        {
            return;
        }

        sampleQueue.Enqueue(new RuntimeFrameSample
        {
            timestampMs = rec.timestampMs,
            trackingState = rec.trackingState,
            hitUiPlane = rec.hitUIPlane,
            aoiId = rec.aoiId,
            screenXNorm = rec.screenXNorm,
            screenYNorm = rec.screenYNorm,
            leftOpenness = rec.leftOpenness,
            rightOpenness = rec.rightOpenness
        });

        int windowFrames = Mathf.Max(1, model != null ? model.window_frames : fallbackWindowFrames);
        while (sampleQueue.Count > windowFrames)
        {
            sampleQueue.Dequeue();
        }

        framesSinceInference++;
        int stepFrames = Mathf.Max(1, model != null ? model.step_frames : fallbackStepFrames);
        if (framesSinceInference < stepFrames)
        {
            return;
        }

        RuntimeFrameSample[] window = sampleQueue.ToArray();
        int validFrames = CountValidFrames(window);
        int minValid = Mathf.Max(1, model != null ? model.min_valid_frames : fallbackMinValidFrames);
        int earlyMinValid = Mathf.Max(minValid, model != null ? model.early_min_valid_frames : fallbackEarlyMinValidFrames);
        if (window.Length < windowFrames && validFrames < earlyMinValid)
        {
            RefreshWaitingText();
            return;
        }

        framesSinceInference = 0;
        RunInference(window, validFrames);
    }

    private void RunInference(RuntimeFrameSample[] window, int validFrames)
    {
        WindowDecision decision = new WindowDecision
        {
            windowStartTsMs = window != null && window.Length > 0 ? window[0].timestampMs : 0,
            windowEndTsMs = window != null && window.Length > 0 ? window[window.Length - 1].timestampMs : 0,
            windowTotalFrames = window != null ? window.Length : 0,
            windowValidFrames = validFrames,
            threshold = Mathf.Clamp01(model != null ? model.threshold : 0.3f)
        };

        int minValid = Mathf.Max(1, model != null ? model.min_valid_frames : fallbackMinValidFrames);
        if (validFrames < minValid)
        {
            decision.decisionReady = false;
            decision.skipReason = "valid_frames_below_min";
            FinalizeDecision(decision);
            return;
        }

        try
        {
            float[] features = BuildWindowFeatures(window);
            float[] probs = Predict(features);
            FillDecision(decision, probs);
        }
        catch (Exception e)
        {
            decision.decisionReady = false;
            decision.skipReason = "inference_exception";
            Debug.LogWarning("[RandomForestIntentionInference] Inference failed. " + e.Message);
        }

        FinalizeDecision(decision);
    }

    private float[] Predict(float[] features)
    {
        int classCount = Mathf.Max(1, model.aoi_count);
        float[] probs = new float[classCount];

        for (int c = 0; c < classCount; c++)
        {
            ForestJson forest = model.forests != null && c < model.forests.Length ? model.forests[c] : null;
            if (forest == null || forest.trees == null || forest.trees.Length == 0)
            {
                continue;
            }

            float sum = 0f;
            int count = 0;
            for (int t = 0; t < forest.trees.Length; t++)
            {
                TreeJson tree = forest.trees[t];
                if (tree == null)
                {
                    continue;
                }
                sum += EvaluateTree(tree, features);
                count++;
            }
            probs[c] = count > 0 ? Mathf.Clamp01(sum / count) : 0f;
        }

        return probs;
    }

    private float EvaluateTree(TreeJson tree, float[] features)
    {
        int node = 0;
        int guard = 0;
        while (tree.left != null && tree.right != null && node >= 0 && node < tree.left.Length && guard++ < 2048)
        {
            int left = tree.left[node];
            int right = tree.right[node];
            if (left < 0 && right < 0)
            {
                break;
            }

            int featureIndex = tree.feature != null && node < tree.feature.Length ? tree.feature[node] : -1;
            float threshold = tree.threshold != null && node < tree.threshold.Length ? tree.threshold[node] : 0f;
            float value = featureIndex >= 0 && features != null && featureIndex < features.Length ? features[featureIndex] : 0f;
            node = value <= threshold ? left : right;
            if (node < 0)
            {
                break;
            }
        }

        if (tree.value == null || node < 0 || node >= tree.value.Length)
        {
            return 0f;
        }
        return Mathf.Clamp01(tree.value[node]);
    }

    private float[] BuildWindowFeatures(RuntimeFrameSample[] window)
    {
        List<float> xs = new List<float>();
        List<float> ys = new List<float>();
        List<float> speed = new List<float>();
        List<float> openness = new List<float>();
        List<int> validAoi = new List<int>();
        float validCount = 0f;
        float[] dwell = new float[AoiCount];
        int[] firstOrder = new int[AoiCount];
        for (int i = 0; i < firstOrder.Length; i++) firstOrder[i] = -1;

        bool hasPrev = false;
        float prevX = 0f;
        float prevY = 0f;
        long prevTs = 0L;

        for (int i = 0; i < window.Length; i++)
        {
            RuntimeFrameSample sample = window[i];
            bool valid = sample.trackingState == 1 && sample.hitUiPlane == 1 && IsNormalized01(sample.screenXNorm) && IsNormalized01(sample.screenYNorm);
            if (!valid)
            {
                hasPrev = false;
                continue;
            }

            validCount++;
            xs.Add(sample.screenXNorm);
            ys.Add(sample.screenYNorm);
            if (sample.aoiId >= 0 && sample.aoiId < AoiCount)
            {
                dwell[sample.aoiId] += 1f;
                validAoi.Add(sample.aoiId);
                if (firstOrder[sample.aoiId] < 0)
                {
                    firstOrder[sample.aoiId] = validAoi.Count;
                }
            }

            float open = 0.5f * (SanitizeFloat(sample.leftOpenness, 0f) + SanitizeFloat(sample.rightOpenness, 0f));
            if (IsFinite(open) && open > 0f)
            {
                openness.Add(open);
            }

            if (hasPrev && sample.timestampMs > prevTs)
            {
                float dt = Mathf.Max(0.001f, (sample.timestampMs - prevTs) / 1000f);
                float dx = sample.screenXNorm - prevX;
                float dy = sample.screenYNorm - prevY;
                speed.Add(Mathf.Sqrt(dx * dx + dy * dy) / dt);
            }

            hasPrev = true;
            prevX = sample.screenXNorm;
            prevY = sample.screenYNorm;
            prevTs = sample.timestampMs;
        }

        if (validCount > 0f)
        {
            for (int i = 0; i < dwell.Length; i++)
            {
                dwell[i] /= validCount;
            }
        }

        int transitions = 0;
        for (int i = 1; i < validAoi.Count; i++)
        {
            if (validAoi[i] != validAoi[i - 1])
            {
                transitions++;
            }
        }

        float entropy = 0f;
        for (int i = 0; i < dwell.Length; i++)
        {
            if (dwell[i] > 1e-8f)
            {
                entropy += -dwell[i] * Mathf.Log(dwell[i]);
            }
        }

        float[] f = new float[32];
        f[0] = window.Length > 0 ? validCount / window.Length : 0f;
        f[1] = validAoi.Count > 1 ? transitions / Mathf.Max(1f, validAoi.Count - 1f) : 0f;
        f[2] = entropy;
        f[3] = Percentile(speed, 50f);
        f[4] = Percentile(speed, 90f);
        f[5] = Mean(xs);
        f[6] = Mean(ys);
        f[7] = Std(xs, f[5]);
        f[8] = Std(ys, f[6]);
        f[9] = Mean(openness);
        f[10] = Std(openness, f[9]);
        f[11] = BlinkRatio(openness);

        for (int i = 0; i < AoiCount; i++)
        {
            f[12 + i] = dwell[i];
            f[22 + i] = firstOrder[i] < 0 ? 0f : firstOrder[i] / Mathf.Max(1f, validAoi.Count);
        }

        return f;
    }

    private int CountValidFrames(RuntimeFrameSample[] window)
    {
        int count = 0;
        if (window == null)
        {
            return count;
        }
        for (int i = 0; i < window.Length; i++)
        {
            if (window[i].trackingState == 1)
            {
                count++;
            }
        }
        return count;
    }

    private void FillDecision(WindowDecision decision, float[] probs)
    {
        if (decision == null || probs == null || probs.Length == 0)
        {
            return;
        }

        int[] topIndices = GetTopIndices(probs, Mathf.Min(Mathf.Max(1, topK), probs.Length));
        if (topIndices.Length > 0)
        {
            decision.top1AoiId = topIndices[0];
            decision.top1Prob = probs[topIndices[0]];
        }
        if (topIndices.Length > 1)
        {
            decision.top2AoiId = topIndices[1];
            decision.top2Prob = probs[topIndices[1]];
        }
        if (topIndices.Length > 2)
        {
            decision.top3AoiId = topIndices[2];
            decision.top3Prob = probs[topIndices[2]];
        }

        decision.predictedAoiIds.Clear();
        for (int i = 0; i < topIndices.Length; i++)
        {
            int id = topIndices[i];
            if (id >= 0 && id < probs.Length && probs[id] >= decision.threshold)
            {
                decision.predictedAoiIds.Add(id);
            }
        }

        decision.decisionReady = true;
        decision.skipReason = "";
    }

    private void FinalizeDecision(WindowDecision decision)
    {
        LastDecision = decision;
        UpdateResultText(decision);
        OnWindowFinalized?.Invoke(decision);
    }

    private void UpdateResultText(WindowDecision decision)
    {
        if (resultText == null)
        {
            return;
        }

        if (decision == null || !decision.decisionReady)
        {
            RefreshWaitingText();
            return;
        }

        resultText.text = BuildRecommendationText(decision.predictedAoiIds);
    }

    private void RefreshWaitingText()
    {
        if (resultText != null)
        {
            resultText.text = WaitingText;
        }
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

    private void SetStatus(string message)
    {
        if (resultText != null)
        {
            resultText.text = message;
        }
    }

    private static int[] GetTopIndices(float[] values, int k)
    {
        int count = Mathf.Min(k, values != null ? values.Length : 0);
        int[] indices = new int[count];
        float[] scores = new float[count];
        for (int i = 0; i < count; i++)
        {
            indices[i] = -1;
            scores[i] = float.NegativeInfinity;
        }

        for (int i = 0; i < values.Length; i++)
        {
            for (int slot = 0; slot < count; slot++)
            {
                if (values[i] > scores[slot])
                {
                    for (int move = count - 1; move > slot; move--)
                    {
                        scores[move] = scores[move - 1];
                        indices[move] = indices[move - 1];
                    }
                    scores[slot] = values[i];
                    indices[slot] = i;
                    break;
                }
            }
        }

        return indices;
    }

    private static float Mean(List<float> values)
    {
        if (values == null || values.Count == 0) return 0f;
        float sum = 0f;
        for (int i = 0; i < values.Count; i++) sum += values[i];
        return sum / values.Count;
    }

    private static float Std(List<float> values, float mean)
    {
        if (values == null || values.Count == 0) return 0f;
        float sum = 0f;
        for (int i = 0; i < values.Count; i++)
        {
            float d = values[i] - mean;
            sum += d * d;
        }
        return Mathf.Sqrt(sum / values.Count);
    }

    private static float Percentile(List<float> values, float percentile)
    {
        if (values == null || values.Count == 0) return 0f;
        List<float> sorted = new List<float>(values);
        sorted.Sort();
        if (sorted.Count == 1) return sorted[0];
        float pos = Mathf.Clamp(percentile, 0f, 100f) / 100f * (sorted.Count - 1);
        int lo = Mathf.FloorToInt(pos);
        int hi = Mathf.CeilToInt(pos);
        return lo == hi ? sorted[lo] : Mathf.Lerp(sorted[lo], sorted[hi], pos - lo);
    }

    private static float BlinkRatio(List<float> openness)
    {
        if (openness == null || openness.Count == 0) return 0f;
        int count = 0;
        for (int i = 0; i < openness.Count; i++)
        {
            if (openness[i] < 0.2f) count++;
        }
        return count / (float)openness.Count;
    }

    private static float SanitizeFloat(float value, float fallback)
    {
        return float.IsNaN(value) || float.IsInfinity(value) ? fallback : value;
    }

    private static bool IsNormalized01(float value)
    {
        return IsFinite(value) && value >= 0f && value <= 1f;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
