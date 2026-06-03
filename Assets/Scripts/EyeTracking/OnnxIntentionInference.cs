using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class OnnxIntentionInference : MonoBehaviour
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

        public int top1AoiId = InvalidAoiConst;
        public int top2AoiId = InvalidAoiConst;
        public int top3AoiId = InvalidAoiConst;
        public float top1Prob;
        public float top2Prob;
        public float top3Prob;
        public float threshold;
        public List<int> predictedAoiIds = new List<int>();
        public bool partialWindow;

        private const int InvalidAoiConst = -1;
    }

    [Serializable]
    public enum OutputInterpretation
    {
        Auto = 0,
        MultiLabelSigmoid = 1,
        MultiClassSoftmax = 2,
        RawLogits = 3
    }

    [Serializable]
    private class TrainingStatsJson
    {
        public string stage;
        public string dataset;
        public string[] feature_names;
        public string[] active_feature_names;
        public string[] dropped_feature_names;
        public int[] continuous_feature_idxs;
        public int[] binary_feature_idxs;
        public int[] onehot_feature_idxs;
        public float[] mean;
        public float[] std;
        public float threshold = 0.5f;
        public int window_frames = 150;
        public int step_frames = 15;
        public int min_valid_frames = 20;
        public int feature_dim = 47;
        public int aoi_count = 10;
        public int aoi_label_base = 0;
    }

    private class RuntimeFrameSample
    {
        public long timestampMs;
        public int trackingState;
        public int hitUiPlane;
        public int aoiId;
        public float screenXNorm;
        public float screenYNorm;
        public float gazeDirX;
        public float gazeDirY;
        public float gazeDirZ;
        public float pupilDiameter;
        public int pupilValid;
        public float leftOpenness;
        public float rightOpenness;
    }

    public event Action<WindowDecision> OnWindowFinalized;

    public WindowDecision LastDecision { get; private set; }

    [Header("Dependencies")]
    public ModelAsset modelAsset;
    public GazeDataLogger dataLogger;
    public Text resultText;

    [Header("Sentis")]
    public BackendType backendType = BackendType.CPU;
    public OutputInterpretation outputInterpretation = OutputInterpretation.Auto;
    public bool logModelSignature = true;
    public bool applySoftmax = true;

    [Header("Runtime")]
    public int fallbackSequenceLength = 150;
    public int fallbackFeatureDim = 11;
    [Tooltip("Run ONNX inference once per interval using the latest 150-frame window.")]
    public float inferenceIntervalSeconds = 3f;
    public int runEveryNFrames = 5;
    public int topK = 3;
    public float invalidFillValue = 0f;
    public bool allowPartialWindowInference = true;
    public int earlyMinValidFramesForInference = 40;
    public bool normalizeAoiIdToUnit = true;
    public string[] featureOrder = new string[0];
    public string[] classLabels = new string[0];

    [Header("Training Alignment")]
    public TextAsset trainingStatsAsset;
    public bool useTrainingStats = true;
    public string trainingStatsResourcePath = "eye_transformer_stats";
    public bool useTrainingWindowConfig = true;
    public bool useSigmoidOutput = true;
    public int minValidFramesForInference = 20;
    [Tooltip("Require the offline finetune stats JSON produced by Tools/train_eye_transformer_modified.py.")]
    public bool requireFinetunedStats = true;

    private const int InvalidAoi = -1;
    private const int AoiCountDefault = 10;
    private const int FeatureDimDefault = 47;
    private const string RecommendationPrefix = "\u63a8\u8350\u6ce8\u89c6\uff1a";
    private const string ControlSuffix = "\u63a7\u4ef6";
    private const string ChineseComma = "\uff0c";
    private const string WaitingText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";
    private const string NoTargetText = "\u63a8\u8350\u6ce8\u89c6\uff1a[] \u63a7\u4ef6";

    private const int IDX_SCREEN_X = 0;
    private const int IDX_SCREEN_Y = 1;
    private const int IDX_DELTA_X = 2;
    private const int IDX_DELTA_Y = 3;
    private const int IDX_GAZE_SPEED = 4;
    private const int IDX_GAZE_DIR_X = 5;
    private const int IDX_GAZE_DIR_Y = 6;
    private const int IDX_GAZE_DIR_Z = 7;
    private const int IDX_GAZE_DIR_SPEED = 8;
    private const int IDX_LEFT_OPEN = 9;
    private const int IDX_RIGHT_OPEN = 10;
    private const int IDX_OPEN_MEAN = 11;
    private const int IDX_OPEN_DELTA = 12;
    private const int IDX_PUPIL_DIAMETER = 13;
    private const int IDX_PUPIL_VALID = 14;
    private const int IDX_TRACKING_VALID = 15;
    private const int IDX_HIT_UI_PLANE = 16;
    private const int IDX_AOI_ONEHOT_START = 17;
    private const int IDX_PUPIL_MEAN = 27;
    private const int IDX_PUPIL_STD = 28;
    private const int IDX_SPEED_P50 = 29;
    private const int IDX_SPEED_P90 = 30;
    private const int IDX_VALID_RATIO = 31;
    private const int IDX_AOI_TRANSITION_RATE = 32;
    private const int IDX_AOI_ENTROPY = 33;
    private const int IDX_WINDOW_OPENNESS_MEAN = 34;
    private const int IDX_WINDOW_OPENNESS_STD = 35;
    private const int IDX_BLINK_RATIO = 36;
    private const int IDX_AOI_DWELL_START = 37;

    private static readonly string[] FeatureNames = new[]
    {
        "screen_x_norm", "screen_y_norm", "delta_x", "delta_y", "gaze_speed",
        "gaze_dir_x", "gaze_dir_y", "gaze_dir_z", "gaze_dir_speed", "left_openness",
        "right_openness", "openness_mean", "openness_delta", "pupil_diameter", "pupil_valid",
        "tracking_valid", "hit_ui_plane", "aoi_onehot_0", "aoi_onehot_1", "aoi_onehot_2",
        "aoi_onehot_3", "aoi_onehot_4", "aoi_onehot_5", "aoi_onehot_6", "aoi_onehot_7",
        "aoi_onehot_8", "aoi_onehot_9", "pupil_mean", "pupil_std", "speed_p50",
        "speed_p90", "valid_ratio", "aoi_transition_rate", "aoi_entropy", "window_openness_mean",
        "window_openness_std", "blink_ratio", "aoi_dwell_0", "aoi_dwell_1", "aoi_dwell_2",
        "aoi_dwell_3", "aoi_dwell_4", "aoi_dwell_5", "aoi_dwell_6", "aoi_dwell_7",
        "aoi_dwell_8", "aoi_dwell_9"
    };

    private readonly Queue<RuntimeFrameSample> sampleQueue = new Queue<RuntimeFrameSample>();
    private Worker worker;
    private Model runtimeModel;
    private TrainingStatsJson trainingStats;
    private bool[] activeFeatureMask;
    private int[] continuousFeatureIndices;
    private float[] featureMean;
    private float[] featureStd;
    private bool initialized;
    private bool subscribed;
    private bool loggedFeatureOrderWarning;
    private int sequenceLength;
    private int featureDim;
    private int classCount;
    private double lastInferenceRealtime = double.NegativeInfinity;
    private int inferenceCount;
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
        DisposeWorker();
    }

    private void OnDestroy()
    {
        Unsubscribe();
        DisposeWorker();
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

        if (modelAsset == null)
        {
            Debug.LogError("[OnnxIntentionInference] modelAsset is null.");
            return;
        }
        if (dataLogger == null)
        {
            Debug.LogError("[OnnxIntentionInference] dataLogger is null.");
            return;
        }

        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, backendType);

        sequenceLength = Mathf.Max(1, fallbackSequenceLength);
        featureDim = Mathf.Max(1, fallbackFeatureDim);
        classCount = Mathf.Max(1, AoiCountDefault);

        TryResolveModelSignature();
        LoadTrainingStatsIfNeeded();
        ApplyTrainingDefaultsIfNeeded();
        if (!ValidateDeploymentConfig())
        {
            DisposeWorker();
            return;
        }
        ResolveClassLabels();

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

    private void DisposeWorker()
    {
        if (worker != null)
        {
            worker.Dispose();
            worker = null;
        }

        runtimeModel = null;
    }

    private void TryResolveModelSignature()
    {
        if (runtimeModel == null || runtimeModel.inputs == null || runtimeModel.inputs.Count == 0)
        {
            return;
        }

        try
        {
            Model.Input input = runtimeModel.inputs[0];
            if (input.shape.IsStatic())
            {
                TensorShape shape = input.shape.ToTensorShape();
                if (shape.rank >= 2)
                {
                    sequenceLength = Mathf.Max(1, shape[shape.rank - 2]);
                    featureDim = Mathf.Max(1, shape[shape.rank - 1]);
                }
                else if (shape.rank == 1)
                {
                    featureDim = Mathf.Max(1, shape[0]);
                }
            }

            if (logModelSignature)
            {
                Debug.Log($"[OnnxIntentionInference] Model input shape resolved to sequence={sequenceLength}, featureDim={featureDim}");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[OnnxIntentionInference] Failed to inspect model signature, fallback to inspector values. {e.Message}");
        }
    }

    private void LoadTrainingStatsIfNeeded()
    {
        if (!useTrainingStats)
        {
            SetupFallbackFeatureProcessing();
            return;
        }

        TextAsset statsAsset = trainingStatsAsset;
        if (statsAsset == null && !string.IsNullOrEmpty(trainingStatsResourcePath))
        {
            statsAsset = Resources.Load<TextAsset>(trainingStatsResourcePath);
        }

        if (statsAsset == null || string.IsNullOrEmpty(statsAsset.text))
        {
            Debug.LogWarning("[OnnxIntentionInference] Training stats JSON not found. Runtime will use fallback feature processing.");
            SetupFallbackFeatureProcessing();
            return;
        }

        try
        {
            trainingStats = JsonUtility.FromJson<TrainingStatsJson>(statsAsset.text);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[OnnxIntentionInference] Failed to parse training stats JSON. {e.Message}");
            trainingStats = null;
        }

        if (trainingStats == null || trainingStats.feature_dim <= 0)
        {
            SetupFallbackFeatureProcessing();
            return;
        }

        featureDim = trainingStats.feature_dim;
        classCount = trainingStats.aoi_count > 0 ? trainingStats.aoi_count : classCount;

        activeFeatureMask = new bool[featureDim];
        for (int i = 0; i < activeFeatureMask.Length; i++)
        {
            activeFeatureMask[i] = true;
        }

        if (trainingStats.dropped_feature_names != null)
        {
            for (int i = 0; i < trainingStats.dropped_feature_names.Length; i++)
            {
                int featureIndex = FeatureNameToIndex(trainingStats.dropped_feature_names[i]);
                if (featureIndex >= 0 && featureIndex < activeFeatureMask.Length)
                {
                    activeFeatureMask[featureIndex] = false;
                }
            }
        }

        continuousFeatureIndices = trainingStats.continuous_feature_idxs != null ? trainingStats.continuous_feature_idxs : new int[0];
        featureMean = NormalizeStatsArray(trainingStats.mean, featureDim, 0f);
        featureStd = NormalizeStatsArray(trainingStats.std, featureDim, 1f);
    }

    private void ApplyTrainingDefaultsIfNeeded()
    {
        if (trainingStats == null)
        {
            return;
        }

        if (useTrainingWindowConfig)
        {
            sequenceLength = Mathf.Max(1, trainingStats.window_frames);
            runEveryNFrames = Mathf.Max(1, trainingStats.step_frames);
            minValidFramesForInference = Mathf.Max(1, trainingStats.min_valid_frames);
            earlyMinValidFramesForInference = Mathf.Max(minValidFramesForInference, 40);
        }

        classCount = trainingStats.aoi_count > 0 ? trainingStats.aoi_count : classCount;
    }

    private bool ValidateDeploymentConfig()
    {
        if (!requireFinetunedStats)
        {
            return true;
        }

        if (!useTrainingStats || trainingStats == null)
        {
            SetFatalStatus("Transformer stats missing. Export Assets/Resources/eye_transformer_stats.json from the offline finetune script.");
            return false;
        }

        if (!string.Equals(trainingStats.stage, "finetune", StringComparison.OrdinalIgnoreCase))
        {
            SetFatalStatus($"Transformer stats stage must be 'finetune', got '{trainingStats.stage}'.");
            return false;
        }

        if (trainingStats.feature_dim != FeatureDimDefault)
        {
            SetFatalStatus($"Transformer feature_dim mismatch: expected {FeatureDimDefault}, got {trainingStats.feature_dim}.");
            return false;
        }

        if (trainingStats.window_frames != 150)
        {
            SetFatalStatus($"Transformer window_frames mismatch: expected 150, got {trainingStats.window_frames}.");
            return false;
        }

        if (trainingStats.aoi_count != AoiCountDefault || trainingStats.aoi_label_base != 0)
        {
            SetFatalStatus($"Transformer AOI config mismatch: expected 10 classes with label base 0, got count={trainingStats.aoi_count}, base={trainingStats.aoi_label_base}.");
            return false;
        }

        if (trainingStats.feature_names == null || trainingStats.feature_names.Length != FeatureDimDefault)
        {
            SetFatalStatus("Transformer feature_names must contain the 47-feature Unity runtime order.");
            return false;
        }

        for (int i = 0; i < FeatureNames.Length; i++)
        {
            if (!string.Equals(trainingStats.feature_names[i], FeatureNames[i], StringComparison.Ordinal))
            {
                SetFatalStatus($"Transformer feature order mismatch at {i}: expected {FeatureNames[i]}, got {trainingStats.feature_names[i]}.");
                return false;
            }
        }

        Debug.Log("[OnnxIntentionInference] Offline finetune stats validated: 150x47, AOI 0-9, sigmoid multi-label.");
        return true;
    }

    private void SetFatalStatus(string message)
    {
        Debug.LogError("[OnnxIntentionInference] " + message);
        if (resultText != null)
        {
            resultText.text = "Transformer 配置错误\n" + message;
        }
    }

    private void SetupFallbackFeatureProcessing()
    {
        trainingStats = null;
        featureDim = Mathf.Max(featureDim, FeatureDimDefault);
        activeFeatureMask = new bool[featureDim];
        for (int i = 0; i < activeFeatureMask.Length; i++)
        {
            activeFeatureMask[i] = true;
        }

        continuousFeatureIndices = new[]
        {
            IDX_SCREEN_X, IDX_SCREEN_Y, IDX_DELTA_X, IDX_DELTA_Y, IDX_GAZE_SPEED,
            IDX_GAZE_DIR_X, IDX_GAZE_DIR_Y, IDX_GAZE_DIR_Z, IDX_GAZE_DIR_SPEED,
            IDX_LEFT_OPEN, IDX_RIGHT_OPEN, IDX_OPEN_MEAN, IDX_OPEN_DELTA, IDX_PUPIL_DIAMETER,
            IDX_PUPIL_MEAN, IDX_PUPIL_STD, IDX_SPEED_P50, IDX_SPEED_P90, IDX_VALID_RATIO,
            IDX_AOI_TRANSITION_RATE, IDX_AOI_ENTROPY, IDX_WINDOW_OPENNESS_MEAN, IDX_WINDOW_OPENNESS_STD,
            IDX_BLINK_RATIO, IDX_AOI_DWELL_START + 0, IDX_AOI_DWELL_START + 1, IDX_AOI_DWELL_START + 2,
            IDX_AOI_DWELL_START + 3, IDX_AOI_DWELL_START + 4, IDX_AOI_DWELL_START + 5, IDX_AOI_DWELL_START + 6,
            IDX_AOI_DWELL_START + 7, IDX_AOI_DWELL_START + 8, IDX_AOI_DWELL_START + 9
        };
        featureMean = NormalizeStatsArray(null, featureDim, 0f);
        featureStd = NormalizeStatsArray(null, featureDim, 1f);
    }

    private void ResolveClassLabels()
    {
        if (classLabels != null && classLabels.Length == classCount)
        {
            return;
        }

        string[] resolved = new string[classCount];
        for (int i = 0; i < classCount; i++)
        {
            resolved[i] = $"Control_{i}";
        }
        classLabels = resolved;
    }

    private void OnFrameRecorded(GazeDataLogger.FrameRecord rec)
    {
        if (!initialized || rec == null)
        {
            return;
        }

        if (featureOrder != null && featureOrder.Length > 0 && !loggedFeatureOrderWarning)
        {
            loggedFeatureOrderWarning = true;
            Debug.Log("[OnnxIntentionInference] Inspector featureOrder is ignored. Runtime now follows training stats feature order for ONNX alignment.");
        }

        RuntimeFrameSample sample = new RuntimeFrameSample
        {
            timestampMs = rec.timestampMs,
            trackingState = rec.trackingState,
            hitUiPlane = rec.hitUIPlane,
            aoiId = rec.aoiId,
            screenXNorm = rec.screenXNorm,
            screenYNorm = rec.screenYNorm,
            gazeDirX = rec.gazeDirX,
            gazeDirY = rec.gazeDirY,
            gazeDirZ = rec.gazeDirZ,
            pupilDiameter = rec.pupilDiameter,
            pupilValid = rec.pupilValid,
            leftOpenness = rec.leftOpenness,
            rightOpenness = rec.rightOpenness
        };

        sampleQueue.Enqueue(sample);
        while (sampleQueue.Count > sequenceLength)
        {
            sampleQueue.Dequeue();
        }

        framesSinceInference++;
        RuntimeFrameSample[] queuedWindow = sampleQueue.ToArray();
        int queuedValidFrames = CountTrackingValidFrames(queuedWindow);
        bool partialWindow = sampleQueue.Count < sequenceLength;

        if (partialWindow && (!allowPartialWindowInference || queuedValidFrames < Mathf.Max(minValidFramesForInference, earlyMinValidFramesForInference)))
        {
            RefreshWaitingText();
            return;
        }

        if (framesSinceInference < Mathf.Max(1, runEveryNFrames))
        {
            return;
        }

        framesSinceInference = 0;
        lastInferenceRealtime = Time.realtimeSinceStartupAsDouble;
        RunInferenceForCurrentWindow(partialWindow);
    }

    private void RunInferenceForCurrentWindow(bool partialWindow)
    {
        RuntimeFrameSample[] window = BuildInferenceWindow(sampleQueue.ToArray());
        if (window == null || window.Length == 0)
        {
            return;
        }

        int validFrames;
        float[] inputData = BuildWindowInput(window, out validFrames);
        WindowDecision decision = new WindowDecision
        {
            windowStartTsMs = window[0].timestampMs,
            windowEndTsMs = window[window.Length - 1].timestampMs,
            windowTotalFrames = window.Length,
            windowValidFrames = validFrames,
            threshold = ResolveDecisionThreshold(),
            partialWindow = partialWindow
        };

        if (validFrames < Mathf.Max(1, minValidFramesForInference))
        {
            decision.decisionReady = false;
            decision.skipReason = partialWindow ? "partial_window_valid_frames_below_min" : "valid_frames_below_min";
            LastDecision = decision;
            UpdateResultText(decision);
            OnWindowFinalized?.Invoke(decision);
            return;
        }

        Tensor<float> inputTensor = null;
        Tensor<float> outputTensor = null;
        Tensor<float> cpuTensor = null;

        try
        {
            inputTensor = new Tensor<float>(new TensorShape(1, sequenceLength, featureDim), inputData);
            worker.Schedule(inputTensor);
            outputTensor = worker.PeekOutput() as Tensor<float>;
            if (outputTensor == null)
            {
                decision.decisionReady = false;
                decision.skipReason = "null_output";
                LastDecision = decision;
                UpdateResultText(decision);
                OnWindowFinalized?.Invoke(decision);
                return;
            }

            cpuTensor = outputTensor.ReadbackAndClone();
            float[] rawValues = cpuTensor.DownloadToArray();
            float[] logits = ExtractClassVector(rawValues, classCount);
            if (logits == null || logits.Length == 0)
            {
                decision.decisionReady = false;
                decision.skipReason = "empty_logits";
                LastDecision = decision;
                UpdateResultText(decision);
                OnWindowFinalized?.Invoke(decision);
                return;
            }

            float[] probs = ConvertLogitsToScores(logits);
            FillDecision(decision, probs);
        }
        catch (Exception e)
        {
            decision.decisionReady = false;
            decision.skipReason = "inference_exception";
            Debug.LogWarning($"[OnnxIntentionInference] Inference failed. {e.Message}");
        }
        finally
        {
            if (cpuTensor != null)
            {
                cpuTensor.Dispose();
            }
            if (inputTensor != null)
            {
                inputTensor.Dispose();
            }
        }

        inferenceCount++;
        LastDecision = decision;
        UpdateResultText(decision);
        OnWindowFinalized?.Invoke(decision);
    }

    private RuntimeFrameSample[] BuildInferenceWindow(RuntimeFrameSample[] source)
    {
        if (source == null)
        {
            return new RuntimeFrameSample[0];
        }

        if (source.Length >= sequenceLength)
        {
            return source;
        }

        RuntimeFrameSample[] padded = new RuntimeFrameSample[sequenceLength];
        int missing = sequenceLength - source.Length;
        long firstTs = source.Length > 0 ? source[0].timestampMs : DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        for (int i = 0; i < missing; i++)
        {
            padded[i] = new RuntimeFrameSample
            {
                timestampMs = firstTs - (missing - i) * 20L,
                trackingState = 0,
                hitUiPlane = 0,
                aoiId = InvalidAoi,
                screenXNorm = invalidFillValue,
                screenYNorm = invalidFillValue,
                gazeDirX = 0f,
                gazeDirY = 0f,
                gazeDirZ = 0f,
                pupilDiameter = 0f,
                pupilValid = 0,
                leftOpenness = 0f,
                rightOpenness = 0f
            };
        }

        Array.Copy(source, 0, padded, missing, source.Length);
        return padded;
    }

    private int CountTrackingValidFrames(RuntimeFrameSample[] window)
    {
        int count = 0;
        if (window == null)
        {
            return count;
        }
        for (int i = 0; i < window.Length; i++)
        {
            if (window[i] != null && window[i].trackingState == 1)
            {
                count++;
            }
        }
        return count;
    }

    private float[] BuildWindowInput(RuntimeFrameSample[] window, out int validFrames)
    {
        int n = window.Length;
        int safeFeatureDim = Mathf.Max(featureDim, FeatureDimDefault);
        float[] x = new float[n * safeFeatureDim];

        float[] sxRaw = new float[n];
        float[] syRaw = new float[n];
        float[] tracking = new float[n];
        float[] hit = new float[n];
        float[] pupilRaw = new float[n];
        float[] leftOpen = new float[n];
        float[] rightOpen = new float[n];
        int[] aoiIds = new int[n];
        bool[] coordValid = new bool[n];

        validFrames = 0;

        for (int i = 0; i < n; i++)
        {
            RuntimeFrameSample sample = window[i];
            sxRaw[i] = SanitizeFloat(sample.screenXNorm, float.NaN);
            syRaw[i] = SanitizeFloat(sample.screenYNorm, float.NaN);
            tracking[i] = sample.trackingState == 1 ? 1f : 0f;
            hit[i] = sample.hitUiPlane == 1 ? 1f : 0f;
            pupilRaw[i] = sample.pupilValid > 0 ? SanitizeFloat(sample.pupilDiameter, 0f) : 0f;
            leftOpen[i] = SanitizeFloat(sample.leftOpenness, 0f);
            rightOpen[i] = SanitizeFloat(sample.rightOpenness, 0f);
            aoiIds[i] = sample.aoiId;

            coordValid[i] = tracking[i] == 1f
                            && hit[i] == 1f
                            && IsFinite(sxRaw[i])
                            && IsFinite(syRaw[i]);
            if (tracking[i] > 0.5f)
            {
                validFrames++;
            }
        }

        float[] sx = NormalizeScreenAxis(sxRaw, coordValid);
        float[] sy = NormalizeScreenAxis(syRaw, coordValid);
        float[] dx = new float[n];
        float[] dy = new float[n];
        float[] speed = new float[n];
        float[] dirSpeed = new float[n];
        float[] dirX = new float[n];
        float[] dirY = new float[n];
        float[] dirZ = new float[n];

        bool hasPrev = false;
        float prevX = 0f;
        float prevY = 0f;
        float prevDirX = 0f;
        float prevDirY = 0f;
        float prevDirZ = 0f;
        long prevTs = 0L;

        for (int i = 0; i < n; i++)
        {
            RuntimeFrameSample sample = window[i];
            if (!coordValid[i])
            {
                hasPrev = false;
                continue;
            }

            dirX[i] = SanitizeFloat(sample.gazeDirX, 0f);
            dirY[i] = SanitizeFloat(sample.gazeDirY, 0f);
            dirZ[i] = SanitizeFloat(sample.gazeDirZ, 0f);

            if (hasPrev && sample.timestampMs > prevTs)
            {
                float dt = Mathf.Max((sample.timestampMs - prevTs) / 1000f, 0.001f);
                dx[i] = sx[i] - prevX;
                dy[i] = sy[i] - prevY;
                speed[i] = Mathf.Sqrt(dx[i] * dx[i] + dy[i] * dy[i]) / dt;

                float ddx = dirX[i] - prevDirX;
                float ddy = dirY[i] - prevDirY;
                float ddz = dirZ[i] - prevDirZ;
                dirSpeed[i] = Mathf.Sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / dt;
            }

            hasPrev = true;
            prevX = sx[i];
            prevY = sy[i];
            prevDirX = dirX[i];
            prevDirY = dirY[i];
            prevDirZ = dirZ[i];
            prevTs = sample.timestampMs;
        }

        float[] pupilValid = new float[n];
        float[] pupil = new float[n];
        float[] openMean = new float[n];
        float[] openDelta = new float[n];

        for (int i = 0; i < n; i++)
        {
            pupilValid[i] = IsFinite(pupilRaw[i]) && pupilRaw[i] > 0f ? 1f : 0f;
            pupil[i] = pupilValid[i] > 0.5f ? pupilRaw[i] : 0f;
            openMean[i] = 0.5f * (leftOpen[i] + rightOpen[i]);
            openDelta[i] = Mathf.Abs(leftOpen[i] - rightOpen[i]);

            SetFeature(x, safeFeatureDim, i, IDX_SCREEN_X, sx[i]);
            SetFeature(x, safeFeatureDim, i, IDX_SCREEN_Y, sy[i]);
            SetFeature(x, safeFeatureDim, i, IDX_DELTA_X, dx[i]);
            SetFeature(x, safeFeatureDim, i, IDX_DELTA_Y, dy[i]);
            SetFeature(x, safeFeatureDim, i, IDX_GAZE_SPEED, speed[i]);
            SetFeature(x, safeFeatureDim, i, IDX_GAZE_DIR_X, dirX[i]);
            SetFeature(x, safeFeatureDim, i, IDX_GAZE_DIR_Y, dirY[i]);
            SetFeature(x, safeFeatureDim, i, IDX_GAZE_DIR_Z, dirZ[i]);
            SetFeature(x, safeFeatureDim, i, IDX_GAZE_DIR_SPEED, dirSpeed[i]);
            SetFeature(x, safeFeatureDim, i, IDX_LEFT_OPEN, leftOpen[i]);
            SetFeature(x, safeFeatureDim, i, IDX_RIGHT_OPEN, rightOpen[i]);
            SetFeature(x, safeFeatureDim, i, IDX_OPEN_MEAN, openMean[i]);
            SetFeature(x, safeFeatureDim, i, IDX_OPEN_DELTA, openDelta[i]);
            SetFeature(x, safeFeatureDim, i, IDX_PUPIL_DIAMETER, pupil[i]);
            SetFeature(x, safeFeatureDim, i, IDX_PUPIL_VALID, pupilValid[i]);
            SetFeature(x, safeFeatureDim, i, IDX_TRACKING_VALID, tracking[i]);
            SetFeature(x, safeFeatureDim, i, IDX_HIT_UI_PLANE, hit[i]);

            int aoiId = aoiIds[i];
            if (aoiId >= 0 && aoiId < AoiCountDefault)
            {
                SetFeature(x, safeFeatureDim, i, IDX_AOI_ONEHOT_START + aoiId, 1f);
            }
        }

        AppendWindowFeatures(x, safeFeatureDim, pupil, pupilValid, tracking, aoiIds, leftOpen, rightOpen, speed);
        ApplyFeatureMaskAndStandardization(x, safeFeatureDim, n);

        if (safeFeatureDim != featureDim)
        {
            float[] resized = new float[n * featureDim];
            int copyCols = Mathf.Min(safeFeatureDim, featureDim);
            for (int row = 0; row < n; row++)
            {
                Array.Copy(x, row * safeFeatureDim, resized, row * featureDim, copyCols);
            }
            return resized;
        }

        return x;
    }

    private void AppendWindowFeatures(float[] x, int stride, float[] pupil, float[] pupilValid, float[] tracking, int[] aoiIds, float[] leftOpen, float[] rightOpen, float[] speed)
    {
        List<float> validPupil = new List<float>();
        List<float> validSpeed = new List<float>();
        List<int> validAoi = new List<int>();
        List<float> validOpen = new List<float>();
        float trackingSum = 0f;

        for (int i = 0; i < tracking.Length; i++)
        {
            if (pupilValid[i] > 0.5f)
            {
                validPupil.Add(pupil[i]);
            }
            if (tracking[i] > 0.5f)
            {
                validSpeed.Add(speed[i]);
                trackingSum += tracking[i];
                if (aoiIds[i] >= 0 && aoiIds[i] < AoiCountDefault)
                {
                    validAoi.Add(aoiIds[i]);
                }
            }

            float mean = 0.5f * (leftOpen[i] + rightOpen[i]);
            if (IsFinite(mean) && mean > 0f)
            {
                validOpen.Add(mean);
            }
        }

        float pupilMean = Mean(validPupil);
        float pupilStd = Std(validPupil, pupilMean);
        float speedP50 = Percentile(validSpeed, 50f);
        float speedP90 = Percentile(validSpeed, 90f);
        float validRatio = tracking.Length > 0 ? trackingSum / tracking.Length : 0f;

        float[] dwell = new float[AoiCountDefault];
        if (validAoi.Count > 0)
        {
            float denom = validAoi.Count;
            for (int i = 0; i < validAoi.Count; i++)
            {
                int aoiId = validAoi[i];
                if (aoiId >= 0 && aoiId < AoiCountDefault)
                {
                    dwell[aoiId] += 1f;
                }
            }

            if (denom > 0f)
            {
                for (int i = 0; i < dwell.Length; i++)
                {
                    dwell[i] /= denom;
                }
            }
        }

        int transitions = 0;
        int prev = InvalidAoi;
        for (int i = 0; i < validAoi.Count; i++)
        {
            int current = validAoi[i];
            if (prev >= 0 && current != prev)
            {
                transitions++;
            }
            prev = current;
        }

        float aoiTransitionRate = validAoi.Count > 1 ? transitions / Mathf.Max(1f, validAoi.Count - 1f) : 0f;
        float aoiEntropy = 0f;
        for (int i = 0; i < dwell.Length; i++)
        {
            if (dwell[i] > 1e-8f)
            {
                aoiEntropy += -dwell[i] * Mathf.Log(dwell[i]);
            }
        }

        float windowOpenMean = Mean(validOpen);
        float windowOpenStd = Std(validOpen, windowOpenMean);
        float blinkRatio = 0f;
        if (validOpen.Count > 0)
        {
            int blinkCount = 0;
            for (int i = 0; i < validOpen.Count; i++)
            {
                if (validOpen[i] < 0.2f)
                {
                    blinkCount++;
                }
            }
            blinkRatio = blinkCount / (float)validOpen.Count;
        }

        for (int row = 0; row < tracking.Length; row++)
        {
            SetFeature(x, stride, row, IDX_PUPIL_MEAN, pupilMean);
            SetFeature(x, stride, row, IDX_PUPIL_STD, pupilStd);
            SetFeature(x, stride, row, IDX_SPEED_P50, speedP50);
            SetFeature(x, stride, row, IDX_SPEED_P90, speedP90);
            SetFeature(x, stride, row, IDX_VALID_RATIO, validRatio);
            SetFeature(x, stride, row, IDX_AOI_TRANSITION_RATE, aoiTransitionRate);
            SetFeature(x, stride, row, IDX_AOI_ENTROPY, aoiEntropy);
            SetFeature(x, stride, row, IDX_WINDOW_OPENNESS_MEAN, windowOpenMean);
            SetFeature(x, stride, row, IDX_WINDOW_OPENNESS_STD, windowOpenStd);
            SetFeature(x, stride, row, IDX_BLINK_RATIO, blinkRatio);

            for (int aoi = 0; aoi < AoiCountDefault; aoi++)
            {
                SetFeature(x, stride, row, IDX_AOI_DWELL_START + aoi, dwell[aoi]);
            }
        }
    }

    private void ApplyFeatureMaskAndStandardization(float[] x, int stride, int rowCount)
    {
        if (activeFeatureMask != null)
        {
            int maxCols = Mathf.Min(stride, activeFeatureMask.Length);
            for (int row = 0; row < rowCount; row++)
            {
                int rowStart = row * stride;
                for (int col = 0; col < maxCols; col++)
                {
                    if (!activeFeatureMask[col])
                    {
                        x[rowStart + col] = 0f;
                    }
                }
            }
        }

        if (continuousFeatureIndices == null || featureMean == null || featureStd == null)
        {
            return;
        }

        for (int row = 0; row < rowCount; row++)
        {
            int rowStart = row * stride;
            for (int i = 0; i < continuousFeatureIndices.Length; i++)
            {
                int col = continuousFeatureIndices[i];
                if (col < 0 || col >= stride || col >= featureMean.Length || col >= featureStd.Length)
                {
                    continue;
                }

                float std = Mathf.Abs(featureStd[col]) < 1e-6f ? 1f : featureStd[col];
                x[rowStart + col] = (x[rowStart + col] - featureMean[col]) / std;
            }
        }
    }

    private float[] ExtractClassVector(float[] rawValues, int expectedClassCount)
    {
        if (rawValues == null || rawValues.Length == 0)
        {
            return null;
        }

        if (expectedClassCount <= 0)
        {
            return rawValues;
        }

        if (rawValues.Length == expectedClassCount)
        {
            return rawValues;
        }

        if (rawValues.Length > expectedClassCount && rawValues.Length % expectedClassCount == 0)
        {
            int blocks = rawValues.Length / expectedClassCount;
            float[] lastBlock = new float[expectedClassCount];
            Array.Copy(rawValues, (blocks - 1) * expectedClassCount, lastBlock, 0, expectedClassCount);
            return lastBlock;
        }

        float[] truncated = new float[Mathf.Min(expectedClassCount, rawValues.Length)];
        Array.Copy(rawValues, 0, truncated, 0, truncated.Length);
        return truncated;
    }

    private float[] ConvertLogitsToScores(float[] logits)
    {
        if (logits == null)
        {
            return new float[0];
        }

        if (ShouldUseSigmoid())
        {
            float[] scores = new float[logits.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                scores[i] = Sigmoid(logits[i]);
            }
            return scores;
        }

        if (outputInterpretation == OutputInterpretation.RawLogits)
        {
            float[] clone = new float[logits.Length];
            Array.Copy(logits, clone, logits.Length);
            return clone;
        }

        if (outputInterpretation == OutputInterpretation.MultiClassSoftmax || applySoftmax)
        {
            return Softmax(logits);
        }

        float[] fallback = new float[logits.Length];
        Array.Copy(logits, fallback, logits.Length);
        return fallback;
    }

    private bool ShouldUseSigmoid()
    {
        if (outputInterpretation == OutputInterpretation.MultiLabelSigmoid)
        {
            return true;
        }
        if (outputInterpretation == OutputInterpretation.MultiClassSoftmax || outputInterpretation == OutputInterpretation.RawLogits)
        {
            return false;
        }
        return useSigmoidOutput;
    }

    private void FillDecision(WindowDecision decision, float[] probs)
    {
        if (decision == null)
        {
            return;
        }

        if (probs == null || probs.Length == 0)
        {
            decision.decisionReady = false;
            decision.skipReason = "empty_scores";
            return;
        }

        decision.predictedAoiIds.Clear();

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

        float threshold = decision.threshold;
        for (int i = 0; i < topIndices.Length; i++)
        {
            int aoiId = topIndices[i];
            if (aoiId >= 0 && aoiId < probs.Length && probs[aoiId] >= threshold)
            {
                decision.predictedAoiIds.Add(aoiId);
            }
        }

        decision.decisionReady = true;
        decision.skipReason = "";
    }

    private void UpdateResultText(WindowDecision decision)
    {
        if (resultText == null)
        {
            return;
        }

        if (decision == null)
        {
            RefreshWaitingText();
            return;
        }

        if (!decision.decisionReady)
        {
            resultText.text = WaitingText;
            return;
        }

        resultText.text = BuildRecommendationText(decision.predictedAoiIds);
    }

    private void RefreshWaitingText()
    {
        if (resultText == null)
        {
            return;
        }

        resultText.text = WaitingText;
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

    private string LabelFor(int aoiId)
    {
        if (aoiId < 0)
        {
            return "NONE";
        }

        if (classLabels != null && aoiId < classLabels.Length && !string.IsNullOrEmpty(classLabels[aoiId]))
        {
            return classLabels[aoiId];
        }

        return $"Control_{aoiId}";
    }

    private float ResolveDecisionThreshold()
    {
        if (trainingStats != null)
        {
            return Mathf.Clamp01(trainingStats.threshold);
        }

        return 0.5f;
    }

    private static float[] NormalizeStatsArray(float[] source, int length, float defaultValue)
    {
        float[] result = new float[length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = defaultValue;
        }

        if (source == null)
        {
            return result;
        }

        int copyLength = Mathf.Min(length, source.Length);
        for (int i = 0; i < copyLength; i++)
        {
            result[i] = source[i];
        }

        return result;
    }

    private static int FeatureNameToIndex(string featureName)
    {
        if (string.IsNullOrEmpty(featureName))
        {
            return -1;
        }

        for (int i = 0; i < FeatureNames.Length; i++)
        {
            if (string.Equals(FeatureNames[i], featureName, StringComparison.Ordinal))
            {
                return i;
            }
        }

        return -1;
    }

    private static void SetFeature(float[] data, int stride, int row, int col, float value)
    {
        if (data == null || row < 0 || col < 0 || col >= stride)
        {
            return;
        }

        int index = row * stride + col;
        if (index < 0 || index >= data.Length)
        {
            return;
        }

        data[index] = IsFinite(value) ? value : 0f;
    }

    private static float SanitizeFloat(float value, float fallback)
    {
        return float.IsNaN(value) || float.IsInfinity(value) ? fallback : value;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static float[] NormalizeScreenAxis(float[] values, bool[] validMask)
    {
        float[] output = new float[values.Length];
        int validCount = 0;
        float minValue = float.PositiveInfinity;
        float maxValue = float.NegativeInfinity;

        for (int i = 0; i < values.Length; i++)
        {
            if (!validMask[i] || !IsFinite(values[i]))
            {
                continue;
            }

            validCount++;
            minValue = Mathf.Min(minValue, values[i]);
            maxValue = Mathf.Max(maxValue, values[i]);
        }

        bool alreadyNormalized = validCount == 0 || (minValue >= 0f && maxValue <= 1f);
        if (alreadyNormalized)
        {
            for (int i = 0; i < values.Length; i++)
            {
                output[i] = validMask[i] && IsFinite(values[i]) ? values[i] : 0f;
            }
            return output;
        }

        List<float> validValues = new List<float>(validCount);
        for (int i = 0; i < values.Length; i++)
        {
            if (validMask[i] && IsFinite(values[i]))
            {
                validValues.Add(values[i]);
            }
        }

        float lo = Percentile(validValues, 1f);
        float hi = Percentile(validValues, 99f);
        if (Mathf.Abs(hi - lo) < 1e-6f)
        {
            lo = validValues.Count > 0 ? Min(validValues) : 0f;
            hi = validValues.Count > 0 ? Max(validValues) : 1f;
        }
        if (Mathf.Abs(hi - lo) < 1e-6f)
        {
            return output;
        }

        for (int i = 0; i < values.Length; i++)
        {
            if (!validMask[i] || !IsFinite(values[i]))
            {
                output[i] = 0f;
                continue;
            }

            float normalized = (values[i] - lo) / (hi - lo);
            output[i] = Mathf.Clamp01(normalized);
        }

        return output;
    }

    private static float Min(List<float> values)
    {
        if (values == null || values.Count == 0)
        {
            return 0f;
        }

        float value = values[0];
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] < value)
            {
                value = values[i];
            }
        }
        return value;
    }

    private static float Max(List<float> values)
    {
        if (values == null || values.Count == 0)
        {
            return 0f;
        }

        float value = values[0];
        for (int i = 1; i < values.Count; i++)
        {
            if (values[i] > value)
            {
                value = values[i];
            }
        }
        return value;
    }

    private static float Mean(List<float> values)
    {
        if (values == null || values.Count == 0)
        {
            return 0f;
        }

        float sum = 0f;
        for (int i = 0; i < values.Count; i++)
        {
            sum += values[i];
        }
        return sum / values.Count;
    }

    private static float Std(List<float> values, float mean)
    {
        if (values == null || values.Count == 0)
        {
            return 0f;
        }

        float sum = 0f;
        for (int i = 0; i < values.Count; i++)
        {
            float delta = values[i] - mean;
            sum += delta * delta;
        }
        return Mathf.Sqrt(sum / values.Count);
    }

    private static float Percentile(List<float> values, float percentile)
    {
        if (values == null || values.Count == 0)
        {
            return 0f;
        }

        List<float> sorted = new List<float>(values);
        sorted.Sort();

        if (sorted.Count == 1)
        {
            return sorted[0];
        }

        float position = Mathf.Clamp(percentile, 0f, 100f) / 100f * (sorted.Count - 1);
        int lower = Mathf.FloorToInt(position);
        int upper = Mathf.CeilToInt(position);
        if (lower == upper)
        {
            return sorted[lower];
        }

        float t = position - lower;
        return Mathf.Lerp(sorted[lower], sorted[upper], t);
    }

    private static float[] Softmax(float[] logits)
    {
        float[] probs = new float[logits.Length];
        if (logits.Length == 0)
        {
            return probs;
        }

        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = Mathf.Exp(logits[i] - max);
            sum += probs[i];
        }

        if (sum <= 1e-6f)
        {
            return probs;
        }

        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sum;
        }

        return probs;
    }

    private static float Sigmoid(float value)
    {
        if (value >= 0f)
        {
            float exp = Mathf.Exp(-value);
            return 1f / (1f + exp);
        }

        float expNeg = Mathf.Exp(value);
        return expNeg / (1f + expNeg);
    }

    private static int[] GetTopIndices(float[] values, int k)
    {
        if (values == null || values.Length == 0 || k <= 0)
        {
            return new int[0];
        }

        int resultCount = Mathf.Min(k, values.Length);
        int[] indices = new int[resultCount];
        float[] scores = new float[resultCount];

        for (int i = 0; i < resultCount; i++)
        {
            indices[i] = InvalidAoi;
            scores[i] = float.NegativeInfinity;
        }

        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            for (int slot = 0; slot < resultCount; slot++)
            {
                if (value > scores[slot])
                {
                    for (int move = resultCount - 1; move > slot; move--)
                    {
                        scores[move] = scores[move - 1];
                        indices[move] = indices[move - 1];
                    }
                    scores[slot] = value;
                    indices[slot] = i;
                    break;
                }
            }
        }

        return indices;
    }
}
