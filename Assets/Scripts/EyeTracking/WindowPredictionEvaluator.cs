using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class WindowPredictionEvaluator : MonoBehaviour
{
    [Serializable]
    private class TaskInterval
    {
        public int roundIndex;
        public string mode;
        public long startTsMs;
        public long endTsMs;
        public List<int> targets = new List<int>();
        public string outcome = "";
        public bool ruleLatencyRecorded;
        public bool bayesLatencyRecorded;
        public bool transformerLatencyRecorded;
        public bool randomForestLatencyRecorded;
        public long ruleFirstCorrectLatencyMs = -1;
        public long bayesFirstCorrectLatencyMs = -1;
        public long transformerFirstCorrectLatencyMs = -1;
        public long randomForestFirstCorrectLatencyMs = -1;
    }

    private class WindowEvalRow
    {
        public string sourceModel;
        public int roundIndex;
        public string mode;
        public long windowStartTsMs;
        public long windowEndTsMs;
        public float overlapRatio;
        public string targets;
        public int evaluable;
        public string skipReason;

        public int ruleTop1 = -1;
        public int ruleTop2 = -1;
        public int ruleTop3 = -1;
        public int ruleTop1Correct;
        public int ruleHit3Correct;

        public int bayesTop1 = -1;
        public int bayesTop2 = -1;
        public int bayesTop3 = -1;
        public float bayesTop1Prob;
        public int bayesAccepted;
        public int bayesTop1Correct;
        public int bayesHit3Correct;

        public int transformerTop1 = -1;
        public int transformerTop2 = -1;
        public int transformerTop3 = -1;
        public float transformerTop1Prob;
        public int transformerPredictedCount;
        public string transformerPredicted;
        public int transformerTop1Correct;
        public int transformerHit3Correct;
        public int transformerSubsetCorrect;
        public int transformerPartialWindow;

        public int randomForestTop1 = -1;
        public int randomForestTop2 = -1;
        public int randomForestTop3 = -1;
        public float randomForestTop1Prob;
        public int randomForestPredictedCount;
        public string randomForestPredicted;
        public int randomForestTop1Correct;
        public int randomForestHit3Correct;
        public int randomForestSubsetCorrect;
    }

    [Header("Dependencies")]
    public RealtimeIntentionRecommender recommender;
    public OnnxIntentionInference transformerInference;
    public RandomForestIntentionInference randomForestInference;
    public TaskManager taskManager;
    public GazeDataLogger dataLogger;

    [Header("Evaluation")]
    [Tooltip("窗口与 task_active 区间重叠达到该比例才记为可评估窗口。")]
    public float minTaskOverlapRatio = 0.8f;
    [Tooltip("为 true 时，仅统计 recommender 已形成决策的窗口；否则连等待窗口也会落表但记为 skip。")]
    public bool requireDecisionReady = true;

    [Header("Output")]
    public Text evaluationText;
    public string outputFolderPath = @"F:\TestData";

    private readonly List<TaskInterval> taskHistory = new List<TaskInterval>();
    private readonly List<WindowEvalRow> windowRows = new List<WindowEvalRow>();
    private TaskInterval activeTask;

    private bool initialized;
    private bool subscribed;
    private string sessionStamp;
    private string resolvedFolderPath;
    private string windowsCsvPath;
    private string summaryCsvPath;
    private bool exported;

    private int recommenderTotalWindows;
    private int recommenderEvaluatedWindows;
    private int recommenderSkippedWindows;
    private int transformerTotalWindows;
    private int transformerEvaluatedWindows;
    private int transformerSkippedWindows;
    private int randomForestTotalWindows;
    private int randomForestEvaluatedWindows;
    private int randomForestSkippedWindows;
    private int ruleTop1Correct;
    private int ruleHit3Correct;
    private int bayesAcceptedWindows;
    private int bayesTop1Correct;
    private int bayesHit3Correct;
    private int transformerTop1Correct;
    private int transformerHit3Correct;
    private int transformerSubsetCorrect;
    private int transformerPartialWindows;
    private int randomForestTop1Correct;
    private int randomForestHit3Correct;
    private int randomForestSubsetCorrect;
    private int ruleLatencyTaskCount;
    private int bayesLatencyTaskCount;
    private int transformerLatencyTaskCount;
    private int randomForestLatencyTaskCount;
    private long ruleLatencySumMs;
    private long bayesLatencySumMs;
    private long transformerLatencySumMs;
    private long randomForestLatencySumMs;

    private void Start()
    {
        InitializeIfNeeded();
        Subscribe();
        RefreshEvaluationText();
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

    private void OnApplicationQuit()
    {
        ExportIfNeeded();
    }

    private void InitializeIfNeeded()
    {
        if (initialized)
        {
            return;
        }

        if (recommender == null)
        {
            recommender = FindObjectOfType<RealtimeIntentionRecommender>();
        }
        if (transformerInference == null)
        {
            transformerInference = FindObjectOfType<OnnxIntentionInference>();
        }
        if (taskManager == null)
        {
            taskManager = FindObjectOfType<TaskManager>();
        }
        if (randomForestInference == null)
        {
            randomForestInference = FindObjectOfType<RandomForestIntentionInference>();
        }
        if (dataLogger == null)
        {
            dataLogger = FindObjectOfType<GazeDataLogger>();
        }

        if (recommender == null || taskManager == null || transformerInference == null)
        {
            Debug.LogWarning("[WindowPredictionEvaluator] Missing recommender, transformerInference, or taskManager, evaluator disabled.");
            return;
        }

        sessionStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        resolvedFolderPath = ResolveAndCreateFolder(string.IsNullOrEmpty(outputFolderPath)
            ? (dataLogger != null ? dataLogger.folderPath : string.Empty)
            : outputFolderPath);

        string subjectId = dataLogger != null ? dataLogger.subjectId : "UnknownSubject";
        string sessionId = dataLogger != null ? dataLogger.sessionId : "UnknownSession";
        windowsCsvPath = Path.Combine(resolvedFolderPath, $"EvalWindows_{subjectId}_{sessionId}_{sessionStamp}.csv");
        summaryCsvPath = Path.Combine(resolvedFolderPath, $"EvalSummary_{subjectId}_{sessionId}_{sessionStamp}.csv");

        initialized = true;
    }

    private void Subscribe()
    {
        if (!initialized || subscribed)
        {
            return;
        }

        recommender.OnWindowFinalized += OnWindowFinalized;
        transformerInference.OnWindowFinalized += OnTransformerWindowFinalized;
        if (randomForestInference != null)
        {
            randomForestInference.OnWindowFinalized += OnRandomForestWindowFinalized;
        }
        taskManager.OnTaskEvaluationStarted += OnTaskEvaluationStarted;
        taskManager.OnTaskEvaluationEnded += OnTaskEvaluationEnded;
        taskManager.OnExperimentFinished += OnExperimentFinished;
        subscribed = true;
    }

    private void Unsubscribe()
    {
        if (!subscribed)
        {
            return;
        }

        if (recommender != null)
        {
            recommender.OnWindowFinalized -= OnWindowFinalized;
        }
        if (transformerInference != null)
        {
            transformerInference.OnWindowFinalized -= OnTransformerWindowFinalized;
        }
        if (randomForestInference != null)
        {
            randomForestInference.OnWindowFinalized -= OnRandomForestWindowFinalized;
        }
        if (taskManager != null)
        {
            taskManager.OnTaskEvaluationStarted -= OnTaskEvaluationStarted;
            taskManager.OnTaskEvaluationEnded -= OnTaskEvaluationEnded;
            taskManager.OnExperimentFinished -= OnExperimentFinished;
        }
        subscribed = false;
    }

    private void OnTaskEvaluationStarted(TaskManager.TaskEvaluationSnapshot snapshot)
    {
        if (snapshot == null)
        {
            return;
        }

        activeTask = new TaskInterval
        {
            roundIndex = snapshot.roundIndex,
            mode = snapshot.mode ?? "",
            startTsMs = snapshot.timestampMs,
            endTsMs = long.MaxValue,
            outcome = ""
        };
        if (snapshot.targetControls != null)
        {
            activeTask.targets.AddRange(snapshot.targetControls);
        }
    }

    private void OnTaskEvaluationEnded(TaskManager.TaskEvaluationSnapshot snapshot)
    {
        if (snapshot == null)
        {
            return;
        }

        if (activeTask == null)
        {
            activeTask = new TaskInterval
            {
                roundIndex = snapshot.roundIndex,
                mode = snapshot.mode ?? "",
                startTsMs = snapshot.timestampMs,
                endTsMs = snapshot.timestampMs
            };
            if (snapshot.targetControls != null)
            {
                activeTask.targets.AddRange(snapshot.targetControls);
            }
        }

        activeTask.endTsMs = snapshot.timestampMs;
        activeTask.outcome = snapshot.outcome ?? "";
        taskHistory.Add(activeTask);
        activeTask = null;
    }

    private void OnExperimentFinished()
    {
        ExportIfNeeded();
    }

    private void OnWindowFinalized(RealtimeIntentionRecommender.WindowDecision decision)
    {
        if (decision == null)
        {
            return;
        }

        recommenderTotalWindows++;

        WindowEvalRow row = new WindowEvalRow
        {
            sourceModel = "recommender",
            roundIndex = -1,
            mode = "",
            windowStartTsMs = decision.windowStartTsMs,
            windowEndTsMs = decision.windowEndTsMs,
            overlapRatio = 0f,
            targets = "",
            evaluable = 0,
            skipReason = "",
            ruleTop1 = decision.ruleTop1AoiId,
            ruleTop2 = decision.ruleTop2AoiId,
            ruleTop3 = decision.ruleTop3AoiId,
            bayesTop1 = decision.bayesTop1AoiId,
            bayesTop2 = decision.bayesTop2AoiId,
            bayesTop3 = decision.bayesTop3AoiId,
            bayesTop1Prob = decision.bayesTop1Prob,
            bayesAccepted = decision.bayesAccepted ? 1 : 0
        };

        if (!decision.decisionReady && requireDecisionReady)
        {
            row.skipReason = string.IsNullOrEmpty(decision.skipReason) ? "decision_not_ready" : decision.skipReason;
            MarkSkipped(row);
            return;
        }

        TaskInterval matchedTask;
        float overlapRatio;
        if (!TryFindBestTaskInterval(decision.windowStartTsMs, decision.windowEndTsMs, out matchedTask, out overlapRatio))
        {
            row.skipReason = "no_matching_task";
            MarkSkipped(row);
            return;
        }

        row.roundIndex = matchedTask.roundIndex;
        row.mode = matchedTask.mode ?? "";
        row.overlapRatio = overlapRatio;
        row.targets = JoinTargets(matchedTask.targets);

        if (overlapRatio < Mathf.Clamp01(minTaskOverlapRatio))
        {
            row.skipReason = "mixed_window_overlap_low";
            MarkSkipped(row);
            return;
        }

        if (matchedTask.targets == null || matchedTask.targets.Count == 0)
        {
            row.skipReason = "empty_targets";
            MarkSkipped(row);
            return;
        }

        recommenderEvaluatedWindows++;
        row.evaluable = 1;

        row.ruleTop1Correct = IsTargetHit(decision.ruleTop1AoiId, matchedTask.targets) ? 1 : 0;
        row.ruleHit3Correct = IsTargetHit(decision.ruleTop1AoiId, matchedTask.targets)
                              || IsTargetHit(decision.ruleTop2AoiId, matchedTask.targets)
                              || IsTargetHit(decision.ruleTop3AoiId, matchedTask.targets)
            ? 1
            : 0;

        bool bayesTop1Hit = decision.bayesAccepted && IsTargetHit(decision.bayesTop1AoiId, matchedTask.targets);
        bool bayesHit3 = decision.bayesAccepted &&
                         (IsTargetHit(decision.bayesTop1AoiId, matchedTask.targets)
                          || IsTargetHit(decision.bayesTop2AoiId, matchedTask.targets)
                          || IsTargetHit(decision.bayesTop3AoiId, matchedTask.targets));

        row.bayesTop1Correct = bayesTop1Hit ? 1 : 0;
        row.bayesHit3Correct = bayesHit3 ? 1 : 0;

        ruleTop1Correct += row.ruleTop1Correct;
        ruleHit3Correct += row.ruleHit3Correct;

        if (decision.bayesAccepted)
        {
            bayesAcceptedWindows++;
        }
        bayesTop1Correct += row.bayesTop1Correct;
        bayesHit3Correct += row.bayesHit3Correct;

        TryRecordFirstCorrectLatency(matchedTask, decision, row);

        windowRows.Add(row);
        RefreshEvaluationText();
    }

    private void OnTransformerWindowFinalized(OnnxIntentionInference.WindowDecision decision)
    {
        if (decision == null)
        {
            return;
        }

        transformerTotalWindows++;

        WindowEvalRow row = new WindowEvalRow
        {
            sourceModel = "transformer",
            roundIndex = -1,
            mode = "",
            windowStartTsMs = decision.windowStartTsMs,
            windowEndTsMs = decision.windowEndTsMs,
            overlapRatio = 0f,
            targets = "",
            evaluable = 0,
            skipReason = "",
            transformerTop1 = decision.top1AoiId,
            transformerTop2 = decision.top2AoiId,
            transformerTop3 = decision.top3AoiId,
            transformerTop1Prob = decision.top1Prob,
            transformerPredicted = JoinTargets(decision.predictedAoiIds),
            transformerPredictedCount = decision.predictedAoiIds != null ? decision.predictedAoiIds.Count : 0,
            transformerPartialWindow = decision.partialWindow ? 1 : 0
        };

        if (!decision.decisionReady && requireDecisionReady)
        {
            row.skipReason = string.IsNullOrEmpty(decision.skipReason) ? "decision_not_ready" : decision.skipReason;
            MarkSkipped(row, true);
            return;
        }

        TaskInterval matchedTask;
        float overlapRatio;
        if (!TryFindBestTaskInterval(decision.windowStartTsMs, decision.windowEndTsMs, out matchedTask, out overlapRatio))
        {
            row.skipReason = "no_matching_task";
            MarkSkipped(row, true);
            return;
        }

        row.roundIndex = matchedTask.roundIndex;
        row.mode = matchedTask.mode ?? "";
        row.overlapRatio = overlapRatio;
        row.targets = JoinTargets(matchedTask.targets);

        if (overlapRatio < Mathf.Clamp01(minTaskOverlapRatio))
        {
            row.skipReason = "mixed_window_overlap_low";
            MarkSkipped(row, true);
            return;
        }

        if (matchedTask.targets == null || matchedTask.targets.Count == 0)
        {
            row.skipReason = "empty_targets";
            MarkSkipped(row, true);
            return;
        }

        transformerEvaluatedWindows++;
        row.evaluable = 1;
        row.transformerTop1Correct = IsTargetHit(decision.top1AoiId, matchedTask.targets) ? 1 : 0;
        row.transformerHit3Correct =
            IsTargetHit(decision.top1AoiId, matchedTask.targets)
            || IsTargetHit(decision.top2AoiId, matchedTask.targets)
            || IsTargetHit(decision.top3AoiId, matchedTask.targets)
                ? 1
                : 0;
        row.transformerSubsetCorrect = IsExactTargetSetMatch(decision.predictedAoiIds, matchedTask.targets) ? 1 : 0;

        transformerTop1Correct += row.transformerTop1Correct;
        transformerHit3Correct += row.transformerHit3Correct;
        transformerSubsetCorrect += row.transformerSubsetCorrect;
        if (decision.partialWindow)
        {
            transformerPartialWindows++;
        }

        TryRecordTransformerLatency(matchedTask, decision, row);

        windowRows.Add(row);
        RefreshEvaluationText();
    }

    private void OnRandomForestWindowFinalized(RandomForestIntentionInference.WindowDecision decision)
    {
        if (decision == null)
        {
            return;
        }

        randomForestTotalWindows++;

        WindowEvalRow row = new WindowEvalRow
        {
            sourceModel = "random_forest",
            roundIndex = -1,
            mode = "",
            windowStartTsMs = decision.windowStartTsMs,
            windowEndTsMs = decision.windowEndTsMs,
            overlapRatio = 0f,
            targets = "",
            evaluable = 0,
            skipReason = "",
            randomForestTop1 = decision.top1AoiId,
            randomForestTop2 = decision.top2AoiId,
            randomForestTop3 = decision.top3AoiId,
            randomForestTop1Prob = decision.top1Prob,
            randomForestPredicted = JoinTargets(decision.predictedAoiIds),
            randomForestPredictedCount = decision.predictedAoiIds != null ? decision.predictedAoiIds.Count : 0
        };

        if (!decision.decisionReady && requireDecisionReady)
        {
            row.skipReason = string.IsNullOrEmpty(decision.skipReason) ? "decision_not_ready" : decision.skipReason;
            MarkSkippedRandomForest(row);
            return;
        }

        TaskInterval matchedTask;
        float overlapRatio;
        if (!TryFindBestTaskInterval(decision.windowStartTsMs, decision.windowEndTsMs, out matchedTask, out overlapRatio))
        {
            row.skipReason = "no_matching_task";
            MarkSkippedRandomForest(row);
            return;
        }

        row.roundIndex = matchedTask.roundIndex;
        row.mode = matchedTask.mode ?? "";
        row.overlapRatio = overlapRatio;
        row.targets = JoinTargets(matchedTask.targets);

        if (overlapRatio < Mathf.Clamp01(minTaskOverlapRatio))
        {
            row.skipReason = "mixed_window_overlap_low";
            MarkSkippedRandomForest(row);
            return;
        }

        if (matchedTask.targets == null || matchedTask.targets.Count == 0)
        {
            row.skipReason = "empty_targets";
            MarkSkippedRandomForest(row);
            return;
        }

        randomForestEvaluatedWindows++;
        row.evaluable = 1;
        row.randomForestTop1Correct = IsTargetHit(decision.top1AoiId, matchedTask.targets) ? 1 : 0;
        row.randomForestHit3Correct =
            IsTargetHit(decision.top1AoiId, matchedTask.targets)
            || IsTargetHit(decision.top2AoiId, matchedTask.targets)
            || IsTargetHit(decision.top3AoiId, matchedTask.targets)
                ? 1
                : 0;
        row.randomForestSubsetCorrect = IsExactTargetSetMatch(decision.predictedAoiIds, matchedTask.targets) ? 1 : 0;

        randomForestTop1Correct += row.randomForestTop1Correct;
        randomForestHit3Correct += row.randomForestHit3Correct;
        randomForestSubsetCorrect += row.randomForestSubsetCorrect;

        TryRecordRandomForestLatency(matchedTask, decision, row);

        windowRows.Add(row);
        RefreshEvaluationText();
    }

    private void TryRecordFirstCorrectLatency(TaskInterval task, RealtimeIntentionRecommender.WindowDecision decision, WindowEvalRow row)
    {
        if (task == null)
        {
            return;
        }

        long latencyMs = Math.Max(0L, decision.windowEndTsMs - task.startTsMs);

        if (row.ruleTop1Correct == 1 && !task.ruleLatencyRecorded)
        {
            task.ruleLatencyRecorded = true;
            task.ruleFirstCorrectLatencyMs = latencyMs;
            ruleLatencyTaskCount++;
            ruleLatencySumMs += latencyMs;
        }

        if (row.bayesTop1Correct == 1 && !task.bayesLatencyRecorded)
        {
            task.bayesLatencyRecorded = true;
            task.bayesFirstCorrectLatencyMs = latencyMs;
            bayesLatencyTaskCount++;
            bayesLatencySumMs += latencyMs;
        }
    }

    private void TryRecordTransformerLatency(TaskInterval task, OnnxIntentionInference.WindowDecision decision, WindowEvalRow row)
    {
        if (task == null)
        {
            return;
        }

        long latencyMs = Math.Max(0L, decision.windowEndTsMs - task.startTsMs);
        if (row.transformerTop1Correct == 1 && !task.transformerLatencyRecorded)
        {
            task.transformerLatencyRecorded = true;
            task.transformerFirstCorrectLatencyMs = latencyMs;
            transformerLatencyTaskCount++;
            transformerLatencySumMs += latencyMs;
        }
    }

    private void TryRecordRandomForestLatency(TaskInterval task, RandomForestIntentionInference.WindowDecision decision, WindowEvalRow row)
    {
        if (task == null)
        {
            return;
        }

        long latencyMs = Math.Max(0L, decision.windowEndTsMs - task.startTsMs);
        if (row.randomForestTop1Correct == 1 && !task.randomForestLatencyRecorded)
        {
            task.randomForestLatencyRecorded = true;
            task.randomForestFirstCorrectLatencyMs = latencyMs;
            randomForestLatencyTaskCount++;
            randomForestLatencySumMs += latencyMs;
        }
    }

    private void MarkSkipped(WindowEvalRow row, bool isTransformer = false)
    {
        if (isTransformer)
        {
            transformerSkippedWindows++;
        }
        else
        {
            recommenderSkippedWindows++;
        }
        windowRows.Add(row);
        RefreshEvaluationText();
    }

    private void MarkSkippedRandomForest(WindowEvalRow row)
    {
        randomForestSkippedWindows++;
        windowRows.Add(row);
        RefreshEvaluationText();
    }

    private bool TryFindBestTaskInterval(long windowStartTsMs, long windowEndTsMs, out TaskInterval bestTask, out float bestOverlapRatio)
    {
        bestTask = null;
        bestOverlapRatio = 0f;

        if (windowStartTsMs <= 0 || windowEndTsMs <= windowStartTsMs)
        {
            return false;
        }

        EvaluateOverlap(windowStartTsMs, windowEndTsMs, activeTask, ref bestTask, ref bestOverlapRatio);

        for (int i = 0; i < taskHistory.Count; i++)
        {
            EvaluateOverlap(windowStartTsMs, windowEndTsMs, taskHistory[i], ref bestTask, ref bestOverlapRatio);
        }

        return bestTask != null;
    }

    private void EvaluateOverlap(long windowStartTsMs, long windowEndTsMs, TaskInterval task, ref TaskInterval bestTask, ref float bestOverlapRatio)
    {
        if (task == null)
        {
            return;
        }

        long taskEnd = task.endTsMs == long.MaxValue ? windowEndTsMs : task.endTsMs;
        long overlapMs = Math.Min(windowEndTsMs, taskEnd) - Math.Max(windowStartTsMs, task.startTsMs);
        if (overlapMs <= 0)
        {
            return;
        }

        long windowMs = Math.Max(1, windowEndTsMs - windowStartTsMs);
        float ratio = overlapMs / (float)windowMs;
        if (ratio > bestOverlapRatio)
        {
            bestOverlapRatio = ratio;
            bestTask = task;
        }
    }

    private bool IsTargetHit(int aoiId, List<int> targets)
    {
        return aoiId >= 0 && targets != null && targets.Contains(aoiId);
    }

    private bool IsExactTargetSetMatch(List<int> predicted, List<int> targets)
    {
        if (targets == null || targets.Count == 0)
        {
            return predicted == null || predicted.Count == 0;
        }

        if (predicted == null)
        {
            return false;
        }

        HashSet<int> predictedSet = new HashSet<int>(predicted);
        HashSet<int> targetSet = new HashSet<int>(targets);
        if (predictedSet.Count != targetSet.Count)
        {
            return false;
        }

        foreach (int target in targetSet)
        {
            if (!predictedSet.Contains(target))
            {
                return false;
            }
        }

        return true;
    }

    private string JoinTargets(List<int> targets)
    {
        if (targets == null || targets.Count == 0)
        {
            return "";
        }

        return string.Join("|", targets);
    }

    private void RefreshEvaluationText()
    {
        if (evaluationText == null)
        {
            return;
        }

        float ruleAcc = recommenderEvaluatedWindows > 0 ? (float)ruleTop1Correct / recommenderEvaluatedWindows : 0f;
        float bayesAcc = recommenderEvaluatedWindows > 0 ? (float)bayesTop1Correct / recommenderEvaluatedWindows : 0f;
        float bayesCoverage = recommenderEvaluatedWindows > 0 ? (float)bayesAcceptedWindows / recommenderEvaluatedWindows : 0f;
        float transformerAcc = transformerEvaluatedWindows > 0 ? (float)transformerTop1Correct / transformerEvaluatedWindows : 0f;
        float transformerHit3 = transformerEvaluatedWindows > 0 ? (float)transformerHit3Correct / transformerEvaluatedWindows : 0f;
        float transformerSubset = transformerEvaluatedWindows > 0 ? (float)transformerSubsetCorrect / transformerEvaluatedWindows : 0f;
        float randomForestAcc = randomForestEvaluatedWindows > 0 ? (float)randomForestTop1Correct / randomForestEvaluatedWindows : 0f;
        float randomForestHit3 = randomForestEvaluatedWindows > 0 ? (float)randomForestHit3Correct / randomForestEvaluatedWindows : 0f;
        float randomForestSubset = randomForestEvaluatedWindows > 0 ? (float)randomForestSubsetCorrect / randomForestEvaluatedWindows : 0f;

        evaluationText.text =
            $"Recommender Eval: {recommenderEvaluatedWindows}/{recommenderTotalWindows}\n" +
            $"Transformer Eval: {transformerEvaluatedWindows}/{transformerTotalWindows}\n" +
            $"RandomForest Eval: {randomForestEvaluatedWindows}/{randomForestTotalWindows}\n" +
            $"Rule Top1: {ruleTop1Correct}/{Mathf.Max(1, recommenderEvaluatedWindows)} ({ruleAcc:P1})\n" +
            $"Bayes Top1: {bayesTop1Correct}/{Mathf.Max(1, recommenderEvaluatedWindows)} ({bayesAcc:P1})\n" +
            $"Bayes Coverage: {bayesAcceptedWindows}/{Mathf.Max(1, recommenderEvaluatedWindows)} ({bayesCoverage:P1})\n" +
            $"Transformer Top1: {transformerTop1Correct}/{Mathf.Max(1, transformerEvaluatedWindows)} ({transformerAcc:P1})\n" +
            $"Transformer Hit@3: {transformerHit3Correct}/{Mathf.Max(1, transformerEvaluatedWindows)} ({transformerHit3:P1})\n" +
            $"Transformer Subset: {transformerSubsetCorrect}/{Mathf.Max(1, transformerEvaluatedWindows)} ({transformerSubset:P1})\n" +
            $"RF Top1: {randomForestTop1Correct}/{Mathf.Max(1, randomForestEvaluatedWindows)} ({randomForestAcc:P1})\n" +
            $"RF Hit@3: {randomForestHit3Correct}/{Mathf.Max(1, randomForestEvaluatedWindows)} ({randomForestHit3:P1})\n" +
            $"RF Subset: {randomForestSubsetCorrect}/{Mathf.Max(1, randomForestEvaluatedWindows)} ({randomForestSubset:P1})";
    }

    private void ExportIfNeeded()
    {
        if (!initialized || exported)
        {
            return;
        }

        ExportWindowsCsv();
        ExportSummaryCsv();
        exported = true;
    }

    private void ExportWindowsCsv()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("source_model,round_index,mode,window_start_ms,window_end_ms,overlap_ratio,targets,evaluable,skip_reason,rule_top1,rule_top2,rule_top3,rule_top1_correct,rule_hit3_correct,bayes_top1,bayes_top2,bayes_top3,bayes_top1_prob,bayes_accepted,bayes_top1_correct,bayes_hit3_correct,transformer_top1,transformer_top2,transformer_top3,transformer_top1_prob,transformer_predicted_count,transformer_predicted,transformer_top1_correct,transformer_hit3_correct,transformer_subset_correct,transformer_partial_window,rf_top1,rf_top2,rf_top3,rf_top1_prob,rf_predicted_count,rf_predicted,rf_top1_correct,rf_hit3_correct,rf_subset_correct");

        for (int i = 0; i < windowRows.Count; i++)
        {
            WindowEvalRow row = windowRows[i];
            sb.AppendLine(
                $"{Safe(row.sourceModel)},{row.roundIndex},{Safe(row.mode)},{row.windowStartTsMs},{row.windowEndTsMs},{row.overlapRatio:F4},{Safe(row.targets)}," +
                $"{row.evaluable},{Safe(row.skipReason)},{row.ruleTop1},{row.ruleTop2},{row.ruleTop3},{row.ruleTop1Correct},{row.ruleHit3Correct}," +
                $"{row.bayesTop1},{row.bayesTop2},{row.bayesTop3},{row.bayesTop1Prob:F4},{row.bayesAccepted},{row.bayesTop1Correct},{row.bayesHit3Correct}," +
                $"{row.transformerTop1},{row.transformerTop2},{row.transformerTop3},{row.transformerTop1Prob:F4},{row.transformerPredictedCount},{Safe(row.transformerPredicted)},{row.transformerTop1Correct},{row.transformerHit3Correct},{row.transformerSubsetCorrect},{row.transformerPartialWindow}," +
                $"{row.randomForestTop1},{row.randomForestTop2},{row.randomForestTop3},{row.randomForestTop1Prob:F4},{row.randomForestPredictedCount},{Safe(row.randomForestPredicted)},{row.randomForestTop1Correct},{row.randomForestHit3Correct},{row.randomForestSubsetCorrect}");
        }

        File.WriteAllText(windowsCsvPath, sb.ToString(), Encoding.UTF8);
    }

    private void ExportSummaryCsv()
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("metric,value");
        sb.AppendLine($"recommender_total_windows,{recommenderTotalWindows}");
        sb.AppendLine($"recommender_evaluated_windows,{recommenderEvaluatedWindows}");
        sb.AppendLine($"recommender_skipped_windows,{recommenderSkippedWindows}");
        sb.AppendLine($"rule_top1_correct,{ruleTop1Correct}");
        sb.AppendLine($"rule_top1_accuracy,{Ratio(ruleTop1Correct, recommenderEvaluatedWindows):F4}");
        sb.AppendLine($"rule_hit3_correct,{ruleHit3Correct}");
        sb.AppendLine($"rule_hit3_accuracy,{Ratio(ruleHit3Correct, recommenderEvaluatedWindows):F4}");
        sb.AppendLine($"bayes_accepted_windows,{bayesAcceptedWindows}");
        sb.AppendLine($"bayes_coverage,{Ratio(bayesAcceptedWindows, recommenderEvaluatedWindows):F4}");
        sb.AppendLine($"bayes_top1_correct,{bayesTop1Correct}");
        sb.AppendLine($"bayes_top1_accuracy,{Ratio(bayesTop1Correct, recommenderEvaluatedWindows):F4}");
        sb.AppendLine($"bayes_hit3_correct,{bayesHit3Correct}");
        sb.AppendLine($"bayes_hit3_accuracy,{Ratio(bayesHit3Correct, recommenderEvaluatedWindows):F4}");
        sb.AppendLine($"rule_first_correct_latency_tasks,{ruleLatencyTaskCount}");
        sb.AppendLine($"rule_first_correct_latency_ms_avg,{AverageLatency(ruleLatencySumMs, ruleLatencyTaskCount):F2}");
        sb.AppendLine($"bayes_first_correct_latency_tasks,{bayesLatencyTaskCount}");
        sb.AppendLine($"bayes_first_correct_latency_ms_avg,{AverageLatency(bayesLatencySumMs, bayesLatencyTaskCount):F2}");
        sb.AppendLine($"transformer_total_windows,{transformerTotalWindows}");
        sb.AppendLine($"transformer_evaluated_windows,{transformerEvaluatedWindows}");
        sb.AppendLine($"transformer_skipped_windows,{transformerSkippedWindows}");
        sb.AppendLine($"transformer_top1_correct,{transformerTop1Correct}");
        sb.AppendLine($"transformer_top1_accuracy,{Ratio(transformerTop1Correct, transformerEvaluatedWindows):F4}");
        sb.AppendLine($"transformer_hit3_correct,{transformerHit3Correct}");
        sb.AppendLine($"transformer_hit3_accuracy,{Ratio(transformerHit3Correct, transformerEvaluatedWindows):F4}");
        sb.AppendLine($"transformer_subset_correct,{transformerSubsetCorrect}");
        sb.AppendLine($"transformer_subset_accuracy,{Ratio(transformerSubsetCorrect, transformerEvaluatedWindows):F4}");
        sb.AppendLine($"transformer_partial_windows,{transformerPartialWindows}");
        sb.AppendLine($"transformer_first_correct_latency_tasks,{transformerLatencyTaskCount}");
        sb.AppendLine($"transformer_first_correct_latency_ms_avg,{AverageLatency(transformerLatencySumMs, transformerLatencyTaskCount):F2}");
        sb.AppendLine($"random_forest_total_windows,{randomForestTotalWindows}");
        sb.AppendLine($"random_forest_evaluated_windows,{randomForestEvaluatedWindows}");
        sb.AppendLine($"random_forest_skipped_windows,{randomForestSkippedWindows}");
        sb.AppendLine($"random_forest_top1_correct,{randomForestTop1Correct}");
        sb.AppendLine($"random_forest_top1_accuracy,{Ratio(randomForestTop1Correct, randomForestEvaluatedWindows):F4}");
        sb.AppendLine($"random_forest_hit3_correct,{randomForestHit3Correct}");
        sb.AppendLine($"random_forest_hit3_accuracy,{Ratio(randomForestHit3Correct, randomForestEvaluatedWindows):F4}");
        sb.AppendLine($"random_forest_subset_correct,{randomForestSubsetCorrect}");
        sb.AppendLine($"random_forest_subset_accuracy,{Ratio(randomForestSubsetCorrect, randomForestEvaluatedWindows):F4}");
        sb.AppendLine($"random_forest_first_correct_latency_tasks,{randomForestLatencyTaskCount}");
        sb.AppendLine($"random_forest_first_correct_latency_ms_avg,{AverageLatency(randomForestLatencySumMs, randomForestLatencyTaskCount):F2}");

        File.WriteAllText(summaryCsvPath, sb.ToString(), Encoding.UTF8);
    }

    private float Ratio(int numerator, int denominator)
    {
        return denominator > 0 ? numerator / (float)denominator : 0f;
    }

    private float AverageLatency(long sumMs, int count)
    {
        return count > 0 ? sumMs / (float)count : 0f;
    }

    private string ResolveAndCreateFolder(string preferredFolder)
    {
        string fallbackFolder = Path.Combine(Application.persistentDataPath, "TestData");

        if (string.IsNullOrEmpty(preferredFolder))
        {
            EnsureDirectory(fallbackFolder);
            return fallbackFolder;
        }

        if (Application.platform != RuntimePlatform.WindowsEditor &&
            Application.platform != RuntimePlatform.WindowsPlayer)
        {
            if (Application.platform == RuntimePlatform.Android)
            {
                string androidDataFolder = $"/storage/emulated/0/Android/data/{Application.identifier}/files/TestData";
                try
                {
                    EnsureDirectory(androidDataFolder);
                    return androidDataFolder;
                }
                catch
                {
                }
            }

            EnsureDirectory(fallbackFolder);
            return fallbackFolder;
        }

        try
        {
            EnsureDirectory(preferredFolder);
            return preferredFolder;
        }
        catch
        {
            EnsureDirectory(fallbackFolder);
            return fallbackFolder;
        }
    }

    private void EnsureDirectory(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
    }

    private string Safe(string value)
    {
        return string.IsNullOrEmpty(value) ? "" : value.Replace(",", "_");
    }
}
