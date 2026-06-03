using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class RecommendationFeedbackManager : MonoBehaviour
{
    public event Action<RecommendationResult> OnRecommendationDisplayed;

    [Header("Dependencies")]
    public RealtimeIntentionRecommender recommender;
    public OnnxIntentionInference transformerInference;
    public RandomForestIntentionInference randomForestInference;
    public EyeTrackingManager eyeTrackingManager;
    public TaskManager taskManager;

    [Header("Feedback")]
    public string preferredSource = "random_forest";
    public bool fallbackToTransformer = true;
    public bool fallbackToBayes = true;
    public bool fallbackToRule = true;
    public float holdSeconds = 3.5f;
    public bool requireActiveTaskForFeedback = true;
    public bool filterFeedbackToCurrentTask = true;
    public bool suppressFeedbackWhenNoTruthMatch = true;
    public bool selectBestAgainstCurrentTask = true;
    public bool enableNextTargetFilter = true;
    public int minDisplayRecommendations = 2;
    public int maxDisplayRecommendations = 3;
    public bool enableClusterLayout = false;
    public bool preserveOriginalGridWhenMoving = true;
    public int gridColumns = 5;
    public Vector2 clusterCenter01 = new Vector2(0.5f, 0.5f);
    public float clusterSpacing01 = 0.13f;
    public float layoutLerpSpeed = 8f;

    private readonly Dictionary<int, ControlItem> controlsById = new Dictionary<int, ControlItem>();
    private readonly Dictionary<RectTransform, Vector2> originalAnchoredPositions = new Dictionary<RectTransform, Vector2>();
    private readonly Dictionary<string, RecommendationResult> latestBySource = new Dictionary<string, RecommendationResult>();
    private readonly List<ControlItem> activeRecommendedControls = new List<ControlItem>();
    private readonly List<RectTransform> activeClusterRects = new List<RectTransform>();
    private readonly List<ControlItem> orderedControls = new List<ControlItem>();
    private readonly List<RectTransform> orderedRects = new List<RectTransform>();
    private readonly Dictionary<RectTransform, Vector2> activeLayoutTargets = new Dictionary<RectTransform, Vector2>();

    private float activeUntil = -1f;
    private RecommendationResult activeResult;
    private bool initialized;
    private bool subscribed;

    public RecommendationResult ActiveResult => activeResult;

    private void Start()
    {
        InitializeIfNeeded();
        Subscribe();
    }

    private void OnEnable()
    {
        InitializeIfNeeded();
        Subscribe();
    }

    private void OnDisable()
    {
        Unsubscribe();
        ClearFeedback();
    }

    private void Update()
    {
        if (!initialized)
        {
            InitializeIfNeeded();
        }

        if (activeResult != null && Time.time > activeUntil)
        {
            ClearFeedback();
        }
    }

    private void InitializeIfNeeded()
    {
        if (initialized)
        {
            return;
        }

        if (recommender == null) recommender = FindObjectOfType<RealtimeIntentionRecommender>();
        if (transformerInference == null) transformerInference = FindObjectOfType<OnnxIntentionInference>();
        if (randomForestInference == null) randomForestInference = FindObjectOfType<RandomForestIntentionInference>();
        if (eyeTrackingManager == null) eyeTrackingManager = FindObjectOfType<EyeTrackingManager>();
        if (taskManager == null) taskManager = FindObjectOfType<TaskManager>();

        RebuildControlCache();
        initialized = true;
    }

    private void Subscribe()
    {
        if (!initialized || subscribed)
        {
            return;
        }

        if (recommender != null) recommender.OnWindowFinalized += OnRecommenderWindow;
        if (transformerInference != null) transformerInference.OnWindowFinalized += OnTransformerWindow;
        if (randomForestInference != null) randomForestInference.OnWindowFinalized += OnRandomForestWindow;
        if (taskManager != null)
        {
            taskManager.OnTaskEvaluationStarted += OnTaskChanged;
            taskManager.OnTaskEvaluationEnded += OnTaskChanged;
            taskManager.OnExperimentFinished += OnExperimentFinished;
        }
        subscribed = true;
    }

    private void Unsubscribe()
    {
        if (!subscribed)
        {
            return;
        }

        if (recommender != null) recommender.OnWindowFinalized -= OnRecommenderWindow;
        if (transformerInference != null) transformerInference.OnWindowFinalized -= OnTransformerWindow;
        if (randomForestInference != null) randomForestInference.OnWindowFinalized -= OnRandomForestWindow;
        if (taskManager != null)
        {
            taskManager.OnTaskEvaluationStarted -= OnTaskChanged;
            taskManager.OnTaskEvaluationEnded -= OnTaskChanged;
            taskManager.OnExperimentFinished -= OnExperimentFinished;
        }
        subscribed = false;
    }

    private void RebuildControlCache()
    {
        controlsById.Clear();
        orderedControls.Clear();
        orderedRects.Clear();
        List<ControlItem> controls = null;
        if (eyeTrackingManager != null && eyeTrackingManager.controlItems != null && eyeTrackingManager.controlItems.Count > 0)
        {
            controls = eyeTrackingManager.controlItems;
        }
        else if (taskManager != null && taskManager.allControls != null && taskManager.allControls.Count > 0)
        {
            controls = taskManager.allControls;
        }

        if (controls == null)
        {
            return;
        }

        ControlIdUtility.NormalizeLegacyOneBasedIds(controls, "RecommendationFeedbackManager");
        for (int i = 0; i < controls.Count; i++)
        {
            ControlItem item = controls[i];
            if (item == null || item.controlID < 0 || controlsById.ContainsKey(item.controlID))
            {
                continue;
            }
            controlsById[item.controlID] = item;
            orderedControls.Add(item);
            RectTransform rt = item.GetComponent<RectTransform>();
            if (rt != null && !originalAnchoredPositions.ContainsKey(rt))
            {
                originalAnchoredPositions[rt] = rt.anchoredPosition;
            }
        }

        orderedControls.Sort((a, b) => a.controlID.CompareTo(b.controlID));
        for (int i = 0; i < orderedControls.Count; i++)
        {
            RectTransform rt = orderedControls[i] != null ? orderedControls[i].GetComponent<RectTransform>() : null;
            if (rt != null)
            {
                orderedRects.Add(rt);
            }
        }
    }

    private void OnTaskChanged(TaskManager.TaskEvaluationSnapshot snapshot)
    {
        latestBySource.Clear();
        ClearFeedback();
    }

    private void OnExperimentFinished()
    {
        latestBySource.Clear();
        ClearFeedback();
    }

    private void OnRecommenderWindow(RealtimeIntentionRecommender.WindowDecision decision)
    {
        if (decision == null || !decision.decisionReady)
        {
            return;
        }

        RecommendationResult rule = new RecommendationResult("rule", decision.windowStartTsMs, decision.windowEndTsMs);
        AddTop(rule, decision.ruleTop1AoiId, decision.ruleTop1Hits);
        AddTop(rule, decision.ruleTop2AoiId, decision.ruleTop2Hits);
        AddTop(rule, decision.ruleTop3AoiId, decision.ruleTop3Hits);
        Publish(rule);

        if (decision.bayesAccepted)
        {
            RecommendationResult bayes = new RecommendationResult("bayes", decision.windowStartTsMs, decision.windowEndTsMs);
            AddTop(bayes, decision.bayesTop1AoiId, decision.bayesTop1Prob);
            AddTop(bayes, decision.bayesTop2AoiId, decision.bayesTop2Prob);
            AddTop(bayes, decision.bayesTop3AoiId, decision.bayesTop3Prob);
            Publish(bayes);
        }
    }

    private void OnTransformerWindow(OnnxIntentionInference.WindowDecision decision)
    {
        if (decision == null || !decision.decisionReady)
        {
            return;
        }

        RecommendationResult result = new RecommendationResult("transformer", decision.windowStartTsMs, decision.windowEndTsMs);
        AddTop(result, decision.top1AoiId, decision.top1Prob);
        AddTop(result, decision.top2AoiId, decision.top2Prob);
        AddTop(result, decision.top3AoiId, decision.top3Prob);
        Publish(result);
    }

    private void OnRandomForestWindow(RandomForestIntentionInference.WindowDecision decision)
    {
        if (decision == null || !decision.decisionReady)
        {
            return;
        }

        RecommendationResult result = new RecommendationResult("random_forest", decision.windowStartTsMs, decision.windowEndTsMs);
        AddTop(result, decision.top1AoiId, decision.top1Prob);
        AddTop(result, decision.top2AoiId, decision.top2Prob);
        AddTop(result, decision.top3AoiId, decision.top3Prob);
        Publish(result);
    }

    private void AddTop(RecommendationResult result, int aoiId, float score)
    {
        if (result == null || aoiId < 0 || result.rawTopIds.Contains(aoiId))
        {
            return;
        }
        result.rawTopIds.Add(aoiId);
        result.scores.Add(score);
    }

    private void Publish(RecommendationResult result)
    {
        if (result == null || result.rawTopIds.Count == 0)
        {
            return;
        }

        result.displayTopIds = BuildDisplayIds(result.rawTopIds);
        result.isPostProcessed = !SameIds(result.rawTopIds, result.displayTopIds);
        latestBySource[result.source] = result;

        RecommendationResult selected = SelectBestResult();
        if (selected == null)
        {
            return;
        }

        if (!CanShowFeedback(selected))
        {
            ClearFeedback();
            if (IsTaskActiveForFeedback())
            {
                OnRecommendationDisplayed?.Invoke(selected);
            }
            return;
        }

        ApplyFeedback(selected);
    }

    private RecommendationResult SelectBestResult()
    {
        if (selectBestAgainstCurrentTask && IsTaskActiveForFeedback())
        {
            RecommendationResult best = SelectBestResultForCurrentTask();
            if (best != null)
            {
                return best;
            }
        }

        RecommendationResult result;
        if (!string.IsNullOrEmpty(preferredSource) && latestBySource.TryGetValue(preferredSource, out result))
        {
            return result;
        }
        if (fallbackToTransformer && latestBySource.TryGetValue("transformer", out result)) return result;
        if (fallbackToBayes && latestBySource.TryGetValue("bayes", out result)) return result;
        if (fallbackToRule && latestBySource.TryGetValue("rule", out result)) return result;
        return null;
    }

    private RecommendationResult SelectBestResultForCurrentTask()
    {
        if (taskManager == null)
        {
            return null;
        }

        List<int> targets = taskManager.GetCurrentTargetControls();
        if (targets == null || targets.Count == 0)
        {
            return null;
        }

        RecommendationResult best = null;
        float bestScore = float.NegativeInfinity;

        foreach (KeyValuePair<string, RecommendationResult> pair in latestBySource)
        {
            RecommendationResult candidate = pair.Value;
            if (candidate == null || candidate.rawTopIds == null || candidate.rawTopIds.Count == 0)
            {
                continue;
            }

            if (!IsSourceEnabled(candidate.source))
            {
                continue;
            }

            List<int> displayIds = BuildDisplayIds(candidate.rawTopIds);
            candidate.displayTopIds = displayIds;
            candidate.isPostProcessed = !SameIds(candidate.rawTopIds, candidate.displayTopIds);

            int kept = 0;
            for (int i = 0; i < displayIds.Count; i++)
            {
                if (targets.Contains(displayIds[i]))
                {
                    kept++;
                }
            }

            int rawWrong = 0;
            for (int i = 0; i < candidate.rawTopIds.Count; i++)
            {
                if (!targets.Contains(candidate.rawTopIds[i]))
                {
                    rawWrong++;
                }
            }

            float sourceBonus = GetSourcePriority(candidate.source) * 0.01f;
            float score = kept * 10f - rawWrong + sourceBonus;
            if (score > bestScore)
            {
                bestScore = score;
                best = candidate;
            }
        }

        return best;
    }

    private bool IsSourceEnabled(string source)
    {
        if (string.Equals(source, "random_forest", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }
        if (string.Equals(source, "transformer", StringComparison.OrdinalIgnoreCase))
        {
            return fallbackToTransformer;
        }
        if (string.Equals(source, "bayes", StringComparison.OrdinalIgnoreCase))
        {
            return fallbackToBayes;
        }
        if (string.Equals(source, "rule", StringComparison.OrdinalIgnoreCase))
        {
            return fallbackToRule;
        }
        return true;
    }

    private int GetSourcePriority(string source)
    {
        if (string.Equals(source, preferredSource, StringComparison.OrdinalIgnoreCase)) return 4;
        if (string.Equals(source, "transformer", StringComparison.OrdinalIgnoreCase)) return 3;
        if (string.Equals(source, "bayes", StringComparison.OrdinalIgnoreCase)) return 2;
        if (string.Equals(source, "rule", StringComparison.OrdinalIgnoreCase)) return 1;
        return 0;
    }

    private List<int> BuildDisplayIds(List<int> rawIds)
    {
        List<int> display = new List<int>();
        if (rawIds == null)
        {
            return display;
        }

        int maxCount = Mathf.Clamp(maxDisplayRecommendations, 1, 3);
        int minCount = Mathf.Clamp(minDisplayRecommendations, 1, maxCount);

        if (enableNextTargetFilter && taskManager != null)
        {
            List<int> targets = taskManager.GetCurrentTargetControls();
            int currentAoi = eyeTrackingManager != null ? eyeTrackingManager.LastAoiId : -1;
            if (targets != null && targets.Count > 0)
            {
                for (int i = 0; i < rawIds.Count && display.Count < maxCount; i++)
                {
                    int id = rawIds[i];
                    if (!targets.Contains(id) || id == currentAoi || display.Contains(id))
                    {
                        continue;
                    }

                    ControlItem item;
                    if (controlsById.TryGetValue(id, out item) && item != null && item.IsReadyForConfirm())
                    {
                        continue;
                    }
                    display.Add(id);
                }

                for (int i = 0; i < targets.Count && display.Count < minCount; i++)
                {
                    int id = targets[i];
                    if (id < 0 || id == currentAoi || display.Contains(id))
                    {
                        continue;
                    }

                    ControlItem item;
                    if (controlsById.TryGetValue(id, out item) && item != null && item.IsReadyForConfirm())
                    {
                        continue;
                    }

                    display.Add(id);
                }

                for (int i = 0; i < rawIds.Count && display.Count < maxCount; i++)
                {
                    int id = rawIds[i];
                    if (id >= 0 && targets.Contains(id) && !display.Contains(id))
                    {
                        display.Add(id);
                    }
                }

                for (int i = 0; i < targets.Count && display.Count < minCount; i++)
                {
                    int id = targets[i];
                    if (id >= 0 && !display.Contains(id))
                    {
                        display.Add(id);
                    }
                }
            }

            if (filterFeedbackToCurrentTask)
            {
                return display;
            }
        }

        if (requireActiveTaskForFeedback && !IsTaskActiveForFeedback())
        {
            return display;
        }

        for (int i = 0; i < rawIds.Count && display.Count < maxCount; i++)
        {
            int id = rawIds[i];
            if (id >= 0 && !display.Contains(id))
            {
                display.Add(id);
            }
        }

        return display;
    }

    private bool CanShowFeedback(RecommendationResult result)
    {
        if (result == null)
        {
            return false;
        }

        if (requireActiveTaskForFeedback && !IsTaskActiveForFeedback())
        {
            return false;
        }

        if (suppressFeedbackWhenNoTruthMatch && (result.displayTopIds == null || result.displayTopIds.Count == 0))
        {
            return false;
        }

        return true;
    }

    private bool IsTaskActiveForFeedback()
    {
        return taskManager != null && taskManager.HasActiveTaskForEvaluation;
    }

    private bool SameIds(List<int> a, List<int> b)
    {
        if (a == null || b == null || a.Count != b.Count)
        {
            return false;
        }
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }

    private void ApplyFeedback(RecommendationResult result)
    {
        if (result == null)
        {
            return;
        }

        if (controlsById.Count == 0)
        {
            RebuildControlCache();
        }

        ClearControlHighlights();
        activeResult = result;
        activeUntil = Time.time + Mathf.Max(0.2f, holdSeconds);

        RefreshAoiCaches();
        OnRecommendationDisplayed?.Invoke(result);
    }

    private float GetScoreForId(RecommendationResult result, int id)
    {
        if (result == null || result.rawTopIds == null || result.scores == null)
        {
            return 0f;
        }

        int index = result.rawTopIds.IndexOf(id);
        return index >= 0 && index < result.scores.Count ? Mathf.Clamp01(result.scores[index]) : 0f;
    }

    private void ClearFeedback()
    {
        ClearControlHighlights();
        activeResult = null;
        activeUntil = -1f;
        RefreshAoiCaches();
    }

    private void ClearControlHighlights()
    {
        for (int i = 0; i < activeRecommendedControls.Count; i++)
        {
            if (activeRecommendedControls[i] != null)
            {
                activeRecommendedControls[i].ClearRecommendation();
            }
        }
        activeRecommendedControls.Clear();
    }

    private void UpdateClusterMotion()
    {
        if (!enableClusterLayout)
        {
            ReturnAllClusteredControls();
            return;
        }

        if (activeResult == null || activeClusterRects.Count == 0 || eyeTrackingManager == null || eyeTrackingManager.uiPlaneRect == null)
        {
            ReturnAllClusteredControls();
            return;
        }

        if (preserveOriginalGridWhenMoving)
        {
            UpdateGridSlotMotion();
            return;
        }

        RectTransform plane = eyeTrackingManager.uiPlaneRect;
        Rect rect = plane.rect;
        float spacing = Mathf.Max(8f, Mathf.Min(rect.width, rect.height) * clusterSpacing01);
        Vector2 center = new Vector2(
            Mathf.Lerp(rect.xMin, rect.xMax, Mathf.Clamp01(clusterCenter01.x)),
            Mathf.Lerp(rect.yMin, rect.yMax, Mathf.Clamp01(clusterCenter01.y)));

        int count = activeClusterRects.Count;
        for (int i = 0; i < count; i++)
        {
            RectTransform rt = activeClusterRects[i];
            if (rt == null)
            {
                continue;
            }

            float offset = (i - (count - 1) * 0.5f) * spacing;
            Vector2 target = center + new Vector2(offset, 0f);
            rt.anchoredPosition = Vector2.Lerp(rt.anchoredPosition, target, Time.deltaTime * layoutLerpSpeed);
        }
    }

    private void ReturnAllClusteredControls()
    {
        foreach (KeyValuePair<RectTransform, Vector2> pair in originalAnchoredPositions)
        {
            RectTransform rt = pair.Key;
            if (rt == null)
            {
                continue;
            }
            rt.anchoredPosition = Vector2.Lerp(rt.anchoredPosition, pair.Value, Time.deltaTime * layoutLerpSpeed);
        }
    }

    private void BuildActiveLayoutTargets(RecommendationResult result)
    {
        activeLayoutTargets.Clear();
        if (!preserveOriginalGridWhenMoving || result == null || result.displayTopIds == null || result.displayTopIds.Count == 0)
        {
            return;
        }

        if (orderedRects.Count == 0)
        {
            RebuildControlCache();
        }

        List<RectTransform> slots = GetOrderedGridSlots();
        if (slots.Count == 0)
        {
            return;
        }

        List<ControlItem> recommended = new List<ControlItem>();
        for (int i = 0; i < result.displayTopIds.Count; i++)
        {
            ControlItem item;
            if (controlsById.TryGetValue(result.displayTopIds[i], out item) && item != null && !recommended.Contains(item))
            {
                recommended.Add(item);
            }
        }

        if (recommended.Count == 0)
        {
            return;
        }

        List<ControlItem> layoutOrder = new List<ControlItem>(orderedControls.Count);
        int startSlot = ChooseRecommendedStartSlot(slots.Count, recommended.Count);

        for (int i = 0; i < startSlot && layoutOrder.Count < orderedControls.Count; i++)
        {
            AddNextNonRecommended(layoutOrder, recommended);
        }

        for (int i = 0; i < recommended.Count && layoutOrder.Count < orderedControls.Count; i++)
        {
            layoutOrder.Add(recommended[i]);
        }

        while (layoutOrder.Count < orderedControls.Count)
        {
            AddNextNonRecommended(layoutOrder, recommended);
        }

        int count = Mathf.Min(layoutOrder.Count, slots.Count);
        for (int i = 0; i < count; i++)
        {
            ControlItem item = layoutOrder[i];
            RectTransform rt = item != null ? item.GetComponent<RectTransform>() : null;
            RectTransform slot = slots[i];
            if (rt == null || slot == null)
            {
                continue;
            }

            Vector2 slotPosition;
            if (originalAnchoredPositions.TryGetValue(slot, out slotPosition))
            {
                activeLayoutTargets[rt] = slotPosition;
            }
        }
    }

    private List<RectTransform> GetOrderedGridSlots()
    {
        List<RectTransform> slots = new List<RectTransform>();
        for (int i = 0; i < orderedRects.Count; i++)
        {
            RectTransform rt = orderedRects[i];
            if (rt != null && originalAnchoredPositions.ContainsKey(rt))
            {
                slots.Add(rt);
            }
        }

        slots.Sort((a, b) =>
        {
            Vector2 pa = originalAnchoredPositions[a];
            Vector2 pb = originalAnchoredPositions[b];
            if (Mathf.Abs(pa.y - pb.y) > 0.01f)
            {
                return pb.y.CompareTo(pa.y);
            }
            return pa.x.CompareTo(pb.x);
        });

        return slots;
    }

    private int ChooseRecommendedStartSlot(int slotCount, int recommendedCount)
    {
        int columns = Mathf.Max(1, gridColumns);
        int rowStart = Mathf.Max(0, columns / 2 - recommendedCount / 2);
        return Mathf.Clamp(rowStart, 0, Mathf.Max(0, slotCount - recommendedCount));
    }

    private void AddNextNonRecommended(List<ControlItem> layoutOrder, List<ControlItem> recommended)
    {
        for (int i = 0; i < orderedControls.Count; i++)
        {
            ControlItem candidate = orderedControls[i];
            if (candidate == null || recommended.Contains(candidate) || layoutOrder.Contains(candidate))
            {
                continue;
            }

            layoutOrder.Add(candidate);
            return;
        }
    }

    private void UpdateGridSlotMotion()
    {
        if (activeLayoutTargets.Count == 0)
        {
            ReturnAllClusteredControls();
            return;
        }

        foreach (KeyValuePair<RectTransform, Vector2> pair in activeLayoutTargets)
        {
            RectTransform rt = pair.Key;
            if (rt == null)
            {
                continue;
            }

            rt.anchoredPosition = Vector2.Lerp(rt.anchoredPosition, pair.Value, Time.deltaTime * layoutLerpSpeed);
        }
    }

    private void RefreshAoiCaches()
    {
        if (recommender != null)
        {
            recommender.RefreshAoiCache();
        }
    }
}
