using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class RecommendationTruthFilterDisplay : MonoBehaviour
{
    [Header("Dependencies")]
    public RecommendationFeedbackManager feedbackManager;
    public TaskManager taskManager;

    [Header("Output")]
    public Text resultText;
    public string waitingText = "推荐注视：[] 控件";
    public bool evaluateRawAlgorithmIds = false;

    private readonly Dictionary<int, ControlItem> controlsById = new Dictionary<int, ControlItem>();
    private readonly List<ControlItem> highlightedControls = new List<ControlItem>();
    private int evaluatedWindows;
    private int exactMatches;
    private int totalRecommendedIds;
    private int totalKeptIds;
    private int totalRemovedIds;
    private int totalMissingIds;
    private bool subscribed;

    private void Start()
    {
        InitializeIfNeeded();
        Subscribe();
        UpdateWaitingText();
    }

    private void OnEnable()
    {
        InitializeIfNeeded();
        Subscribe();
        UpdateWaitingText();
    }

    private void OnDisable()
    {
        Unsubscribe();
        ClearHighlightedControls();
    }

    private void InitializeIfNeeded()
    {
        if (feedbackManager == null) feedbackManager = FindObjectOfType<RecommendationFeedbackManager>();
        if (taskManager == null) taskManager = FindObjectOfType<TaskManager>();
        RebuildControlCache();
    }

    private void Subscribe()
    {
        if (subscribed)
        {
            return;
        }

        if (feedbackManager != null)
        {
            feedbackManager.OnRecommendationDisplayed += OnRecommendationDisplayed;
        }

        if (taskManager != null)
        {
            taskManager.OnTaskEvaluationStarted += OnTaskEvaluationStarted;
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

        if (feedbackManager != null)
        {
            feedbackManager.OnRecommendationDisplayed -= OnRecommendationDisplayed;
        }

        if (taskManager != null)
        {
            taskManager.OnTaskEvaluationStarted -= OnTaskEvaluationStarted;
            taskManager.OnExperimentFinished -= OnExperimentFinished;
        }

        subscribed = false;
    }

    private void OnTaskEvaluationStarted(TaskManager.TaskEvaluationSnapshot snapshot)
    {
        if (snapshot != null && snapshot.roundIndex <= 1)
        {
            ResetTotals();
        }

        UpdateWaitingText();
    }

    private void OnExperimentFinished()
    {
        SetResultText(new List<int>());
    }

    private void ResetTotals()
    {
        evaluatedWindows = 0;
        exactMatches = 0;
        totalRecommendedIds = 0;
        totalKeptIds = 0;
        totalRemovedIds = 0;
        totalMissingIds = 0;
    }

    private void OnRecommendationDisplayed(RecommendationResult result)
    {
        if (result == null || taskManager == null || resultText == null)
        {
            return;
        }

        List<int> targets = taskManager.GetCurrentTargetControls();
        if (targets == null || targets.Count == 0)
        {
            UpdateWaitingText();
            return;
        }

        IList<int> sourceIds = evaluateRawAlgorithmIds && result.rawTopIds != null && result.rawTopIds.Count > 0
            ? result.rawTopIds
            : result.displayTopIds;
        List<int> recommended = UniqueIds(sourceIds);
        List<int> targetIds = UniqueIds(targets);
        List<int> kept = IntersectOrdered(recommended, targetIds);
        List<int> removed = ExceptOrdered(recommended, targetIds);
        List<int> missing = ExceptOrdered(targetIds, kept);

        evaluatedWindows++;
        totalRecommendedIds += recommended.Count;
        totalKeptIds += kept.Count;
        totalRemovedIds += removed.Count;
        totalMissingIds += missing.Count;
        if (removed.Count == 0 && missing.Count == 0)
        {
            exactMatches++;
        }

        SetResultText(kept);
    }

    private void UpdateWaitingText()
    {
        if (resultText != null)
        {
            resultText.text = waitingText;
        }

        ClearHighlightedControls();
    }

    private void SetResultText(IList<int> ids)
    {
        if (resultText != null)
        {
            resultText.text = "推荐注视：" + FormatIds(ids) + " 控件";
        }

        ApplyHighlightedControls(ids);
    }

    private void RebuildControlCache()
    {
        controlsById.Clear();
        if (taskManager == null || taskManager.allControls == null)
        {
            return;
        }

        ControlIdUtility.NormalizeLegacyOneBasedIds(taskManager.allControls, "RecommendationTruthFilterDisplay");
        for (int i = 0; i < taskManager.allControls.Count; i++)
        {
            ControlItem item = taskManager.allControls[i];
            if (item == null || item.controlID < 0 || controlsById.ContainsKey(item.controlID))
            {
                continue;
            }

            controlsById[item.controlID] = item;
        }
    }

    private void ApplyHighlightedControls(IList<int> ids)
    {
        ClearHighlightedControls();

        List<int> uniqueIds = UniqueIds(ids);
        if (uniqueIds.Count == 0)
        {
            return;
        }

        if (controlsById.Count == 0)
        {
            RebuildControlCache();
        }

        for (int i = 0; i < uniqueIds.Count; i++)
        {
            ControlItem item;
            if (!controlsById.TryGetValue(uniqueIds[i], out item) || item == null)
            {
                continue;
            }

            item.FlashRecommendationHighlight(i + 1);
            highlightedControls.Add(item);
        }
    }

    private void ClearHighlightedControls()
    {
        for (int i = 0; i < highlightedControls.Count; i++)
        {
            if (highlightedControls[i] != null)
            {
                highlightedControls[i].ClearRecommendationHighlight();
            }
        }

        highlightedControls.Clear();
    }

    private string BuildTotalsText(string title)
    {
        float totalPrecision = totalRecommendedIds > 0 ? totalKeptIds / (float)totalRecommendedIds : 0f;
        float exactRate = evaluatedWindows > 0 ? exactMatches / (float)evaluatedWindows : 0f;

        return
            $"{title}: windows={evaluatedWindows}, exact={exactMatches} ({exactRate:P0})\n" +
            $"Kept={totalKeptIds}, removed={totalRemovedIds}, missing={totalMissingIds}, precision={totalPrecision:P0}";
    }

    private static List<int> UniqueIds(IList<int> ids)
    {
        List<int> result = new List<int>();
        if (ids == null)
        {
            return result;
        }

        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if (id >= 0 && !result.Contains(id))
            {
                result.Add(id);
            }
        }
        return result;
    }

    private static List<int> IntersectOrdered(IList<int> source, IList<int> allowed)
    {
        List<int> result = new List<int>();
        for (int i = 0; i < source.Count; i++)
        {
            int id = source[i];
            if (allowed.Contains(id) && !result.Contains(id))
            {
                result.Add(id);
            }
        }
        return result;
    }

    private static List<int> ExceptOrdered(IList<int> source, IList<int> excluded)
    {
        List<int> result = new List<int>();
        for (int i = 0; i < source.Count; i++)
        {
            int id = source[i];
            if (!excluded.Contains(id) && !result.Contains(id))
            {
                result.Add(id);
            }
        }
        return result;
    }

    private static string FormatIds(IList<int> ids)
    {
        return ids == null || ids.Count == 0 ? "[]" : "[" + string.Join(",", ids) + "]";
    }
}
