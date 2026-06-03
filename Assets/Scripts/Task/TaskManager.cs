using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TaskManager : MonoBehaviour
{
    [Serializable]
    public class TaskEvaluationSnapshot
    {
        public int roundIndex;
        public string mode;
        public string phase;
        public long timestampMs;
        public List<int> targetControls = new List<int>();
        public string outcome;
    }

    public event Action<TaskEvaluationSnapshot> OnTaskEvaluationStarted;
    public event Action<TaskEvaluationSnapshot> OnTaskEvaluationEnded;
    public event Action OnExperimentFinished;

    [Header("Controls")]
    public List<ControlItem> allControls;

    [Header("UI References")]
    public Text taskText;
    public Text timerText;
    public Text targetControlsText;

    [Header("Pause Settings")]
    public Button pauseButton;
    public Text pauseButtonText;
    public Button confirmButton;

    [Header("Experiment Settings")]
    public Dropdown modeDropdown;
    public Dropdown roundsDropdown;
    public Button startExperimentButton;
    public GameObject settingsPanel;

    [Header("Simulation Fidelity")]
    public bool useDeterministicSimulationSeed = true;
    public int simulationSeed = 20260414;
    public Vector2 dynamicReleaseDelayRange = new Vector2(5f, 30f);
    public Vector2Int dynamicTargetCountRange = new Vector2Int(2, 5);

    [Header("Data Logger")]
    public GazeDataLogger dataLogger;

    private const float TIME_LIMIT = 90f;

    private TaskData currentTask;
    private float timer;
    private float dynamicWaitTimer;

    private int totalRounds = 3;
    private int currentRound;
    private int successCount;

    private bool taskRunning;
    private bool isWaitingNext;
    private bool isPaused;
    private bool isExperimentStarted;
    private bool isDynamicWaiting;
    private bool hasExportedCsv;

    public bool HasActiveTaskForEvaluation => taskRunning && !isDynamicWaiting && !isWaitingNext && currentTask != null && currentTask.targetControls != null && currentTask.targetControls.Count > 0;
    public int CurrentRoundIndex => currentRound;
    public string CurrentMode => (modeDropdown != null && modeDropdown.value == 1) ? "Dynamic" : "Static";

    public List<int> GetCurrentTargetControls()
    {
        if (!HasActiveTaskForEvaluation)
        {
            return new List<int>();
        }

        return new List<int>(currentTask.targetControls);
    }

    private void Start()
    {
        ControlIdUtility.NormalizeLegacyOneBasedIds(allControls, "TaskManager");

        if (pauseButton != null)
        {
            pauseButton.onClick.RemoveAllListeners();
            pauseButton.onClick.AddListener(TogglePause);
        }

        if (startExperimentButton != null)
        {
            startExperimentButton.onClick.RemoveAllListeners();
            startExperimentButton.onClick.AddListener(OnStartExperimentClicked);
        }

        if (confirmButton == null)
        {
            GameObject confirmObj = GameObject.Find("Confirm Button");
            if (confirmObj != null)
            {
                confirmButton = confirmObj.GetComponent<Button>();
            }
        }

        if (confirmButton != null)
        {
            confirmButton.onClick.RemoveAllListeners();
            confirmButton.onClick.AddListener(TryConfirmTask);
        }

        InitDropdowns();

        Time.timeScale = 1f;
        if (taskText != null) taskText.text = "请选择模式和轮次，然后点击开始实验";
        if (targetControlsText != null) targetControlsText.text = "";
        if (timerText != null) timerText.text = "0.0";

        if (pauseButton != null)
            pauseButton.interactable = false;
        if (pauseButtonText != null)
            pauseButtonText.text = "暂停";

        SetConfirmButtonInteractable(false);
    }

    private void InitDropdowns()
    {
        if (modeDropdown != null)
        {
            modeDropdown.ClearOptions();
            modeDropdown.AddOptions(new List<string> { "静态", "动态" });
            modeDropdown.value = 0;
        }

        if (roundsDropdown != null)
        {
            roundsDropdown.ClearOptions();
            List<string> options = new List<string>();
            for (int i = 1; i <= 10; i++)
            {
                options.Add(i + " 轮");
            }
            roundsDropdown.AddOptions(options);
            roundsDropdown.value = 2;
        }
    }

    private void OnStartExperimentClicked()
    {
        if (isExperimentStarted)
            return;

        BeginExperimentAfterCalibration();
    }

    private void BeginExperimentAfterCalibration()
    {
        Time.timeScale = 1f;
        isExperimentStarted = true;
        isPaused = false;
        isWaitingNext = false;
        taskRunning = false;
        isDynamicWaiting = false;
        hasExportedCsv = false;
        if (useDeterministicSimulationSeed)
        {
            UnityEngine.Random.InitState(simulationSeed);
        }

        totalRounds = (roundsDropdown != null ? roundsDropdown.value : 2) + 1;
        currentRound = 0;
        successCount = 0;

        if (modeDropdown != null) modeDropdown.interactable = false;
        if (roundsDropdown != null) roundsDropdown.interactable = false;
        if (startExperimentButton != null) startExperimentButton.interactable = false;
        if (pauseButton != null) pauseButton.interactable = true;
        if (pauseButtonText != null) pauseButtonText.text = "暂停";

        foreach (ControlItem control in allControls)
        {
            if (control == null) continue;
            control.ResetControl();
            control.SetExperimentActive(true);
        }

        SetConfirmButtonInteractable(false);
        StartNextRound();
    }

    private void Update()
    {
        if (isPaused || !isExperimentStarted)
        {
            SetConfirmButtonInteractable(false);
            return;
        }

        if (isDynamicWaiting)
        {
            dynamicWaitTimer -= Time.deltaTime;
            if (dynamicWaitTimer <= 0f)
            {
                ActivateDynamicTask();
                return;
            }

            SetConfirmButtonInteractable(false);
            return;
        }

        if (!taskRunning || isWaitingNext)
        {
            SetConfirmButtonInteractable(false);
            return;
        }

        UpdateConfirmButtonState();

        timer -= Time.deltaTime;
        if (timerText != null)
        {
            timerText.text = $"{timer:F1} s";
        }

        if (timer <= 0f)
        {
            OnTaskFailed("Timeout");
        }
    }

    private void StartNextRound()
    {
        isWaitingNext = false;
        isDynamicWaiting = false;
        SetConfirmButtonInteractable(false);

        if (currentRound >= totalRounds)
        {
            ShowFinalResult();
            return;
        }

        foreach (ControlItem control in allControls)
        {
            if (control != null)
            {
                control.ResetControl();
                control.SetExperimentActive(true);
            }
        }

        currentRound++;

        bool isDynamic = modeDropdown != null && modeDropdown.value == 1;
        string modeStr = isDynamic ? "Dynamic" : "Static";

        if (dataLogger != null)
        {
            dataLogger.StartNewRound(currentRound, modeStr);
        }

        if (isDynamic)
        {
            if (dataLogger != null)
            {
                dataLogger.SetRoundPhase("free_view");
            }

            isDynamicWaiting = true;
            taskRunning = false;
            float minDelay = Mathf.Min(dynamicReleaseDelayRange.x, dynamicReleaseDelayRange.y);
            float maxDelay = Mathf.Max(dynamicReleaseDelayRange.x, dynamicReleaseDelayRange.y);
            dynamicWaitTimer = UnityEngine.Random.Range(minDelay, maxDelay);

            if (taskText != null)
            {
                taskText.text =
                    $"第 {currentRound}/{totalRounds} 轮（动态）\n" +
                    "<color=cyan>自由观察中...</color>\n" +
                    "任务将在随机时间出现";
            }

            if (targetControlsText != null)
            {
                targetControlsText.text = "等待目标控件...";
            }
        }
        else
        {
            currentTask = new TaskData
            {
                taskType = TaskType.Static,
                targetControls = GenerateRandomTargets(3),
                timeLimit = TIME_LIMIT
            };

            StartTaskExecution();
        }
    }

    private void ActivateDynamicTask()
    {
        isDynamicWaiting = false;

        int minTargets = Mathf.Clamp(Mathf.Min(dynamicTargetCountRange.x, dynamicTargetCountRange.y), 1, 5);
        int maxTargetsExclusive = Mathf.Clamp(Mathf.Max(dynamicTargetCountRange.x, dynamicTargetCountRange.y), minTargets + 1, 6);
        int targetCount = UnityEngine.Random.Range(minTargets, maxTargetsExclusive);
        currentTask = new TaskData
        {
            taskType = TaskType.Dynamic,
            targetControls = GenerateRandomTargets(targetCount),
            timeLimit = TIME_LIMIT
        };

        StartTaskExecution();
    }

    private void StartTaskExecution()
    {
        timer = currentTask.timeLimit;
        taskRunning = true;
        SetConfirmButtonInteractable(false);

        if (dataLogger != null)
        {
            dataLogger.UpdateCurrentTaskTargets(currentTask.targetControls, "task_active");
        }

        OnTaskEvaluationStarted?.Invoke(CreateTaskEvaluationSnapshot("task_active", ""));

        UpdateTaskInfoUI();
        UpdateTargetControlsUI();
    }

    public void TryConfirmTask()
    {
        if (isPaused || !taskRunning || isWaitingNext || isDynamicWaiting)
            return;

        if (!AreAllTargetControlsReady())
        {
            SetConfirmButtonInteractable(false);
            return;
        }

        bool allTriggered = true;
        List<int> notReadyList = new List<int>();

        foreach (int id in currentTask.targetControls)
        {
            ControlItem item = allControls.Find(x => x.controlID == id);
            if (item == null || !item.IsTriggered())
            {
                allTriggered = false;
                if (item != null) notReadyList.Add(id);
            }
        }

        if (allTriggered)
        {
            OnTaskSuccess();
        }
        else
        {
            string failReason = $"控件 [{string.Join(",", notReadyList)}] 尚未就绪";
            OnTaskFailed("确认失败：" + failReason);
        }
    }

    private void OnTaskSuccess()
    {
        OnTaskEvaluationEnded?.Invoke(CreateTaskEvaluationSnapshot("success", "success"));

        if (dataLogger != null)
        {
            dataLogger.MarkRoundResult("success");
            dataLogger.EndCurrentRound();
        }

        taskRunning = false;
        SetConfirmButtonInteractable(false);
        successCount++;

        if (taskText != null)
        {
            taskText.text = $"<color=green>第 {currentRound} 轮完成</color>";
        }

        WaitAndStartNext();
    }

    private void OnTaskFailed(string reason)
    {
        OnTaskEvaluationEnded?.Invoke(CreateTaskEvaluationSnapshot("failed", "failed"));

        if (dataLogger != null)
        {
            dataLogger.MarkRoundResult("failed");
            dataLogger.EndCurrentRound();
        }

        taskRunning = false;
        SetConfirmButtonInteractable(false);

        if (taskText != null)
        {
            taskText.text = $"<color=red>第 {currentRound} 轮失败\n{reason}</color>";
        }

        WaitAndStartNext();
    }

    private void WaitAndStartNext()
    {
        isWaitingNext = true;
        Invoke(nameof(StartNextRound), 2.0f);
    }

    private void ShowFinalResult()
    {
        Time.timeScale = 1f;
        taskRunning = false;
        isExperimentStarted = false;
        isPaused = false;
        SetConfirmButtonInteractable(false);

        if (modeDropdown != null) modeDropdown.interactable = true;
        if (roundsDropdown != null) roundsDropdown.interactable = true;
        if (startExperimentButton != null) startExperimentButton.interactable = true;
        if (pauseButton != null) pauseButton.interactable = false;
        if (pauseButtonText != null) pauseButtonText.text = "暂停";

        if (timerText != null) timerText.text = "0.0";
        if (targetControlsText != null) targetControlsText.text = "实验已结束";

        float rate = totalRounds > 0 ? (float)successCount / totalRounds * 100f : 0f;

        if (taskText != null)
        {
            string modeText = (modeDropdown != null && modeDropdown.value == 0) ? "静态" : "动态";
            taskText.text =
                "实验结束\n" +
                $"模式：{modeText}\n" +
                $"成功：{successCount} / {totalRounds}\n" +
                $"成功率：{rate:F1}%";
        }

        foreach (ControlItem control in allControls)
        {
            if (control != null) control.SetExperimentActive(false);
        }

        if (dataLogger != null && !hasExportedCsv)
        {
            dataLogger.ExportToCSV();
            hasExportedCsv = true;
        }

        OnExperimentFinished?.Invoke();
    }

    private List<int> GenerateRandomTargets(int count)
    {
        List<int> result = new List<int>();
        while (result.Count < count)
        {
            int id = UnityEngine.Random.Range(0, 10);
            if (!result.Contains(id))
            {
                result.Add(id);
            }
        }

        result.Sort();
        return result;
    }

    private void UpdateTaskInfoUI()
    {
        string modeStr = (modeDropdown != null && modeDropdown.value == 0) ? "静态" : "动态";
        if (taskText != null)
        {
            taskText.text =
                $"当前任务：第 {currentRound}/{totalRounds} 轮（{modeStr}）\n" +
                $"请在 {TIME_LIMIT:F0} 秒内注视目标\n" +
                "目标触发后点击确认";
        }
    }

    private void UpdateTargetControlsUI()
    {
        string targets = currentTask != null && currentTask.targetControls != null
            ? string.Join(", ", currentTask.targetControls)
            : "";

        if (targetControlsText != null)
        {
            targetControlsText.text =
                $"<color=red><b>目标控件</b> [ {targets} ]</color>\n" +
                "全部目标达到数值>=80 且符号=1 后可确认";
        }
    }

    public void TogglePause()
    {
        if (!isExperimentStarted) return;

        SetTimerPaused(!isPaused);
    }

    public void PauseTimer()
    {
        if (!isExperimentStarted) return;

        TogglePause();
    }

    public void ResumeTimer()
    {
        if (!isExperimentStarted) return;

        SetTimerPaused(false);
    }

    private void SetTimerPaused(bool paused)
    {
        if (isPaused == paused)
        {
            return;
        }

        if (paused)
        {
            Time.timeScale = 1f;
            isPaused = true;
            SetConfirmButtonInteractable(false);
            SetExperimentControlsActive(false);

            if (taskRunning && taskText != null)
            {
                taskText.text += "\n<color=yellow>--- 已暂停 ---</color>";
            }
        }
        else
        {
            Time.timeScale = 1f;
            isPaused = false;
            SetExperimentControlsActive(true);

            if (taskRunning)
            {
                UpdateTaskInfoUI();
                UpdateConfirmButtonState();
            }
            else if (isDynamicWaiting && taskText != null)
            {
                taskText.text =
                    $"第 {currentRound}/{totalRounds} 轮（动态）\n" +
                    "<color=cyan>自由观察中...</color>";
                SetConfirmButtonInteractable(false);
            }
        }

        SetPauseButtonLabel();
    }

    private void SetExperimentControlsActive(bool active)
    {
        if (allControls == null)
        {
            return;
        }

        for (int i = 0; i < allControls.Count; i++)
        {
            if (allControls[i] != null)
            {
                allControls[i].SetExperimentActive(active);
            }
        }
    }

    private void SetPauseButtonLabel()
    {
        if (pauseButtonText != null)
        {
            pauseButtonText.text = isPaused ? "继续实验" : "暂停";
        }
    }

    private void UpdateConfirmButtonState()
    {
        bool canConfirm = taskRunning && !isPaused && !isWaitingNext && !isDynamicWaiting && AreAllTargetControlsReady();
        SetConfirmButtonInteractable(canConfirm);
    }

    private bool AreAllTargetControlsReady()
    {
        if (currentTask == null || currentTask.targetControls == null || currentTask.targetControls.Count == 0)
        {
            return false;
        }

        foreach (int id in currentTask.targetControls)
        {
            ControlItem item = allControls.Find(x => x.controlID == id);
            if (item == null || !item.IsReadyForConfirm())
            {
                return false;
            }
        }

        return true;
    }

    private void SetConfirmButtonInteractable(bool canClick)
    {
        if (confirmButton != null && confirmButton.interactable != canClick)
        {
            confirmButton.interactable = canClick;
        }
    }

    private TaskEvaluationSnapshot CreateTaskEvaluationSnapshot(string phase, string outcome)
    {
        TaskEvaluationSnapshot snapshot = new TaskEvaluationSnapshot
        {
            roundIndex = currentRound,
            mode = CurrentMode,
            phase = phase,
            timestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            outcome = outcome ?? string.Empty,
            targetControls = new List<int>()
        };

        if (currentTask != null && currentTask.targetControls != null)
        {
            snapshot.targetControls.AddRange(currentTask.targetControls);
        }

        return snapshot;
    }

    private void OnDestroy()
    {
        Time.timeScale = 1f;
    }
}
