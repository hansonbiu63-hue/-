using UnityEngine;
using UnityEngine.UI;

public class ControlItem : MonoBehaviour
{
    [Header("Control Info")]
    public int controlID;                // 0~9
    public float value = 0f;
    public float triggerThreshold = 80f;

    [Header("UI References")]
    public Text valueText;
    public Text symbolText;
    public Text triggerStateText;
    public Button button;

    [Header("Dynamic Speed Settings")]
    public float maxTotalSpeed = 9.0f;
    public float accelerationRate = 2.5f;
    public float decelerationRate = 1.2f;
    [Tooltip("被注视后符号位保持为1的时长（秒）")]
    public float gazeSymbolHoldSeconds = 8.0f;

    [Header("Recommendation Highlight")]
    [SerializeField] private Color recommendationFlashColor = new Color(1f, 0.92f, 0f, 1f);
    [SerializeField] private int recommendationFlashCount = 3;
    [SerializeField] private float recommendationFlashInterval = 0.18f;

    private Image buttonImage;
    private Color normalColor = new Color(0.08f, 0.10f, 0.12f, 0.92f);
    private Color gazeColor = new Color(0.10f, 0.72f, 0.86f, 0.96f);
    private Color triggeredColor = new Color(0.34f, 0.88f, 0.52f, 0.96f);

    private float baseGrowthSpeed;
    private float currentBonusSpeed;
    private int symbolState;
    private float symbolChangeTimer;

    private float triggerTime;
    private bool isTriggered;
    private bool isExperimentActive;
    private bool isBeingLookedAt;

    private float lastGazeNotifyTime = -999f;
    private const float gazeHighlightHoldSeconds = 0.35f;
    private float recommendationFlashStartTime = -999f;
    private float recommendationFlashEndTime = -999f;
    private int recommendationRank = 0;

    public ControlState State { get; private set; }

    private void Awake()
    {
        CacheButtonVisual();
    }

    private void Start()
    {
        CacheButtonVisual();
        ResetControl();
    }

    private void OnValidate()
    {
        recommendationFlashCount = Mathf.Max(1, recommendationFlashCount);
        recommendationFlashInterval = Mathf.Max(0.02f, recommendationFlashInterval);
    }

    /// <summary>
    /// 由 TaskManager 在开始/结束实验时控制。
    /// </summary>
    public void SetExperimentActive(bool active)
    {
        isExperimentActive = active;
    }

    /// <summary>
    /// 重置控件状态。
    /// </summary>
    public void ResetControl()
    {
        value = 0f;
        isTriggered = false;
        triggerTime = 0f;
        State = ControlState.Idle;

        baseGrowthSpeed = Random.Range(0.5f, 2.5f);
        currentBonusSpeed = 0f;
        symbolState = Random.Range(0, 2);
        symbolChangeTimer = Random.Range(1.0f, 2.0f);

        // 不在这里重置 isExperimentActi ve，生命周期由 TaskManager 管理
        isBeingLookedAt = false;
        lastGazeNotifyTime = -999f;
        ClearRecommendationHighlight();

        UpdateUI(baseGrowthSpeed);
    }

    /// <summary>
    /// 由 EyeTrackingManager 命中时调用。
    /// </summary>
    public void NotifyGaze()
    {
        isBeingLookedAt = true;
        lastGazeNotifyTime = Time.time;
    }

    public void SetRecommendation(int rank, float confidence, float holdSeconds)
    {
        FlashRecommendationHighlight(rank);
    }

    public void ClearRecommendation()
    {
        ClearRecommendationHighlight();
    }

    public void FlashRecommendationHighlight(int rank = 1)
    {
        recommendationRank = Mathf.Max(1, rank);
        recommendationFlashStartTime = Time.time;
        recommendationFlashEndTime = recommendationFlashStartTime + GetRecommendationFlashDuration();
    }

    public void ClearRecommendationHighlight()
    {
        recommendationRank = 0;
        recommendationFlashStartTime = -999f;
        recommendationFlashEndTime = -999f;
    }

    private void Update()
    {
        if (!isExperimentActive)
            return;

        if (isBeingLookedAt)
        {
            currentBonusSpeed += accelerationRate * Time.deltaTime;
        }
        else
        {
            currentBonusSpeed -= decelerationRate * Time.deltaTime;
        }

        currentBonusSpeed = Mathf.Clamp(currentBonusSpeed, 0f, maxTotalSpeed - baseGrowthSpeed);
        float totalSpeed = baseGrowthSpeed + currentBonusSpeed;

        if (State != ControlState.Triggered)
        {
            value += Time.deltaTime * totalSpeed;
        }

        if (isBeingLookedAt)
        {
            symbolState = 1;
            symbolChangeTimer = gazeSymbolHoldSeconds;
        }
        else
        {
            symbolChangeTimer -= Time.deltaTime;
            if (symbolChangeTimer <= 0f)
            {
                symbolState = symbolState == 0 ? 1 : 0;
                symbolChangeTimer = Random.Range(2.0f, 4.0f);
            }
        }

        CheckState();
        UpdateTriggerTimer();
        UpdateUI(totalSpeed);

        // 等待下一帧再次由 EyeTrackingManager 赋值
        isBeingLookedAt = false;
    }

    private void CheckState()
    {
        if (symbolState == 1 && value >= triggerThreshold)
        {
            if (!isTriggered)
            {
                isTriggered = true;
                triggerTime = 20f;
            }

            State = ControlState.Triggered;
        }
        else
        {
            State = ControlState.Idle;
        }
    }

    private void UpdateTriggerTimer()
    {
        if (!isTriggered)
            return;

        triggerTime -= Time.deltaTime;
        if (triggerTime <= 0f)
        {
            isTriggered = false;
        }
    }

    private void UpdateUI(float currentSpeed)
    {
        if (valueText != null)
        {
            valueText.text = $"{(int)value} ({currentSpeed:F1}x)";
        }

        if (symbolText != null)
        {
            symbolText.text = symbolState.ToString();
        }

        if (buttonImage != null)
        {
            bool showGazeHighlight = (Time.time - lastGazeNotifyTime) <= gazeHighlightHoldSeconds;
            bool showRecommendation = IsRecommendationFlashVisible();
            Color targetColor = (isTriggered || State == ControlState.Triggered)
                ? triggeredColor
                : (showRecommendation ? recommendationFlashColor : (showGazeHighlight ? gazeColor : normalColor));
            buttonImage.color = Color.Lerp(buttonImage.color, targetColor, Time.deltaTime * 12f);
        }

        if (triggerStateText != null)
        {
            bool showRecommendation = IsRecommendationFlashActive();
            if (isTriggered || State == ControlState.Triggered)
            {
                triggerStateText.text = "Ready";
            }
            else if (showRecommendation)
            {
                triggerStateText.text = $"R{recommendationRank}";
            }
            else
            {
                triggerStateText.text = "Standby";
            }
        }
    }

    private void CacheButtonVisual()
    {
        Button ownButton = GetComponent<Button>();
        if (ownButton != null)
        {
            button = ownButton;
        }

        buttonImage = GetComponent<Image>();
        if (buttonImage == null && button != null)
        {
            buttonImage = button.GetComponent<Image>();
        }
    }

    private float GetRecommendationFlashDuration()
    {
        return Mathf.Max(1, recommendationFlashCount) * Mathf.Max(0.02f, recommendationFlashInterval) * 2f;
    }

    private bool IsRecommendationFlashActive()
    {
        return recommendationRank > 0 && Time.time <= recommendationFlashEndTime;
    }

    private bool IsRecommendationFlashVisible()
    {
        if (!IsRecommendationFlashActive())
        {
            return false;
        }

        float elapsed = Mathf.Max(0f, Time.time - recommendationFlashStartTime);
        int phase = Mathf.FloorToInt(elapsed / Mathf.Max(0.02f, recommendationFlashInterval));
        return phase % 2 == 0;
    }

    public bool IsTriggered() => State == ControlState.Triggered;

    public bool IsReadyForConfirm()
    {
        return value >= triggerThreshold && symbolState == 1;
    }
}
