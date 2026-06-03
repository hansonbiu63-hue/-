using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

public class GazeDataLogger : MonoBehaviour
{
    public event Action<FrameRecord> OnFrameRecorded;
    public FrameRecord LastRecordedFrame { get; private set; }

    [Header("Output Folder (Windows runtime)")]
    public string folderPath = @"F:\TestData";
    [Tooltip("Android上额外镜像到 /storage/emulated/0/Android/media/<包名>/TestData，便于直接查看和adb拉取")]
    public bool enableAndroidMediaMirror = true;

    [Header("Session")]
    public string subjectId = "S001";
    public string sessionId = "Session_001";

    [Header("Write Behavior")]
    [Tooltip("攒多少帧后落盘。值越小越不容易丢数据。")]
    public int flushEveryNFrames = 50;

    [Serializable]
    public class FrameRecord
    {
        public long timestampMs;
        public int frameIdx;
        public int trackingState;

        public float gazeOriginX;
        public float gazeOriginY;
        public float gazeOriginZ;

        public float gazeDirX;
        public float gazeDirY;
        public float gazeDirZ;

        public float gazePointX;
        public float gazePointY;
        public float gazePointZ;

        public float screenXNorm;
        public float screenYNorm;
        public float rawScreenXNorm;
        public float rawScreenYNorm;

        public int hitUIPlane;
        public int aoiId;
        public string aoiName;

        public float pupilDiameter;
        public float leftPupilDiameter;
        public float rightPupilDiameter;
        public int pupilValid;
        public string pupilStatus;
        public string pupilSource;
        public int aoiValid;
        public float leftOpenness;
        public float rightOpenness;
    }

    private class RoundData
    {
        public int roundIndex;
        public string mode;
        public string phase = "idle";
        public string outcome = "";
        public string targetControlsPipe = "";
        public Dictionary<int, ControlStats> stats = new Dictionary<int, ControlStats>();
    }

    private class ControlStats
    {
        public int hitCount = 0;
        public float totalTime = 0f;
        public float pupilDiameterSum = 0f;
        public int pupilSamples = 0;
        public bool isCurrentlyLooking = false;
    }

    private const int kMinAoiId = 0;
    private const int kMaxAoiId = 9;

    private string frameFilePath;
    private string summaryFilePath;
    private string resolvedFolderPath;
    private string mirrorFolderPath;
    private string mirrorFrameFilePath;
    private string mirrorSummaryFilePath;
    private string androidDataMirrorFolderPath;
    private string androidDataMirrorFrameFilePath;
    private string androidDataMirrorSummaryFilePath;
    private string sessionStamp;

    private readonly StringBuilder frameBuffer = new StringBuilder();
    private int bufferedFrameCount = 0;

    private readonly List<RoundData> allRoundsData = new List<RoundData>();
    private readonly List<string> currentRoundFrameLines = new List<string>();
    private RoundData currentRoundLog;

    private const string kFrameHeader =
        "subject_id,session_id,round_index,mode,phase,target_controls," +
        "timestamp_ms,frame_idx,tracking_state," +
        "gaze_origin_x,gaze_origin_y,gaze_origin_z," +
        "gaze_dir_x,gaze_dir_y,gaze_dir_z," +
        "gaze_point_x,gaze_point_y,gaze_point_z," +
        "screen_x_norm,screen_y_norm,raw_screen_x_norm,raw_screen_y_norm,hit_ui_plane," +
        "aoi_id,aoi_valid,aoi_name,pupil_diameter,left_pupil_diameter,right_pupil_diameter," +
        "pupil_valid,pupil_status,pupil_source,left_openness,right_openness";

    void Awake()
    {
        sessionStamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        resolvedFolderPath = ResolveAndCreateFolder(folderPath);

        frameFilePath = Path.Combine(resolvedFolderPath, $"Frame_{subjectId}_{sessionId}_{sessionStamp}.csv");
        summaryFilePath = Path.Combine(resolvedFolderPath, $"Summary_{subjectId}_{sessionId}_{sessionStamp}.csv");
        SetupAndroidMirrorIfNeeded();
        SetupAndroidDataMirrorIfNeeded();

        WriteFrameHeader();
        Debug.Log($"[GazeDataLogger] Output folder: {resolvedFolderPath}");
        if (HasMirrorOutput)
        {
            Debug.Log($"[GazeDataLogger] Mirror folder: {mirrorFolderPath}");
        }
        if (HasAndroidDataMirrorOutput)
        {
            Debug.Log($"[GazeDataLogger] AndroidData mirror folder: {androidDataMirrorFolderPath}");
        }
    }

    private void WriteFrameHeader()
    {
        File.WriteAllText(frameFilePath, kFrameHeader + Environment.NewLine, Encoding.UTF8);
        if (HasMirrorOutput)
        {
            TryMirrorWrite(() =>
                File.WriteAllText(mirrorFrameFilePath, kFrameHeader + Environment.NewLine, Encoding.UTF8));
        }
        if (HasAndroidDataMirrorOutput)
        {
            TryMirrorWrite(() =>
                File.WriteAllText(androidDataMirrorFrameFilePath, kFrameHeader + Environment.NewLine, Encoding.UTF8));
        }
    }

    public void StartNewRound(int roundIndex, string mode)
    {
        currentRoundLog = new RoundData
        {
            roundIndex = roundIndex,
            mode = mode,
            phase = mode == "Dynamic" ? "free_view" : "task_ready",
            outcome = "",
            targetControlsPipe = ""
        };

        for (int i = kMinAoiId; i <= kMaxAoiId; i++)
        {
            currentRoundLog.stats.Add(i, new ControlStats());
        }

        currentRoundFrameLines.Clear();
    }

    public void SetRoundPhase(string phase)
    {
        if (currentRoundLog == null) return;
        currentRoundLog.phase = phase;
    }

    public void UpdateCurrentTaskTargets(List<int> targets, string phase = "task_active")
    {
        if (currentRoundLog == null) return;
        currentRoundLog.targetControlsPipe = (targets == null || targets.Count == 0)
            ? ""
            : string.Join("|", targets);
        currentRoundLog.phase = phase;
    }

    public void MarkRoundResult(string outcome)
    {
        if (currentRoundLog == null) return;
        currentRoundLog.outcome = outcome;
        currentRoundLog.phase = outcome;
    }

    public void EndCurrentRound()
    {
        if (currentRoundLog == null) return;

        RoundData finishedRound = currentRoundLog;
        allRoundsData.Add(finishedRound);

        FlushFrameBuffer();
        ExportRoundToCSV(finishedRound, currentRoundFrameLines);

        currentRoundFrameLines.Clear();
        currentRoundLog = null;
    }

    public void RecordGaze(string controlName, float deltaTime, float pupilDiameter)
    {
        int controlId;
        if (!TryExtractControlId(controlName, out controlId))
        {
            return;
        }

        RecordGaze(controlId, deltaTime, pupilDiameter);
    }

    public void RecordGaze(int controlId, float deltaTime, float pupilDiameter)
    {
        if (currentRoundLog == null || !IsValidAoiId(controlId) || !currentRoundLog.stats.ContainsKey(controlId)) return;

        ControlStats s = currentRoundLog.stats[controlId];
        s.totalTime += deltaTime;

        if (IsValidPupilTrainingValue(pupilDiameter))
        {
            s.pupilDiameterSum += pupilDiameter;
            s.pupilSamples++;
        }

        if (!s.isCurrentlyLooking)
        {
            s.hitCount++;
            s.isCurrentlyLooking = true;
        }
    }

    public void ResetFrameStatus()
    {
        if (currentRoundLog == null) return;
        foreach (var stat in currentRoundLog.stats.Values)
        {
            stat.isCurrentlyLooking = false;
        }
    }

    public void RecordFrame(FrameRecord rec)
    {
        int roundIndex = currentRoundLog != null ? currentRoundLog.roundIndex : -1;
        string mode = currentRoundLog != null ? currentRoundLog.mode : "";
        string phase = currentRoundLog != null ? currentRoundLog.phase : "";
        string targets = currentRoundLog != null ? currentRoundLog.targetControlsPipe : "";

        string pupilSource;
        float averagedPupilDiameter = ResolveAveragePupilDiameter(
            rec.leftPupilDiameter,
            rec.rightPupilDiameter,
            out pupilSource);

        rec.pupilDiameter = averagedPupilDiameter;
        rec.pupilSource = pupilSource;
        rec.aoiValid = IsValidAoiId(rec.aoiId) ? 1 : 0;

        string line =
            $"{subjectId},{sessionId},{roundIndex},{mode},{phase},{targets}," +
            $"{rec.timestampMs},{rec.frameIdx},{rec.trackingState}," +
            $"{F(rec.gazeOriginX)},{F(rec.gazeOriginY)},{F(rec.gazeOriginZ)}," +
            $"{F(rec.gazeDirX)},{F(rec.gazeDirY)},{F(rec.gazeDirZ)}," +
            $"{F(rec.gazePointX)},{F(rec.gazePointY)},{F(rec.gazePointZ)}," +
            $"{F(rec.screenXNorm)},{F(rec.screenYNorm)},{F(rec.rawScreenXNorm)},{F(rec.rawScreenYNorm)},{rec.hitUIPlane}," +
            $"{rec.aoiId},{rec.aoiValid},{Safe(rec.aoiName)}," +
            $"{F(rec.pupilDiameter)},{F(rec.leftPupilDiameter)},{F(rec.rightPupilDiameter)}," +
            $"{rec.pupilValid},{Safe(rec.pupilStatus)},{Safe(rec.pupilSource)}," +
            $"{F(rec.leftOpenness)},{F(rec.rightOpenness)}";

        frameBuffer.AppendLine(line);
        currentRoundFrameLines.Add(line);
        bufferedFrameCount++;

        int threshold = Mathf.Max(1, flushEveryNFrames);
        if (bufferedFrameCount >= threshold)
        {
            FlushFrameBuffer();
        }

        LastRecordedFrame = rec;
        OnFrameRecorded?.Invoke(rec);
    }

    public float ResolveAveragePupilDiameter(float left, float right, out string source)
    {
        bool leftValid = IsValidPupilTrainingValue(left);
        bool rightValid = IsValidPupilTrainingValue(right);

        if (leftValid && rightValid)
        {
            source = "both_mean";
            return (left + right) * 0.5f;
        }

        if (leftValid)
        {
            source = "left_only";
            return left;
        }

        if (rightValid)
        {
            source = "right_only";
            return right;
        }

        source = "none";
        return float.NaN;
    }

    private bool IsValidPupilTrainingValue(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value) && value > 0f;
    }

    private bool IsValidAoiId(int aoiId)
    {
        return aoiId >= kMinAoiId && aoiId <= kMaxAoiId;
    }

    private bool TryExtractControlId(string controlName, out int controlId)
    {
        controlId = -1;
        if (string.IsNullOrEmpty(controlName))
        {
            return false;
        }

        int digitStart = -1;
        for (int i = controlName.Length - 1; i >= 0; i--)
        {
            if (char.IsDigit(controlName[i]))
            {
                digitStart = i;
            }
            else if (digitStart >= 0)
            {
                break;
            }
        }

        if (digitStart < 0)
        {
            return false;
        }

        int digitEnd = digitStart;
        while (digitEnd + 1 < controlName.Length && char.IsDigit(controlName[digitEnd + 1]))
        {
            digitEnd++;
        }

        string digits = controlName.Substring(digitStart, digitEnd - digitStart + 1);
        int parsedId;
        if (!int.TryParse(digits, out parsedId))
        {
            return false;
        }

        if (!IsValidAoiId(parsedId))
        {
            return false;
        }

        controlId = parsedId;
        return true;
    }

    private string FormatControlLabel(int controlId)
    {
        return $"Control_{controlId}";
    }

    private string F(float v)
    {
        return float.IsNaN(v) ? "" : v.ToString("F6");
    }

    private string Safe(string s)
    {
        return string.IsNullOrEmpty(s) ? "NONE" : s.Replace(",", "_");
    }

    public void FlushFrameBuffer()
    {
        if (frameBuffer.Length == 0) return;

        string content = frameBuffer.ToString();
        File.AppendAllText(frameFilePath, content, Encoding.UTF8);
        if (HasMirrorOutput)
        {
            TryMirrorWrite(() => File.AppendAllText(mirrorFrameFilePath, content, Encoding.UTF8));
        }
        if (HasAndroidDataMirrorOutput)
        {
            TryMirrorWrite(() => File.AppendAllText(androidDataMirrorFrameFilePath, content, Encoding.UTF8));
        }

        frameBuffer.Clear();
        bufferedFrameCount = 0;
    }

    public void ExportToCSV()
    {
        FlushFrameBuffer();

        if (allRoundsData.Count == 0)
        {
            Debug.LogWarning("No summary data to export.");
            return;
        }

        StringBuilder csvContent = new StringBuilder();
        csvContent.AppendLine("Round,Mode,Outcome,Phase,Targets,ControlID,GazeCount,DwellTime(s),AvgPupilDiameter");
        string mirrorSummaryPath = null;
        string androidDataSummaryPath = null;

        foreach (var round in allRoundsData)
        {
            for (int controlId = kMinAoiId; controlId <= kMaxAoiId; controlId++)
            {
                ControlStats stats = round.stats[controlId];
                string avgPupil = stats.pupilSamples > 0
                    ? (stats.pupilDiameterSum / stats.pupilSamples).ToString("F3")
                    : "";

                csvContent.AppendLine(
                    $"{round.roundIndex},{round.mode},{round.outcome},{round.phase},{round.targetControlsPipe}," +
                    $"{FormatControlLabel(controlId)},{stats.hitCount},{stats.totalTime:F3},{avgPupil}"
                );
            }
        }

        File.WriteAllText(summaryFilePath, csvContent.ToString(), Encoding.UTF8);
        if (HasMirrorOutput)
        {
            mirrorSummaryPath = Path.Combine(
                mirrorFolderPath,
                $"Summary_{subjectId}_{sessionId}_{sessionStamp}.csv");
            TryMirrorWrite(() => File.WriteAllText(mirrorSummaryPath, csvContent.ToString(), Encoding.UTF8));
        }
        if (HasAndroidDataMirrorOutput)
        {
            androidDataSummaryPath = Path.Combine(
                androidDataMirrorFolderPath,
                $"Summary_{subjectId}_{sessionId}_{sessionStamp}.csv");
            TryMirrorWrite(() => File.WriteAllText(androidDataSummaryPath, csvContent.ToString(), Encoding.UTF8));
        }

        Debug.Log("<color=green>Frame and summary exported:</color>\n" + frameFilePath + "\n" + summaryFilePath);
        if (HasMirrorOutput)
        {
            Debug.Log("<color=green>Mirror summary exported:</color>\n" + mirrorSummaryPath);
        }
        if (HasAndroidDataMirrorOutput)
        {
            Debug.Log("<color=green>AndroidData summary exported:</color>\n" + androidDataSummaryPath);
        }
    }

    private string ResolveAndCreateFolder(string preferredFolder)
    {
        string fallbackFolder = Path.Combine(Application.persistentDataPath, "TestData");

        if (Application.platform != RuntimePlatform.WindowsEditor &&
            Application.platform != RuntimePlatform.WindowsPlayer)
        {
            if (Application.platform == RuntimePlatform.Android)
            {
                string androidDataFolder = $"/storage/emulated/0/Android/data/{Application.identifier}/files/TestData";
                try
                {
                    EnsureDirectory(androidDataFolder);
                    Debug.Log($"[GazeDataLogger] Android output folder: {androidDataFolder}");
                    return androidDataFolder;
                }
                catch (Exception androidPathErr)
                {
                    Debug.LogWarning($"[GazeDataLogger] Android data path unavailable, fallback to persistent path. {androidPathErr.Message}");
                }
            }

            EnsureDirectory(fallbackFolder);
            Debug.LogWarning($"[GazeDataLogger] Non-Windows runtime. Fallback to: {fallbackFolder}");
            return fallbackFolder;
        }

        try
        {
            EnsureDirectory(preferredFolder);
            return preferredFolder;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[GazeDataLogger] Failed to create folder: {preferredFolder}. Fallback to: {fallbackFolder}. Error: {e.Message}");
            EnsureDirectory(fallbackFolder);
            return fallbackFolder;
        }
    }

    private void EnsureDirectory(string path)
    {
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
    }

    private void SetupAndroidMirrorIfNeeded()
    {
        mirrorFolderPath = null;
        mirrorFrameFilePath = null;
        mirrorSummaryFilePath = null;

        if (!enableAndroidMediaMirror || Application.platform != RuntimePlatform.Android)
        {
            return;
        }

        string packageName = Application.identifier;
        string candidate = $"/storage/emulated/0/Android/media/{packageName}/TestData";

        try
        {
            EnsureDirectory(candidate);
            mirrorFolderPath = candidate;
            mirrorFrameFilePath = Path.Combine(mirrorFolderPath, $"Frame_{subjectId}_{sessionId}_{sessionStamp}.csv");
            mirrorSummaryFilePath = Path.Combine(mirrorFolderPath, $"Summary_{subjectId}_{sessionId}_{sessionStamp}.csv");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[GazeDataLogger] Android mirror disabled. {e.Message}");
            mirrorFolderPath = null;
            mirrorFrameFilePath = null;
            mirrorSummaryFilePath = null;
        }
    }

    private void SetupAndroidDataMirrorIfNeeded()
    {
        androidDataMirrorFolderPath = null;
        androidDataMirrorFrameFilePath = null;
        androidDataMirrorSummaryFilePath = null;

        if (Application.platform != RuntimePlatform.Android)
        {
            return;
        }

        string expectedFolder = $"/storage/emulated/0/Android/data/{Application.identifier}/files/TestData";
        if (IsSamePath(expectedFolder, resolvedFolderPath))
        {
            return;
        }

        try
        {
            EnsureDirectory(expectedFolder);
            androidDataMirrorFolderPath = expectedFolder;
            androidDataMirrorFrameFilePath = Path.Combine(androidDataMirrorFolderPath, $"Frame_{subjectId}_{sessionId}_{sessionStamp}.csv");
            androidDataMirrorSummaryFilePath = Path.Combine(androidDataMirrorFolderPath, $"Summary_{subjectId}_{sessionId}_{sessionStamp}.csv");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[GazeDataLogger] AndroidData mirror disabled. {e.Message}");
            androidDataMirrorFolderPath = null;
            androidDataMirrorFrameFilePath = null;
            androidDataMirrorSummaryFilePath = null;
        }
    }

    private bool HasMirrorOutput => !string.IsNullOrEmpty(mirrorFolderPath);
    private bool HasAndroidDataMirrorOutput => !string.IsNullOrEmpty(androidDataMirrorFolderPath);

    private void TryMirrorWrite(Action writeAction)
    {
        if (writeAction == null) return;

        try
        {
            writeAction();
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[GazeDataLogger] Mirror write failed: {e.Message}");
        }
    }

    private void ExportRoundToCSV(RoundData round, List<string> roundFrameLines)
    {
        if (round == null) return;

        string roundFramePath = Path.Combine(
            resolvedFolderPath,
            $"Frame_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");

        string roundSummaryPath = Path.Combine(
            resolvedFolderPath,
            $"Summary_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");

        StringBuilder roundFrameContent = new StringBuilder();
        roundFrameContent.AppendLine(kFrameHeader);

        if (roundFrameLines != null)
        {
            foreach (string line in roundFrameLines)
            {
                roundFrameContent.AppendLine(line);
            }
        }

        File.WriteAllText(roundFramePath, roundFrameContent.ToString(), Encoding.UTF8);
        if (HasMirrorOutput)
        {
            string mirrorRoundFramePath = Path.Combine(
                mirrorFolderPath,
                $"Frame_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");
            TryMirrorWrite(() => File.WriteAllText(mirrorRoundFramePath, roundFrameContent.ToString(), Encoding.UTF8));
        }
        if (HasAndroidDataMirrorOutput)
        {
            string androidDataRoundFramePath = Path.Combine(
                androidDataMirrorFolderPath,
                $"Frame_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");
            TryMirrorWrite(() => File.WriteAllText(androidDataRoundFramePath, roundFrameContent.ToString(), Encoding.UTF8));
        }

        StringBuilder roundSummaryContent = new StringBuilder();
        roundSummaryContent.AppendLine("Round,Mode,Outcome,Phase,Targets,ControlID,GazeCount,DwellTime(s),AvgPupilDiameter");

        for (int controlId = kMinAoiId; controlId <= kMaxAoiId; controlId++)
        {
            ControlStats stats = round.stats[controlId];
            string avgPupil = stats.pupilSamples > 0
                ? (stats.pupilDiameterSum / stats.pupilSamples).ToString("F3")
                : "";

            roundSummaryContent.AppendLine(
                $"{round.roundIndex},{round.mode},{round.outcome},{round.phase},{round.targetControlsPipe}," +
                $"{FormatControlLabel(controlId)},{stats.hitCount},{stats.totalTime:F3},{avgPupil}"
            );
        }

        File.WriteAllText(roundSummaryPath, roundSummaryContent.ToString(), Encoding.UTF8);
        if (HasMirrorOutput)
        {
            string mirrorRoundSummaryPath = Path.Combine(
                mirrorFolderPath,
                $"Summary_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");
            TryMirrorWrite(() => File.WriteAllText(mirrorRoundSummaryPath, roundSummaryContent.ToString(), Encoding.UTF8));
        }
        if (HasAndroidDataMirrorOutput)
        {
            string androidDataRoundSummaryPath = Path.Combine(
                androidDataMirrorFolderPath,
                $"Summary_{subjectId}_{sessionId}_{sessionStamp}_R{round.roundIndex:D2}.csv");
            TryMirrorWrite(() => File.WriteAllText(androidDataRoundSummaryPath, roundSummaryContent.ToString(), Encoding.UTF8));
        }

        Debug.Log("<color=green>Round data exported:</color>\n" + roundFramePath + "\n" + roundSummaryPath);
    }

    private void OnApplicationQuit()
    {
        FlushFrameBuffer();
    }

    private void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            FlushFrameBuffer();
        }
    }

    private void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus)
        {
            FlushFrameBuffer();
        }
    }

    private bool IsSamePath(string a, string b)
    {
        if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b)) return false;
        string na = a.Replace('\\', '/').TrimEnd('/').ToLowerInvariant();
        string nb = b.Replace('\\', '/').TrimEnd('/').ToLowerInvariant();
        return na == nb;
    }
}
