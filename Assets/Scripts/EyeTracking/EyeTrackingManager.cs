using UnityEngine;
using UnityEngine.UI;
using Unity.XR.PXR;
using System;
using System.Collections.Generic;
using System.Reflection;

public class EyeTrackingManager : MonoBehaviour
{
    [Header("UI Display")]
    public Text eyeTrackingText;

    [Header("Debug (Optional)")]
    public Transform point_cube;
    public GazeDataLogger dataLogger;

    [Header("UI Plane Mapping")]
    public RectTransform uiPlaneRect;
    public List<ControlItem> controlItems;
    public bool fallbackToPhysicsRaycast = true;
    [Tooltip("Rotate UI 2D coordinates by 180 degrees (x=1-x, y=1-y)")]
    public bool rotateScreen180 = false;
    [Tooltip("Normalized correction applied after projection and rotation. Positive X moves gaze right, positive Y moves gaze up.")]
    public Vector2 screenOffset01 = Vector2.zero;

    [Header("Direction Mapping")]
    [Tooltip("Invert gaze direction X axis")]
    public bool invertGazeDirectionX = true;
    [Tooltip("Invert gaze direction Y axis")]
    public bool invertGazeDirectionY = true;

    [Header("Sampling")]
    public float samplingInterval = 0.02f;

    private Transform cameraTransform;
    private bool support = false;
    private bool eyeTrackingStarted = false;

    private EyeTrackingMode[] eyeTrackingModes = new EyeTrackingMode[8];
    private EyeTrackingData eyeTrackingData = new EyeTrackingData();
    private EyePupilInfo eyePupilInfo = new EyePupilInfo();

    private float timer = 0f;
    private int frameIndex = 0;
    private const float maxRayDistance = 100f;
    private const int fusedEyeIndex = 2;
    private int lastTrackingCode = int.MinValue;
    private int lastPupilCode = int.MinValue;
    private int lastOpennessCode = int.MinValue;
    private bool hasLoggedPupilDilationFallback = false;
    private bool hasLoggedPupilFieldSnapshot = false;
    private bool hasLoggedInspectorWarnings = false;
    private bool hasRequestedEyeTrackingPermission = false;
    private const string kEyeTrackingPermission = "com.picovr.permission.EYE_TRACKING";

    public int LastAoiId { get; private set; } = -1;
    public Vector2 LastRawScreen01 { get; private set; } = new Vector2(-1f, -1f);
    public Vector2 LastCorrectedScreen01 { get; private set; } = new Vector2(-1f, -1f);

    void Start()
    {
        ValidateInspectorSetup(logWarnings: true);

        Camera mainCam = Camera.main;
        if (mainCam == null)
        {
            SetStatus("主摄像机缺失，无法获取眼动数据");
            Debug.LogError("[EyeTracking] Main Camera is missing.");
            return;
        }

        cameraTransform = mainCam.transform;
        ControlIdUtility.NormalizeLegacyOneBasedIds(controlItems, "EyeTrackingManager");

        int supportModesCount = 0;
        int supportResult = PXR_MotionTracking.GetEyeTrackingSupported(ref support, ref supportModesCount, ref eyeTrackingModes);
        if (supportResult != 0)
        {
            SetStatus($"眼动支持检查失败：{FormatTrackingCode(supportResult)}");
            Debug.LogError($"[EyeTracking] GetEyeTrackingSupported failed: {FormatTrackingCode(supportResult)}");
            return;
        }

        if (!support)
        {
            SetStatus("设备不支持眼动追踪");
            return;
        }

        EnsureEyeTrackingPermissionOnAndroid();

        EyeTrackingStartInfo eyeTrackingStartInfo = new EyeTrackingStartInfo
        {
            needCalibration = 0,
            mode = EyeTrackingMode.PXR_ETM_BOTH
        };

        int startResult = PXR_MotionTracking.StartEyeTracking(ref eyeTrackingStartInfo);
        if (startResult != 0)
        {
            SetStatus($"眼动追踪启动失败：{FormatTrackingCode(startResult)}");
            Debug.LogError($"[EyeTracking] StartEyeTracking failed: {FormatTrackingCode(startResult)}");
            return;
        }

        if (rotateScreen180 && (invertGazeDirectionX || invertGazeDirectionY))
        {
            Debug.LogWarning("[EyeTracking] rotateScreen180 and invertGazeDirection are both enabled. Auto disabling rotateScreen180.");
            rotateScreen180 = false;
        }

        eyeTrackingStarted = true;
        SetStatus("眼动追踪已启动");
    }

    private void OnValidate()
    {
        samplingInterval = Mathf.Max(0.001f, samplingInterval);
        screenOffset01 = new Vector2(
            Mathf.Clamp(screenOffset01.x, -1f, 1f),
            Mathf.Clamp(screenOffset01.y, -1f, 1f));
        ValidateInspectorSetup(logWarnings: false);
    }

    void Update()
    {
        if (!support || !eyeTrackingStarted) return;

        timer += Time.deltaTime;
        if (timer >= samplingInterval)
        {
            ProcessEyeTracking();
            timer = 0f;
        }
    }

    private void ProcessEyeTracking()
    {
        if (dataLogger != null)
        {
            dataLogger.ResetFrameStatus();
        }

        bool tracking = false;
        EyeTrackingState eyeTrackingState = new EyeTrackingState();
        int trackingStateResult = PXR_MotionTracking.GetEyeTrackingState(ref tracking, ref eyeTrackingState);

        long timestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

        if (trackingStateResult != 0 || !tracking)
        {
            int stateCode = (int)eyeTrackingState.code;
            int reportCode = trackingStateResult != 0 ? trackingStateResult : stateCode;
            SetStatus("[-,-] 无有效注视");

            if (reportCode != lastTrackingCode)
            {
                Debug.LogWarning($"[EyeTracking] Tracking invalid. API={FormatTrackingCode(trackingStateResult)}, state={FormatTrackingCode(stateCode)}, mode={eyeTrackingState.currentTrackingMode}");
                lastTrackingCode = reportCode;
            }

            RecordUnavailableFrame(timestampMs);
            return;
        }

        lastTrackingCode = 0;

        EyeTrackingDataGetInfo info = new EyeTrackingDataGetInfo
        {
            displayTime = 0,
            flags = EyeTrackingDataGetFlags.PXR_EYE_DEFAULT
                  | EyeTrackingDataGetFlags.PXR_EYE_POSITION
                  | EyeTrackingDataGetFlags.PXR_EYE_ORIENTATION
        };

        int getDataResult = PXR_MotionTracking.GetEyeTrackingData(ref info, ref eyeTrackingData);
        if (getDataResult != 0)
        {
            SetStatus("[-,-] 眼动数据获取失败");
            Debug.LogWarning($"[EyeTracking] GetEyeTrackingData failed: {FormatTrackingCode(getDataResult)}");
            RecordUnavailableFrame(timestampMs);
            return;
        }

        if (eyeTrackingData.eyeDatas == null || eyeTrackingData.eyeDatas.Length <= fusedEyeIndex)
        {
            SetStatus("[-,-] 眼动数据格式无效");
            Debug.LogWarning("[EyeTracking] eyeDatas does not contain fused-eye data.");
            RecordUnavailableFrame(timestampMs);
            return;
        }

        float pupilDiameter;
        float leftPupil;
        float rightPupil;
        int pupilCode;
        int pupilValid;
        string pupilStatus;
        TryGetPupilDiameter(out pupilDiameter, out leftPupil, out rightPupil, out pupilCode, out pupilValid, out pupilStatus);

        float leftOpenness;
        float rightOpenness;
        int opennessCode;
        TryGetEyeOpenness(out leftOpenness, out rightOpenness, out opennessCode);

        var pose = eyeTrackingData.eyeDatas[fusedEyeIndex].pose;
        Vector3 localOrigin = new Vector3(pose.position.x, pose.position.y, pose.position.z);
        Quaternion localRotation = new Quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        );

        Vector3 worldOrigin = cameraTransform.TransformPoint(localOrigin);
        Vector3 localDirection = localRotation * Vector3.forward;
        if (invertGazeDirectionX) localDirection.x = -localDirection.x;
        if (invertGazeDirectionY) localDirection.y = -localDirection.y;
        Vector3 worldDirection = cameraTransform.TransformDirection(localDirection.normalized);
        Vector3 gazePoint = worldOrigin + worldDirection * 10f;

        Vector3 uiHitWorld;
        Vector2 rawScreen01;
        Vector2 screen01;
        bool hitUIPlane = ProjectToScreen01(worldOrigin, worldDirection, out uiHitWorld, out rawScreen01, out screen01);
        LastRawScreen01 = hitUIPlane ? rawScreen01 : new Vector2(-1f, -1f);
        LastCorrectedScreen01 = hitUIPlane ? screen01 : new Vector2(-1f, -1f);

        if (hitUIPlane)
        {
            gazePoint = uiHitWorld;
        }

        if (point_cube != null)
        {
            point_cube.position = gazePoint;
        }

        ControlItem hitItem = null;
        int aoiId = -1;
        string aoiName = "NONE";

        if (hitUIPlane)
        {
            hitItem = FindControlByRect(uiHitWorld);
        }

        if (hitItem == null && fallbackToPhysicsRaycast)
        {
            hitItem = FindControlByPhysics(worldOrigin, worldDirection);
        }

        if (hitItem != null)
        {
            aoiId = hitItem.controlID;
            aoiName = hitItem.name;
            LastAoiId = aoiId;

            hitItem.NotifyGaze();

            if (dataLogger != null)
            {
                dataLogger.RecordGaze(hitItem.name, samplingInterval, pupilDiameter);
            }

            SetStatus($"raw[{rawScreen01.x:F2},{rawScreen01.y:F2}] corrected[{screen01.x:F2},{screen01.y:F2}] 控件{aoiId}");
        }
        else
        {
            LastAoiId = -1;
            SetStatus(hitUIPlane
                ? $"raw[{rawScreen01.x:F2},{rawScreen01.y:F2}] corrected[{screen01.x:F2},{screen01.y:F2}] 无控件"
                : "[-,-] 无控件");
        }

        if (dataLogger != null)
        {
            dataLogger.RecordFrame(new GazeDataLogger.FrameRecord
            {
                timestampMs = timestampMs,
                frameIdx = frameIndex++,
                trackingState = 1,

                gazeOriginX = worldOrigin.x,
                gazeOriginY = worldOrigin.y,
                gazeOriginZ = worldOrigin.z,

                gazeDirX = worldDirection.x,
                gazeDirY = worldDirection.y,
                gazeDirZ = worldDirection.z,

                gazePointX = gazePoint.x,
                gazePointY = gazePoint.y,
                gazePointZ = gazePoint.z,

                screenXNorm = hitUIPlane ? screen01.x : -1f,
                screenYNorm = hitUIPlane ? screen01.y : -1f,
                rawScreenXNorm = hitUIPlane ? rawScreen01.x : -1f,
                rawScreenYNorm = hitUIPlane ? rawScreen01.y : -1f,

                hitUIPlane = hitUIPlane ? 1 : 0,
                aoiId = aoiId,
                aoiName = aoiName,

                pupilDiameter = pupilDiameter,
                leftPupilDiameter = leftPupil,
                rightPupilDiameter = rightPupil,
                pupilValid = pupilValid,
                pupilStatus = pupilStatus,
                leftOpenness = leftOpenness,
                rightOpenness = rightOpenness
            });
        }
    }

    private void RecordUnavailableFrame(long timestampMs)
    {
        if (dataLogger == null) return;

        dataLogger.RecordFrame(new GazeDataLogger.FrameRecord
        {
            timestampMs = timestampMs,
            frameIdx = frameIndex++,
            trackingState = 0,
            gazeOriginX = 0f,
            gazeOriginY = 0f,
            gazeOriginZ = 0f,
            gazeDirX = 0f,
            gazeDirY = 0f,
            gazeDirZ = 0f,
            gazePointX = 0f,
            gazePointY = 0f,
            gazePointZ = 0f,
            screenXNorm = -1f,
            screenYNorm = -1f,
            rawScreenXNorm = -1f,
            rawScreenYNorm = -1f,
            hitUIPlane = 0,
            aoiId = -1,
            aoiName = "NONE",
            pupilDiameter = float.NaN,
            leftPupilDiameter = float.NaN,
            rightPupilDiameter = float.NaN,
            pupilValid = 0,
            pupilStatus = "tracking_unavailable",
            leftOpenness = float.NaN,
            rightOpenness = float.NaN
        });
    }

    private bool ProjectToScreen01(Vector3 origin, Vector3 direction, out Vector3 hitWorld, out Vector2 rawScreen01, out Vector2 screen01)
    {
        hitWorld = Vector3.zero;
        rawScreen01 = new Vector2(-1f, -1f);
        screen01 = new Vector2(-1f, -1f);

        if (uiPlaneRect == null) return false;

        Plane plane = new Plane(uiPlaneRect.forward, uiPlaneRect.position);
        Ray ray = new Ray(origin, direction.normalized);

        float enter;
        if (!plane.Raycast(ray, out enter)) return false;

        hitWorld = ray.GetPoint(enter);

        Vector3 local3 = uiPlaneRect.InverseTransformPoint(hitWorld);
        Rect rect = uiPlaneRect.rect;

        float x01 = Mathf.InverseLerp(rect.xMin, rect.xMax, local3.x);
        float y01 = Mathf.InverseLerp(rect.yMin, rect.yMax, local3.y);

        if (x01 < 0f || x01 > 1f || y01 < 0f || y01 > 1f)
        {
            return false;
        }

        rawScreen01 = new Vector2(x01, y01);

        if (rotateScreen180)
        {
            x01 = 1f - x01;
            y01 = 1f - y01;
        }

        rawScreen01 = new Vector2(x01, y01);
        Vector2 corrected01 = new Vector2(
            Mathf.Clamp01(rawScreen01.x + screenOffset01.x),
            Mathf.Clamp01(rawScreen01.y + screenOffset01.y));
        screen01 = corrected01;

        float correctedLocalX = Mathf.Lerp(rect.xMin, rect.xMax, corrected01.x);
        float correctedLocalY = Mathf.Lerp(rect.yMin, rect.yMax, corrected01.y);
        hitWorld = uiPlaneRect.TransformPoint(new Vector3(correctedLocalX, correctedLocalY, local3.z));
        return true;
    }

    private ControlItem FindControlByRect(Vector3 hitWorld)
    {
        if (controlItems == null) return null;

        foreach (var item in controlItems)
        {
            if (item == null) continue;

            RectTransform rt = item.GetComponent<RectTransform>();
            if (rt == null) continue;

            Vector3 local = rt.InverseTransformPoint(hitWorld);
            Vector2 local2 = new Vector2(local.x, local.y);

            if (rt.rect.Contains(local2))
            {
                return item;
            }
        }

        return null;
    }

    private void ValidateInspectorSetup(bool logWarnings)
    {
        if (!logWarnings || hasLoggedInspectorWarnings)
        {
            return;
        }

        hasLoggedInspectorWarnings = true;

        if (uiPlaneRect == null)
        {
            Debug.LogWarning("[EyeTracking] uiPlaneRect is not assigned; gaze-to-UI projection will fail.");
        }

        if (controlItems == null || controlItems.Count == 0)
        {
            Debug.LogWarning("[EyeTracking] controlItems is empty; gaze hits cannot map to controls.");
        }
        else
        {
            for (int i = 0; i < controlItems.Count; i++)
            {
                if (controlItems[i] == null)
                {
                    Debug.LogWarning($"[EyeTracking] controlItems[{i}] is null.");
                }
            }
        }

        if (invertGazeDirectionX && invertGazeDirectionY)
        {
            Debug.LogWarning("[EyeTracking] Both invertGazeDirectionX and invertGazeDirectionY are enabled. Verify this matches the headset coordinate system before tuning screenOffset01.");
        }

        if (screenOffset01 != Vector2.zero)
        {
            Debug.Log($"[EyeTracking] screenOffset01 correction is active: {screenOffset01}");
        }
    }

    private ControlItem FindControlByPhysics(Vector3 origin, Vector3 direction)
    {
        RaycastHit hit;
        if (Physics.Raycast(origin, direction, out hit, maxRayDistance))
        {
            ControlItem item = hit.collider.GetComponent<ControlItem>();
            if (item == null)
            {
                item = hit.collider.GetComponentInParent<ControlItem>();
            }
            return item;
        }
        return null;
    }

    private bool TryGetPupilDiameter(out float pupilDiameter, out float leftPupil, out float rightPupil, out int pupilCode, out int pupilValid, out string pupilStatus)
    {
        pupilDiameter = float.NaN;
        leftPupil = float.NaN;
        rightPupil = float.NaN;
        pupilValid = 0;
        pupilStatus = "api_error";

        pupilCode = PXR_MotionTracking.GetEyePupilInfo(ref eyePupilInfo);
        if (pupilCode != 0)
        {
            if (pupilCode != lastPupilCode)
            {
                Debug.LogWarning($"[EyeTracking] GetEyePupilInfo failed: {FormatTrackingCode(pupilCode)}");
                lastPupilCode = pupilCode;
            }
            return false;
        }

        lastPupilCode = 0;
        leftPupil = ResolveEyePupilRawValue(eyePupilInfo, true);
        rightPupil = ResolveEyePupilRawValue(eyePupilInfo, false);
        pupilDiameter = ResolvePupilRawValue(leftPupil, rightPupil);

        if (IsValidPupilValue(pupilDiameter))
        {
            pupilValid = 1;
            pupilStatus = "valid";
            return true;
        }

        float fallbackDiameter = ResolvePupilDiameterFromDilation(eyePupilInfo);
        if (IsValidPupilValue(fallbackDiameter))
        {
            pupilDiameter = fallbackDiameter;
            pupilValid = 1;
            pupilStatus = "dilation_fallback";
            if (!hasLoggedPupilDilationFallback)
            {
                Debug.LogWarning("[EyeTracking] EyePupilInfo diameter/radius fields are invalid, fallback to pupil dilation-style fields.");
                hasLoggedPupilDilationFallback = true;
            }
            return true;
        }

        if (IsKnownZeroPupilValue(leftPupil) || IsKnownZeroPupilValue(rightPupil))
        {
            pupilDiameter = ResolvePupilZeroAwareValue(leftPupil, rightPupil);
            pupilValid = 0;
            pupilStatus = "api_zero";
            LogPupilFieldSnapshotOnce(eyePupilInfo);
            return false;
        }

        pupilStatus = "unresolved";
        LogPupilFieldSnapshotOnce(eyePupilInfo);
        return false;
    }

    private bool TryGetEyeOpenness(out float leftOpenness, out float rightOpenness, out int opennessCode)
    {
        leftOpenness = float.NaN;
        rightOpenness = float.NaN;

        float rawLeft = 0f;
        float rawRight = 0f;
        opennessCode = PXR_MotionTracking.GetEyeOpenness(ref rawLeft, ref rawRight);
        if (opennessCode != 0)
        {
            if (opennessCode != lastOpennessCode)
            {
                Debug.LogWarning($"[EyeTracking] GetEyeOpenness failed: {FormatTrackingCode(opennessCode)}");
                lastOpennessCode = opennessCode;
            }
            return false;
        }

        lastOpennessCode = 0;

        if (IsValidOpennessValue(rawLeft))
        {
            leftOpenness = rawLeft;
        }

        if (IsValidOpennessValue(rawRight))
        {
            rightOpenness = rawRight;
        }

        return true;
    }

    private float ResolvePupilDiameterFromInfo(EyePupilInfo info)
    {
        float left = ResolveEyePupilRawValue(info, true);
        float right = ResolveEyePupilRawValue(info, false);
        return ResolvePupilValue(left, right);
    }

    private float ResolvePupilDiameterFromDilation(EyePupilInfo info)
    {
        float left = GetOptionalPupilFieldValue(info, true, kLeftDilationCandidates, false);
        float right = GetOptionalPupilFieldValue(info, false, kRightDilationCandidates, false);
        return ResolvePupilValue(left, right);
    }

    private float ResolveEyePupilValue(EyePupilInfo info, bool isLeftEye)
    {
        float value = GetOptionalPupilFieldValue(
            info,
            isLeftEye,
            isLeftEye ? kLeftDiameterCandidates : kRightDiameterCandidates,
            false);
        if (!float.IsNaN(value))
        {
            return value;
        }

        float radius = GetOptionalPupilFieldValue(
            info,
            isLeftEye,
            isLeftEye ? kLeftRadiusCandidates : kRightRadiusCandidates,
            false);
        if (!float.IsNaN(radius))
        {
            return radius * 2f;
        }

        return float.NaN;
    }

    private float ResolveEyePupilRawValue(EyePupilInfo info, bool isLeftEye)
    {
        float value = GetOptionalPupilFieldValue(
            info,
            isLeftEye,
            isLeftEye ? kLeftDiameterCandidates : kRightDiameterCandidates,
            true);
        if (!float.IsNaN(value))
        {
            return value;
        }

        float radius = GetOptionalPupilFieldValue(
            info,
            isLeftEye,
            isLeftEye ? kLeftRadiusCandidates : kRightRadiusCandidates,
            true);
        if (!float.IsNaN(radius))
        {
            return radius * 2f;
        }

        return float.NaN;
    }

    private float GetOptionalPupilFieldValue(EyePupilInfo info, bool isLeftEye, string[] candidateNames, bool allowZero)
    {
        Type type = typeof(EyePupilInfo);

        if (candidateNames != null)
        {
            for (int i = 0; i < candidateNames.Length; i++)
            {
                float candidateValue = TryReadFloatMember(type, info, candidateNames[i]);
                if (IsUsablePupilValue(candidateValue, allowZero))
                {
                    return candidateValue;
                }
            }
        }

        return SearchPupilMembersByKeywords(type, info, isLeftEye, allowZero);
    }

    private float SearchPupilMembersByKeywords(Type type, EyePupilInfo info, bool isLeftEye, bool allowZero)
    {
        string eyeToken = isLeftEye ? "left" : "right";
        float bestDiameter = float.NaN;
        float bestRadius = float.NaN;
        float bestDilation = float.NaN;

        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        for (int i = 0; i < fields.Length; i++)
        {
            FieldInfo field = fields[i];
            if (field.FieldType != typeof(float)) continue;

            string name = field.Name.ToLowerInvariant();
            if (!name.Contains(eyeToken) || !name.Contains("pupil")) continue;

            object boxed = info;
            object raw = field.GetValue(boxed);
            if (!(raw is float value) || !IsUsablePupilValue(value, allowZero)) continue;

            if (name.Contains("diameter"))
            {
                bestDiameter = value;
                break;
            }

            if (float.IsNaN(bestRadius) && name.Contains("radius"))
            {
                bestRadius = value;
                continue;
            }

            if (float.IsNaN(bestDilation) && (name.Contains("dilation") || name.Contains("dilat")))
            {
                bestDilation = value;
            }
        }

        if (!float.IsNaN(bestDiameter)) return bestDiameter;
        if (!float.IsNaN(bestRadius)) return bestRadius * 2f;
        return bestDilation;
    }

    private float TryReadFloatMember(Type type, EyePupilInfo info, string memberName)
    {
        const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

        FieldInfo field = type.GetField(memberName, flags);
        if (field != null && field.FieldType == typeof(float))
        {
            object boxed = info;
            object raw = field.GetValue(boxed);
            return raw is float floatValue ? floatValue : float.NaN;
        }

        PropertyInfo property = type.GetProperty(memberName, flags);
        if (property != null && property.PropertyType == typeof(float) && property.CanRead)
        {
            object boxed = info;
            object raw = property.GetValue(boxed, null);
            return raw is float floatValue ? floatValue : float.NaN;
        }

        return float.NaN;
    }

    private void LogPupilFieldSnapshotOnce(EyePupilInfo info)
    {
        if (hasLoggedPupilFieldSnapshot)
        {
            return;
        }

        hasLoggedPupilFieldSnapshot = true;

        Type type = typeof(EyePupilInfo);
        FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
        List<string> entries = new List<string>();

        for (int i = 0; i < fields.Length; i++)
        {
            FieldInfo field = fields[i];
            if (field.FieldType != typeof(float)) continue;

            string nameLower = field.Name.ToLowerInvariant();
            if (!nameLower.Contains("pupil")) continue;

            object boxed = info;
            object raw = field.GetValue(boxed);
            if (raw is float value)
            {
                entries.Add($"{field.Name}={value:F6}");
            }
        }

        if (entries.Count == 0)
        {
            Debug.LogWarning("[EyeTracking] EyePupilInfo returned no readable float pupil fields. pupil_diameter will stay empty.");
            return;
        }

        Debug.LogWarning("[EyeTracking] Unable to resolve pupil diameter from EyePupilInfo. Available pupil fields: " + string.Join(", ", entries));
    }

    private float ResolvePupilValue(float left, float right)
    {
        bool leftValid = IsValidPupilValue(left);
        bool rightValid = IsValidPupilValue(right);

        if (leftValid && rightValid) return (left + right) * 0.5f;
        if (leftValid) return left;
        if (rightValid) return right;
        return float.NaN;
    }

    private float ResolvePupilRawValue(float left, float right)
    {
        bool leftKnown = IsUsablePupilValue(left, true);
        bool rightKnown = IsUsablePupilValue(right, true);

        if (leftKnown && rightKnown) return (left + right) * 0.5f;
        if (leftKnown) return left;
        if (rightKnown) return right;
        return float.NaN;
    }

    private float ResolvePupilZeroAwareValue(float left, float right)
    {
        bool leftKnown = IsKnownZeroPupilValue(left);
        bool rightKnown = IsKnownZeroPupilValue(right);

        if (leftKnown && rightKnown) return 0f;
        if (leftKnown) return 0f;
        if (rightKnown) return 0f;
        return float.NaN;
    }

    private bool IsValidPupilValue(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value) && value > 0f;
    }

    private bool IsUsablePupilValue(float value, bool allowZero)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
        {
            return false;
        }

        return allowZero ? value >= 0f : value > 0f;
    }

    private bool IsKnownZeroPupilValue(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value) && Mathf.Approximately(value, 0f);
    }

    private static readonly string[] kLeftDiameterCandidates =
    {
        "leftEyePupilDiameter",
        "LeftEyePupilDiameter",
        "leftPupilDiameter",
        "LeftPupilDiameter",
        "leftEyeDiameter",
        "LeftEyeDiameter"
    };

    private static readonly string[] kRightDiameterCandidates =
    {
        "rightEyePupilDiameter",
        "RightEyePupilDiameter",
        "rightPupilDiameter",
        "RightPupilDiameter",
        "rightEyeDiameter",
        "RightEyeDiameter"
    };

    private static readonly string[] kLeftRadiusCandidates =
    {
        "leftEyePupilRadius",
        "LeftEyePupilRadius",
        "leftPupilRadius",
        "LeftPupilRadius",
        "leftEyeRadius",
        "LeftEyeRadius"
    };

    private static readonly string[] kRightRadiusCandidates =
    {
        "rightEyePupilRadius",
        "RightEyePupilRadius",
        "rightPupilRadius",
        "RightPupilRadius",
        "rightEyeRadius",
        "RightEyeRadius"
    };

    private static readonly string[] kLeftDilationCandidates =
    {
        "leftEyePupilDilation",
        "LeftEyePupilDilation",
        "leftPupilDilation",
        "LeftPupilDilation",
        "leftEyePupilDilate",
        "LeftEyePupilDilate"
    };

    private static readonly string[] kRightDilationCandidates =
    {
        "rightEyePupilDilation",
        "RightEyePupilDilation",
        "rightPupilDilation",
        "RightPupilDilation",
        "rightEyePupilDilate",
        "RightEyePupilDilate"
    };

    private bool IsValidOpennessValue(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value) && value >= 0f && value <= 1f;
    }

    private string FormatTrackingCode(int code)
    {
        if (Enum.IsDefined(typeof(TrackingStateCode), code))
        {
            return $"{(TrackingStateCode)code} ({code})";
        }

        return $"UNKNOWN ({code})";
    }

    private void EnsureEyeTrackingPermissionOnAndroid()
    {
        if (hasRequestedEyeTrackingPermission || Application.platform != RuntimePlatform.Android)
        {
            return;
        }

        hasRequestedEyeTrackingPermission = true;

        try
        {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (AndroidJavaClass packageManager = new AndroidJavaClass("android.content.pm.PackageManager"))
            {
                int granted = packageManager.GetStatic<int>("PERMISSION_GRANTED");
                int checkResult = activity.Call<int>("checkSelfPermission", kEyeTrackingPermission);
                if (checkResult == granted)
                {
                    Debug.Log("[EyeTracking] Eye tracking permission already granted.");
                    return;
                }

                using (AndroidJavaClass activityCompat = new AndroidJavaClass("androidx.core.app.ActivityCompat"))
                {
                    activityCompat.CallStatic(
                        "requestPermissions",
                        activity,
                        new string[] { kEyeTrackingPermission },
                        1001);
                }

                Debug.LogWarning("[EyeTracking] Requested runtime permission for com.picovr.permission.EYE_TRACKING.");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("[EyeTracking] Failed to verify/request eye tracking permission: " + e.Message);
        }
    }

    private void SetStatus(string message)
    {
        if (eyeTrackingText != null)
        {
            eyeTrackingText.text = message;
        }
    }

    private void OnDisable()
    {
        if (support && eyeTrackingStarted)
        {
            EyeTrackingStopInfo eyeTrackingStopInfo = new EyeTrackingStopInfo();
            PXR_MotionTracking.StopEyeTracking(ref eyeTrackingStopInfo);
            eyeTrackingStarted = false;
        }
    }
}
