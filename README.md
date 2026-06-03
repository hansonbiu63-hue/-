
﻿# test2

## Eye Tracking Setup (PICO 4 Pro)

### Script Paths
- `Assets/Scripts/EyeTracking/EyeTrackingManager.cs`
- `Assets/Scripts/EyeTracking/GazeDataLogger.cs`
- `Assets/Scripts/EyeTracking/EyeTrackingCalibrationManager.cs`
- `Assets/Scripts/Task/TaskManager.cs`
- `Assets/Scripts/Control/ControlItem.cs`

### Scene Mounting
- `ExperimentRoot`
  - `EyeTrackingManager`
  - `GazeDataLogger`
  - `EyeTrackingCalibrationManager` (optional)
- `WorldSpaceCanvas`
  - `TaskPanel` (as UI root plane)
    - `Control_1 ... Control_10` (each has `ControlItem`)
- `ExperimentManager`
  - `TaskManager`

### Inspector Required Fields
- `EyeTrackingManager.uiPlaneRect = TaskPanel`
- `EyeTrackingManager.controlItems = Control_1...Control_10`
- `EyeTrackingManager.dataLogger = GazeDataLogger`
- `TaskManager.dataLogger = GazeDataLogger`
- `EyeTrackingCalibrationManager` is now a compatibility stub (no 9-point calibration logic)

### Debug Pipeline
1. Start static/dynamic task directly.
2. Verify AOI hit in `EyeTrackingManager` status text.
3. Check frame CSV fields:
   - `screen_x_norm`, `screen_y_norm`
   - `aoi_id`, `aoi_name`

## Transformer Offline Training

Runtime training is intentionally not used in Unity. The Unity scene loads a finetuned ONNX model through Sentis and validates it against `Assets/Resources/eye_transformer_stats.json`.

### Stage 1: GazeBaseVR pretraining

```powershell
python Tools\train_eye_transformer_modified.py `
  --stage pretrain `
  --gazebasevr_dir D:\Datasets\GazeBaseVR `
  --out_dir F:\TestData\OUT_DIR `
  --target_hz 50 `
  --window_frames 150 `
  --step_frames 15
```

This stage reads raw GazeBaseVR CSV files with columns such as `n,x,y,lx,ly,rx,ry`, downsamples from 250 Hz to 50 Hz by default, and saves `eye_transformer_pretrained.pt`.

### Stage 2: Unity finetuning and export

```powershell
python Tools\train_eye_transformer_modified.py `
  --stage finetune `
  --frame_dir F:\TestData `
  --pretrained_ckpt F:\TestData\OUT_DIR\eye_transformer_pretrained.pt `
  --out_dir F:\TestData\OUT_DIR `
  --export_onnx Assets\Models\eye_transformer_finetuned.onnx `
  --stats_json Assets\Resources\eye_transformer_stats.json `
  --aoi_label_base 0 `
  --threshold 0.3
```

The exported model uses a 150-frame by 47-feature input and emits 10 sigmoid scores for controls `0-9`. Keep the ONNX and stats JSON together; `OnnxIntentionInference` rejects mismatched stats instead of silently running with the wrong feature order.

### Runtime UI polish

`VRDashboardRuntimePolish` applies low-glare panel colors, stable auto-sized text, and cleaner placeholder copy at runtime so the existing scene bindings stay intact while the dashboard reads less like a raw debug prototype.



