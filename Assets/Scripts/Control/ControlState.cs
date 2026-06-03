using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum ControlState
{
    Idle,       // 未触发（数值 < 阈值）
    Triggered   // 已触发（数值 >= 阈值）
}
