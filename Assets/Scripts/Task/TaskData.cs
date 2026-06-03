using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class TaskData
{
    public TaskType taskType;
    public List<int> targetControls; // АэИз {1,5,6}
    public float timeLimit;           // 10s / 60s
}
