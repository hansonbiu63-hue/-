using UnityEngine;

public class ConfirmInput : MonoBehaviour
{
    public TaskManager taskManager;

    void Update()
    {
        // 空格 / 手柄按钮都可
        if (Input.GetKeyDown(KeyCode.Space))
        {
            taskManager.TryConfirmTask();
        }
    }
}
