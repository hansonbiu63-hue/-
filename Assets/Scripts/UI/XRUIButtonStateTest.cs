using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class XRUIButtonStateToUIText : MonoBehaviour,
    IPointerEnterHandler,
    IPointerExitHandler,
    IPointerClickHandler
{
    [Header("UI 显示")]
    public Text statusText;   // 拖一个界面上的 Text 进来

    void Start()
    {
        if (statusText != null)
            statusText.text = "等待射线交互...";
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        if (statusText != null)
            statusText.text = "射线进入 Button";
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        if (statusText != null)
            statusText.text = "射线离开 Button";
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        if (statusText != null)
            statusText.text = "Button 被点击";
    }
}
