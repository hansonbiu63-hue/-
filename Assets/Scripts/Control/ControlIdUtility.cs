using System.Collections.Generic;
using UnityEngine;

public static class ControlIdUtility
{
    public static void NormalizeLegacyOneBasedIds(IList<ControlItem> controls, string contextTag)
    {
        if (controls == null || controls.Count == 0)
        {
            return;
        }

        bool hasZero = false;
        bool allWithinLegacyRange = true;
        int validCount = 0;

        for (int i = 0; i < controls.Count; i++)
        {
            ControlItem item = controls[i];
            if (item == null)
            {
                continue;
            }

            validCount++;
            if (item.controlID == 0)
            {
                hasZero = true;
            }

            if (item.controlID < 1 || item.controlID > 10)
            {
                allWithinLegacyRange = false;
            }
        }

        bool shouldShiftLegacyRange = validCount > 0 && !hasZero && allWithinLegacyRange;
        if (shouldShiftLegacyRange)
        {
            for (int i = 0; i < controls.Count; i++)
            {
                ControlItem item = controls[i];
                if (item == null)
                {
                    continue;
                }

                item.controlID -= 1;
            }

            Debug.LogWarning($"[{contextTag}] Detected legacy 1-10 control IDs. Auto-normalized to 0-9.");
        }

        ValidateZeroBasedIds(controls, contextTag);
    }

    public static void ValidateZeroBasedIds(IList<ControlItem> controls, string contextTag)
    {
        if (controls == null)
        {
            return;
        }

        HashSet<int> seen = new HashSet<int>();
        for (int i = 0; i < controls.Count; i++)
        {
            ControlItem item = controls[i];
            if (item == null)
            {
                continue;
            }

            if (item.controlID < 0 || item.controlID > 9)
            {
                Debug.LogWarning($"[{contextTag}] controlID out of 0-9 range on {item.name}: {item.controlID}");
            }

            if (!seen.Add(item.controlID))
            {
                Debug.LogWarning($"[{contextTag}] Duplicate controlID detected on {item.name}: {item.controlID}");
            }
        }
    }
}
