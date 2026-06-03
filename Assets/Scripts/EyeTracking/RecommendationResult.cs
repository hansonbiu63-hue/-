using System;
using System.Collections.Generic;

[Serializable]
public class RecommendationResult
{
    public string source;
    public List<int> rawTopIds = new List<int>();
    public List<int> displayTopIds = new List<int>();
    public List<float> scores = new List<float>();
    public long windowStartMs;
    public long windowEndMs;
    public bool isPostProcessed;

    public RecommendationResult()
    {
    }

    public RecommendationResult(string sourceName, long startMs, long endMs)
    {
        source = sourceName;
        windowStartMs = startMs;
        windowEndMs = endMs;
    }
}
