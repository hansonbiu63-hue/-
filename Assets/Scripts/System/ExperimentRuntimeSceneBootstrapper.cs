using UnityEngine;
using UnityEngine.UI;

[DisallowMultipleComponent]
public class ExperimentRuntimeSceneBootstrapper : MonoBehaviour
{
    [Header("Optional Existing Scene Object Names")]
    public string recommendationTruthTextName = "RecommendationTruthFilterText";
    public string evaluatorTextName = "WindowPredictionEvaluatorText";

    [Header("Binding")]
    public bool bindOnStart = true;
    public bool onlyFillMissingReferences = true;

    private void Start()
    {
        if (bindOnStart)
        {
            BindExistingSceneObjects();
        }
    }

    [ContextMenu("Bind Existing Scene Objects")]
    public void BindExistingSceneObjects()
    {
        TaskManager taskManager = FindObjectOfType<TaskManager>();
        RealtimeIntentionRecommender recommender = FindObjectOfType<RealtimeIntentionRecommender>();
        OnnxIntentionInference transformer = FindObjectOfType<OnnxIntentionInference>();
        RandomForestIntentionInference randomForest = FindObjectOfType<RandomForestIntentionInference>();
        RecommendationFeedbackManager feedbackManager = FindObjectOfType<RecommendationFeedbackManager>();
        GazeDataLogger logger = FindObjectOfType<GazeDataLogger>();
        WindowPredictionEvaluator evaluator = FindObjectOfType<WindowPredictionEvaluator>();
        RecommendationTruthFilterDisplay truthDisplay = FindObjectOfType<RecommendationTruthFilterDisplay>();

        if (evaluator != null)
        {
            AssignIfAllowed(ref evaluator.recommender, recommender);
            AssignIfAllowed(ref evaluator.transformerInference, transformer);
            AssignIfAllowed(ref evaluator.randomForestInference, randomForest);
            AssignIfAllowed(ref evaluator.taskManager, taskManager);
            AssignIfAllowed(ref evaluator.dataLogger, logger);

            Text evaluatorText = FindTextByName(evaluatorTextName);
            if (evaluatorText != null)
            {
                AssignIfAllowed(ref evaluator.evaluationText, evaluatorText);
            }
        }

        if (truthDisplay != null)
        {
            AssignIfAllowed(ref truthDisplay.feedbackManager, feedbackManager);
            AssignIfAllowed(ref truthDisplay.taskManager, taskManager);

            Text truthText = FindTextByName(recommendationTruthTextName);
            if (truthText != null)
            {
                AssignIfAllowed(ref truthDisplay.resultText, truthText);
            }
        }
    }

    private void AssignIfAllowed<T>(ref T field, T value) where T : class
    {
        if (value == null)
        {
            return;
        }

        if (!onlyFillMissingReferences || field == null)
        {
            field = value;
        }
    }

    private static GameObject FindGameObjectByName(string objectName)
    {
        if (string.IsNullOrWhiteSpace(objectName))
        {
            return null;
        }

        GameObject direct = GameObject.Find(objectName);
        if (direct != null)
        {
            return direct;
        }

        Transform[] transforms = Resources.FindObjectsOfTypeAll<Transform>();
        for (int i = 0; i < transforms.Length; i++)
        {
            Transform t = transforms[i];
            if (t != null && t.gameObject.scene.IsValid() && t.name == objectName)
            {
                return t.gameObject;
            }
        }

        return null;
    }

    private static Text FindTextByName(string objectName)
    {
        GameObject go = FindGameObjectByName(objectName);
        return go != null ? go.GetComponent<Text>() : null;
    }
}
