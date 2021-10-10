internal interface ISentimentService
{
    SentimentPrediction Predict(string modelPath, string text);
    void Train(List<SentimentData> list, string? modelPath);
}
internal class SentimentService : ISentimentService
{
    /// <summary>
    /// 
    /// </summary>
    /// <param name="text"></param>
    /// <returns></returns>
    public SentimentPrediction Predict(string modelPath, string text)
    {
        Stopwatch st = new Stopwatch();
        st.Start();
        MLContext mlContext = new MLContext();
        var tranformer = mlContext.Model.Load(modelPath, out DataViewSchema schema);
        var prdictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(tranformer);
        SentimentPrediction result = prdictionEngine.Predict(new SentimentData { SentimentText = text });
        st.Stop();
        Console.WriteLine($"----- Phân tích cảm xúc: \"{text}\"");
        Console.WriteLine($"- Cảm xúc: {(result.Prediction ? "Tích cực" : "Tiêu cực")}");
        Console.WriteLine($"- Xác suất/chỉ số: {result.Probability} [{(result.Probability*100):##,#0.##}%]");
        Console.WriteLine($"- Điểm: {result.Score}");
        Console.WriteLine($"----- Thời gian xử lý: {st.Elapsed}");
        return result;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="list"></param>
    /// <param name="modelPath"></param>
    public void Train(List<SentimentData> list, string? modelPath)
    {
        Stopwatch st = new Stopwatch();
        st.Start();

        MLContext mlContext = new MLContext();
        var dataView = LoadData(mlContext, list);
        TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        var trainedModel = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
        Evaluate(mlContext, trainedModel, splitDataView.TestSet);
        mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
        
        st.Stop();
        Console.WriteLine($"----- Hoàn tất đào tạo mô hình! thời gian xử lý: {st.Elapsed}");
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="list"></param>
    /// <returns></returns>
    private IDataView LoadData(MLContext mlContext, List<SentimentData> list)
    {
        Console.WriteLine($"----- Tải dữ liệu, xây dựng và đào tạo mo hình:");
        Console.WriteLine($"- Danh sách cảm xúc nguồn: {list.Count:##,#0}");
        Console.WriteLine($"- Cảm xúc tích cực: {list.Count(k => k.Sentiment):##,#0}");
        Console.WriteLine($"- Cảm xúc tiêu cực: {list.Count(k => !k.Sentiment):##,#0}");
        IDataView dataView = mlContext.Data.LoadFromEnumerable(list);
        return dataView;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="trainSet"></param>
    /// <returns></returns>
    private ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainSet)
    {
        var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText));
        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline.Append(trainer);
        ITransformer trainedModel = trainingPipeline.Fit(trainSet);
        return trainedModel;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="mlContext"></param>
    /// <param name="model"></param>
    /// <param name="splitTestSet"></param>
    private void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
    {
        IDataView predictions = model.Transform(splitTestSet);
        CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label", scoreColumnName: "Score");
        Console.WriteLine($"----- Đánh giá mô hình, số liệu thống kê độ chính xác:");
        Console.WriteLine($"- Accuracy: {metrics.Accuracy}");
        Console.WriteLine($"- ConfusionMatrix: {JsonConvert.SerializeObject(metrics.ConfusionMatrix)}");
        Console.WriteLine($"- Entropy: {metrics.Entropy}");
        Console.WriteLine($"- F1Score: {metrics.F1Score}");

        Console.WriteLine($"- AreaUnderRocCurve: {metrics.AreaUnderRocCurve}");
        Console.WriteLine($"- AreaUnderPrecisionRecallCurve: {metrics.AreaUnderPrecisionRecallCurve}");

        Console.WriteLine($"- LogLoss: {metrics.LogLoss}");
        Console.WriteLine($"- LogLossReduction: {metrics.LogLossReduction}");

        Console.WriteLine($"- NegativePrecision: {metrics.NegativePrecision}");
        Console.WriteLine($"- NegativeRecall: {metrics.NegativeRecall}");

        Console.WriteLine($"- PositivePrecision: {metrics.PositivePrecision}");
        Console.WriteLine($"- PositiveRecall: {metrics.PositiveRecall}");
    }
}