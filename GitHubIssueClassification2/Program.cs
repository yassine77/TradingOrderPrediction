using GitHubIssueClassification2;
using Microsoft.ML;

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "btcusdt_train.csv");
string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "btcusdt_test.csv");
string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model_trading.zip");

MLContext _mlContext;
PredictionEngine<Candle, OrderPrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;





_mlContext = new MLContext(seed: 0);
_trainingDataView = _mlContext.Data.LoadFromTextFile<Candle>(_trainDataPath, hasHeader: true);

var pipeline = ProcessData();
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
Evaluate(_trainingDataView.Schema);
PredictIssue();







IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Order", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Date", outputColumnName: "DateFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Close", outputColumnName: "CloseFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Volume", outputColumnName: "VolumeFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ShortSMA", outputColumnName: "ShortSMAFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "LongSMA", outputColumnName: "LongSMAFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ShortEMA", outputColumnName: "ShortEMAFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "LongEMA", outputColumnName: "LongEMAFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "RSI", outputColumnName: "RSIFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Chopiness", outputColumnName: "ChopinessFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Boll_LowerBand", outputColumnName: "Boll_LowerBandFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Boll_PercentB", outputColumnName: "Boll_PercentBFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Boll_UpperBand", outputColumnName: "Boll_UpperBandFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Boll_Width", outputColumnName: "Boll_WidthFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Boll_ZScore", outputColumnName: "Boll_ZScoreFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "CMF", outputColumnName: "CMFFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "CMF_MoneyFlowMultiplier", outputColumnName: "CMF_MoneyFlowMultiplierFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "CMF_MoneyFlowVolume", outputColumnName: "CMF_MoneyFlowVolumeFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ForceIndex", outputColumnName: "ForceIndexFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "FractalBear", outputColumnName: "FractalBearFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "FractalBull", outputColumnName: "FractalBullFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "StochRSI", outputColumnName: "StochRSIFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "MACD", outputColumnName: "MACDFeaturized"))
        .Append(_mlContext.Transforms.Concatenate("Features", "CloseFeaturized", "VolumeFeaturized", "ShortSMAFeaturized",
        "LongSMAFeaturized", "ShortEMAFeaturized", "LongEMAFeaturized", "RSIFeaturized", "ChopinessFeaturized", "Boll_LowerBandFeaturized",
        "Boll_PercentBFeaturized", "Boll_UpperBandFeaturized", "Boll_WidthFeaturized", "Boll_ZScoreFeaturized", "CMFFeaturized",
        "CMF_MoneyFlowMultiplierFeaturized", "CMF_MoneyFlowVolumeFeaturized", "ForceIndexFeaturized", "FractalBearFeaturized",
        "FractalBullFeaturized", "StochRSIFeaturized", "MACDFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline;
}

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    _trainedModel = trainingPipeline.Fit(trainingDataView);
    _predEngine = _mlContext.Model.CreatePredictionEngine<Candle, OrderPrediction>(_trainedModel);

    Candle issue = new Candle()
    {
        Close = "37720,93",
        Volume = "193470956",
        ShortSMA = "38695,14714",
        LongSMA = "38881,39952",
        ShortEMA = "38441,4096",
        LongEMA = "38967,966",
        RSI = "24,02650898",
        Chopiness = "42,21714794",
        Boll_LowerBand = "37210,895",
        Boll_PercentB = "0,25",
        Boll_UpperBand = "39251,035",
        Boll_Width = "0,05336355",
        Boll_ZScore = "-1",
        CMF = "-0,202887569",
        CMF_MoneyFlowMultiplier = "-0,787342623",
        CMF_MoneyFlowVolume = "-152327929,9",
        ForceIndex = "-29072782263",
        FractalBear = "0",
        FractalBull = "0",
        StochRSI = "35,10017182",
        MACD = "-153,5439788"
    };
    var prediction = _predEngine.Predict(issue);
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Order} ===============");
    return trainingPipeline;
}

void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = _mlContext.Data.LoadFromTextFile<Candle>(_testDataPath, hasHeader: true);
    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
}

void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
    Candle singleIssue = new Candle() 
    {
        Close = "42191,28",
        Volume = "70806949,84",
        ShortSMA = "41980,95143",
        LongSMA = "41822,23333",
        ShortEMA = "42037,73013",
        LongEMA = "41421,28433",
        RSI = "68,5534308",
        Chopiness = "48,69899162",
        Boll_LowerBand = "41956,68",
        Boll_PercentB = "0,75",
        Boll_UpperBand = "42269,48",
        Boll_Width = "0,007427621",
        Boll_ZScore = "1",
        CMF = "0,119874876",
        CMF_MoneyFlowMultiplier = "0,171778898",
        CMF_MoneyFlowVolume = "12163139,79",
        ForceIndex = "3145970446",
        FractalBear = "42400",
        FractalBull = "0",
        StochRSI = "97,07635967",
        MACD = "191,3464127"
    };
    _predEngine = _mlContext.Model.CreatePredictionEngine<Candle, OrderPrediction>(loadedModel);
    var prediction = _predEngine.Predict(singleIssue);
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Order} ===============");
}
