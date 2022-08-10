using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace GitHubIssueClassification2
{
    public class Candle
    {
        [LoadColumn(0)]
        public string Date { get; set; }
        [LoadColumn(1)]
        public string Order { get; set; }
        [LoadColumn(2)]
        public string Close { get; set; }
        [LoadColumn(3)]
        public string Volume { get; set; }
        [LoadColumn(4)]
        public string ShortSMA { get; set; }
        [LoadColumn(5)]
        public string LongSMA { get; set; }
        [LoadColumn(6)]
        public string ShortEMA { get; set; }
        [LoadColumn(7)]
        public string LongEMA { get; set; }
        [LoadColumn(8)]
        public string RSI { get; set; }
        [LoadColumn(9)]
        public string Chopiness { get; set; }
        [LoadColumn(10)]
        public string Boll_LowerBand { get; set; }
        [LoadColumn(11)]
        public string Boll_PercentB { get; set; }
        [LoadColumn(12)]
        public string Boll_UpperBand { get; set; }
        [LoadColumn(13)]
        public string Boll_Width { get; set; }
        [LoadColumn(14)]
        public string Boll_ZScore { get; set; }
        [LoadColumn(15)]
        public string CMF { get; set; }
        [LoadColumn(16)]
        public string CMF_MoneyFlowMultiplier { get; set; }
        [LoadColumn(17)]
        public string CMF_MoneyFlowVolume { get; set; }
        [LoadColumn(18)]
        public string ForceIndex { get; set; }
        [LoadColumn(19)]
        public string FractalBear { get; set; }
        [LoadColumn(20)]
        public string FractalBull { get; set; }
        [LoadColumn(21)]
        public string StochRSI { get; set; }
        [LoadColumn(22)]
        public string MACD { get; set; }
    }

    public class OrderPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Order;
    }
}
