// <SnippetAddUsings>
using Microsoft.ML.Data;
// </SnippetAddUsings>

namespace SentimentAnalysis
{
    // <SnippetDeclareTypes>
    public class mindWaveJasiuData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool doingTask;
        // 0 - not doing task, 1 - doing task

        [LoadColumn(1), ColumnName("Delta")]
        public float Delta;
        [LoadColumn(2), ColumnName("Theta")]
        public float Theta;
        [LoadColumn(3), ColumnName("LowAlpha")]
        public float LowAlpha;
        [LoadColumn(4), ColumnName("HighAlpha")]
        public float HighAlpha;
        [LoadColumn(5), ColumnName("LowBeta")]
        public float LowBeta;
        [LoadColumn(6), ColumnName("HighBeta")]
        public float HighBeta;
        [LoadColumn(7), ColumnName("LowGamma")]
        public float LowGamma;
        [LoadColumn(8), ColumnName("HighGamma")]
        public float HighGamma;
        [LoadColumn(9), ColumnName("Attention")]
        public float Attention;
        [LoadColumn(10), ColumnName("Meditation")]
        public float Meditation;
    }

    public class SentimentPrediction
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
    // </SnippetDeclareTypes>
}
