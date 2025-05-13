// <SnippetAddUsings>
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using System.Data;
// </SnippetAddUsings>

namespace SentimentAnalysis
{
    class Program
    {
        // <SnippetDeclareGlobalVariables>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "mindWaveDataJasiu_cleansedData.csv");
        // </SnippetDeclareGlobalVariables>

        static void Main(string[] args)
        {
            // Create ML.NET context/local environment - allows you to add steps in order to keep everything together
            // as you discover the ML.NET trainers and transforms
            // <SnippetCreateMLContext>
            MLContext mlContext = new MLContext();
            // </SnippetCreateMLContext>

            // <SnippetCallLoadData>
            TrainTestData splitDataView = LoadData(mlContext);
            // </SnippetCallLoadData>

            // <SnippetCallBuildAndTrainModel>
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            // </SnippetCallBuildAndTrainModel>

            // <SnippetCallEvaluate>
            Evaluate(mlContext, model, splitDataView.TestSet);
            // </SnippetCallEvaluate>

            // <SnippetCallUseModelWithSingleItem>
            UseModelWithSingleItem(mlContext, model);
            // </SnippetCallUseModelWithSingleItem>

            // <SnippetCallUseModelWithBatchItems>
            UseModelWithBatchItems(mlContext, model);
            // </SnippetCallUseModelWithBatchItems>

            Console.WriteLine();
            Console.WriteLine("=============== End of process ===============");
        }

        public static TrainTestData LoadData(MLContext mlContext)
        {
            // Note that this case, loading your training data from a file,
            // is the easiest way to get started, but ML.NET also allows you
            // to load data from databases or in-memory collections.
            // <SnippetLoadData>
            IDataView dataView = mlContext.Data.LoadFromTextFile<mindWaveJasiuData>(
                path: _dataPath,
                hasHeader: true,
                separatorChar: ';',
                trimWhitespace: true
            );
            // </SnippetLoadData>

            // You need both a training dataset to train the model and a test dataset to evaluate the model.
            // Split the loaded dataset into train and test datasets
            // Specify test dataset percentage with the `testFraction`parameter
            // <SnippetSplitData>
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            // </SnippetSplitData>

            // <SnippetReturnSplitData>
            return splitDataView;
            // </SnippetReturnSplitData>
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            // Create a flexible pipeline (composed by a chain of estimators) for creating/training the model.
            // This is used to format and clean the data.
            // Convert the text column to numeric vectors (Features column)
            // <SnippetFeaturizeText>
            var estimator = mlContext.Transforms.Concatenate("Features",
                nameof(mindWaveJasiuData.Delta),
                nameof(mindWaveJasiuData.Theta),
                nameof(mindWaveJasiuData.LowAlpha),
                nameof(mindWaveJasiuData.HighAlpha),
                nameof(mindWaveJasiuData.LowBeta),
                nameof(mindWaveJasiuData.HighBeta),
                nameof(mindWaveJasiuData.LowGamma),
                nameof(mindWaveJasiuData.HighGamma),
                nameof(mindWaveJasiuData.Attention),
                nameof(mindWaveJasiuData.Meditation))
            //</SnippetFeaturizeText>
            // append the machine learning task to the estimator
            // <SnippetAddTrainer>
            .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
            // </SnippetAddTrainer>

            // Create and train the model based on the dataset that has been loaded, transformed.
            // <SnippetTrainModel>
            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();
            // </SnippetTrainModel>

            // Returns the model we trained to use for evaluation.
            // <SnippetReturnModel>
            return model;
            // </SnippetReturnModel>
        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            // Evaluate the model and show accuracy stats

            //Take the data in, make transformations, output the data.
            // <SnippetTransformData>
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitTestSet);
            // </SnippetTransformData>

            // BinaryClassificationContext.Evaluate returns a BinaryClassificationEvaluator.CalibratedResult
            // that contains the computed overall metrics.
            // <SnippetEvaluate>
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            // </SnippetEvaluate>

            // The Accuracy metric gets the accuracy of a model, which is the proportion
            // of correct predictions in the test set.

            // The AreaUnderROCCurve metric is equal to the probability that the algorithm ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the model's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            // <SnippetDisplayMetrics>
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            //</SnippetDisplayMetrics>
        }

        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<mindWaveJasiuData, SentimentPrediction>(model);

            // Przykładowe dane EEG (wstaw tu rzeczywiste lub sensowne liczby)
            var sampleData = new mindWaveJasiuData
            {
                Delta = 30000,
                Theta = 20000,
                LowAlpha = 15000,
                HighAlpha = 14000,
                LowBeta = 10000,
                HighBeta = 9000,
                LowGamma = 8000,
                HighGamma = 7000,
                Attention = 60,
                Meditation = 50
            };

            var resultPrediction = predictionFunction.Predict(sampleData);

            Console.WriteLine();
            Console.WriteLine("=============== Pojedyncza predykcja ===============");
            Console.WriteLine($"Prediction: {(resultPrediction.Prediction ? "Doing Task" : "Not Doing Task")}");
            Console.WriteLine($"Probability: {resultPrediction.Probability:P2}");
            Console.WriteLine($"Score: {resultPrediction.Score}");
            Console.WriteLine("====================================================");
        }


        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            // Przykładowa lista dwóch zestawów danych EEG
            var eegSamples = new List<mindWaveJasiuData>
    {
        new mindWaveJasiuData
        {
            Delta = 31000,
            Theta = 21000,
            LowAlpha = 16000,
            HighAlpha = 15000,
            LowBeta = 11000,
            HighBeta = 9500,
            LowGamma = 8500,
            HighGamma = 7500,
            Attention = 65,
            Meditation = 55
        },
        new mindWaveJasiuData
        {
            Delta = 25000,
            Theta = 19000,
            LowAlpha = 12000,
            HighAlpha = 13000,
            LowBeta = 9000,
            HighBeta = 8700,
            LowGamma = 8200,
            HighGamma = 7600,
            Attention = 45,
            Meditation = 40
        }
    };

            var batchData = mlContext.Data.LoadFromEnumerable(eegSamples);
            var predictions = model.Transform(batchData);
            var results = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Predykcje batchowe ===============");
            int i = 1;
            foreach (var prediction in results)
            {
                Console.WriteLine($"Sample {i++} | Prediction: {(prediction.Prediction ? "Doing Task" : "Not Doing Task")} | Probability: {prediction.Probability:P2}");
            }
            Console.WriteLine("===================================================");
        }

    }
}