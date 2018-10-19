using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Data;
using static AnalisisSentimientos.Sentimiento;
using Microsoft.ML;

namespace AnalisisSentimientos
{
    //Objetivo: Desarrollar ML App para indicar comentarios positivos/negativos
    //Algoritmo de clasificación binaria
    //Datos: Archivos de entrenamiento y prueba
    class Program
    {
        static readonly string _rutaDatosEntrenamiento = @"/Users/rodrigoluque/Projects/AnalisisSentimientos/AnalisisSentimientos/bin/Data/sentiment_labelled_sentences/imdb_labelled.txt";
        static readonly string _rutaDatosPrueba = @"/Users/rodrigoluque/Projects/AnalisisSentimientos/AnalisisSentimientos/bin/Data/sentiment_labelled_sentences/yelp_labelled.txt";

        static void Main(string[] args)
        {
            var modelo = EntrenayPredice();
            Evalua(modelo);
            Console.ReadLine();
        }
        public static PredictionModel<DatosSentimiento, PredictSentimiento> EntrenayPredice()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<DatosSentimiento>(_rutaDatosEntrenamiento, hasHeader: false, separator: "tab"));
            pipeline.Add(new TextFeaturizer("Features", "Texto"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });
           

            PredictionModel <DatosSentimiento, PredictSentimiento> modelo = pipeline.Train<DatosSentimiento, PredictSentimiento>();


            IEnumerable<DatosSentimiento> sentimientos = new[]{
                new DatosSentimiento{
                    Texto = "This movie was boring",
                    Etiqueta=0
                },
                 new DatosSentimiento{
                    Texto = "The movie did not get my attention",
                    Etiqueta=0
                },
                 new DatosSentimiento{
                    Texto = "A super exciting and entertaining movie",
                    Etiqueta=1
                },
            };
            var predicciones = modelo.Predict(sentimientos);

            Console.WriteLine();
            Console.WriteLine("Predicción de sentimientos");
            Console.WriteLine("----------------------------");
            var sentimientosyPredicciones = sentimientos.Zip(predicciones, (sent, predic) => (sent, predic));
            foreach (var item in sentimientosyPredicciones)
            {
                Console.WriteLine($"Sentimiento: {item.sent.Texto} | Predicción: {(item.predic.Etiqueta ? "Positivo:)" : "Negativo: (")}");
            
            }
            Console.WriteLine();

            return modelo;
    
        }
    public static void Evalua (PredictionModel<DatosSentimiento,PredictSentimiento> modelo){
            var datosPrueba = (new TextLoader<DatosSentimiento>(_rutaDatosEntrenamiento, useHeader: false, separator: "tab"));
            var evaluador = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metricas = evaluador.Evaluate(modelo, datosPrueba);

            Console.WriteLine();
            Console.WriteLine("Evaluación de métricas de calidad del modelo de Predicción");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Precisión: {metricas.Accuracy:P2}");
            Console.WriteLine($"AUC: {metricas.Auc:P2}");
        }
    }


}
