using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace AnalisisSentimientos
{
    class Sentimiento
    {
        public class DatosSentimiento
        {
            [Column(ordinal: "0")]
            public string Texto;
            [Column(ordinal: "1", name: "Label")]
            public float Etiqueta;
        }



        public class PredictSentimiento
        {

            [ColumnName("PredictedLabel")]
            public bool Etiqueta;
        }


    }
}
