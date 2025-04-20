using System;
using System.Collections.Generic;
using System.IO;

class ModelTrainer
{
    static int vocabSize; // update if needed
    static int embeddingDim = 64;
    static int hiddenDim = 128;
    static int batchSize = 32;
    static int epochs = 10;
    static double learningRate = 0.01;

    static double[,] inputEmbedding;
    static double[,] outputEmbedding;
    static double[,] rnnWeights; // [hiddenDim, embeddingDim + hiddenDim]
    static double[,] outputWeights; // [vocabSize, hiddenDim]

    static List<int[]> inputSequences = new List<int[]>();
    static List<int[]> outputSequences = new List<int[]>();

    static void Main()
    {
        Console.WriteLine("🔄 Loading padded training data...");
        LoadData();

        Console.WriteLine("⚙️ Initializing model parameters...");
        InitializeParameters();

        Console.WriteLine("🚀 Starting training...");
for (int epoch = 1; epoch <= epochs; epoch++)
{
    double totalLoss = 0;
    int totalTokens = 0;

    var (inputBatches, outputBatches) = CreateBatches();

    Console.WriteLine($"\n📦 Epoch {epoch}/{epochs} | Batches: {inputBatches.Count}");
    
    for (int i = 0; i < inputBatches.Count; i++)
    {
        var inputs = inputBatches[i];
        var targets = outputBatches[i];

        double loss = TrainStep(inputs, targets);
        totalLoss += loss;
        totalTokens += inputs.Length * inputs[0].Length;

        double percent = ((i + 1) / (double)inputBatches.Count) * 100;
        Console.Write($"\r   🔁 Batch {i + 1}/{inputBatches.Count} ({percent:F1}%) | Loss: {loss:F4}     ");
    }

    Console.WriteLine($"\n✅ Epoch {epoch}/{epochs} | Avg Loss per Token: {totalLoss / totalTokens:F4}");
}


        Console.WriteLine("🎉 Training complete!");
        SaveAllWeights();

    }

    static void LoadData()
{
    int maxToken = 0;

    foreach (var line in File.ReadAllLines("_PREP_DATA/padded_inputs.txt"))
    {
        var tokens = Array.ConvertAll(line.Split(), int.Parse);
        inputSequences.Add(tokens);
        maxToken = Math.Max(maxToken, tokens.Max());
    }

    foreach (var line in File.ReadAllLines("_PREP_DATA/padded_outputs.txt"))
    {
        var tokens = Array.ConvertAll(line.Split(), int.Parse);
        outputSequences.Add(tokens);
        maxToken = Math.Max(maxToken, tokens.Max());
    }

    vocabSize = maxToken + 1; // +1 for zero-index padding token
}


    static void InitializeParameters()
    {
        Random rand = new Random();

        inputEmbedding = new double[vocabSize, embeddingDim];
        outputEmbedding = new double[vocabSize, embeddingDim];
        rnnWeights = new double[hiddenDim, embeddingDim + hiddenDim];
        outputWeights = new double[vocabSize, hiddenDim];

        void Init(double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = (rand.NextDouble() - 0.5) * 0.1;
        }

        Init(inputEmbedding);
        Init(outputEmbedding);
        Init(rnnWeights);
        Init(outputWeights);
    }



    static List<int[][]> SplitIntoChunks(List<int[]> data, int chunkSize)
        {
            var chunks = new List<int[][]>();
            for (int i = 0; i < data.Count; i += chunkSize)
            {
                int size = Math.Min(chunkSize, data.Count - i);
                chunks.Add(data.GetRange(i, size).ToArray());
            }
            return chunks;
        }


    static (List<int[][]>, List<int[][]>) CreateBatches()
    {
        List<int[]> inputs = new(inputSequences);
        List<int[]> targets = new(outputSequences);

        // Shuffle together
        Random rand = new Random();
        for (int i = 0; i < inputs.Count; i++)
        {
            int j = rand.Next(i, inputs.Count);
            (inputs[i], inputs[j]) = (inputs[j], inputs[i]);
            (targets[i], targets[j]) = (targets[j], targets[i]);
        }

        List<int[][]> inputBatches = new();
        List<int[][]> outputBatches = new();

        for (int i = 0; i < inputs.Count; i += batchSize)
        {
           inputBatches.AddRange(SplitIntoChunks(inputs, batchSize));
           outputBatches.AddRange(SplitIntoChunks(targets, batchSize));
        }

        return (inputBatches, outputBatches);
    }

    static double TrainStep(int[][] inputs, int[][] targets)
{
    double totalLoss = 0;

    for (int batchIndex = 0; batchIndex < inputs.Length; batchIndex++)
    {
        int[] inputSeq = inputs[batchIndex];
        int[] targetSeq = targets[batchIndex];

        double[,] hidden = new double[inputSeq.Length, hiddenDim];

        // Initial hidden state = zeros
        double[] prevHidden = new double[hiddenDim];

        for (int t = 0; t < inputSeq.Length; t++)
        {
            int token = inputSeq[t];
            double[] x = GetRow(inputEmbedding, token);
            double[] combined = Concatenate(x, prevHidden);
            double[] h = Tanh(MatVecMul(rnnWeights, combined));

            for (int j = 0; j < hiddenDim; j++)
                hidden[t, j] = h[j];

            prevHidden = h;
        }

        // Compute logits from final hidden state
        double[] logits = MatVecMul(outputWeights, prevHidden);
        double[] probs = Softmax(logits);

        int targetToken = targetSeq.Last(); // predict last token only for now
        double loss = -Math.Log(probs[targetToken] + 1e-9); // cross-entropy loss

        totalLoss += loss;
    }

    return totalLoss;
}
static double[] MatVecMul(double[,] matrix, double[] vector)
{
    int rows = matrix.GetLength(0);
    int cols = matrix.GetLength(1);
    double[] result = new double[rows];

    for (int i = 0; i < rows; i++)
    {
        result[i] = 0;
        for (int j = 0; j < cols; j++)
            result[i] += matrix[i, j] * vector[j];
    }

    return result;
}

static double[] Tanh(double[] vector)
{
    double[] result = new double[vector.Length];
    for (int i = 0; i < vector.Length; i++)
        result[i] = Math.Tanh(vector[i]);
    return result;
}

static double[] GetRow(double[,] matrix, int row)
{
    int cols = matrix.GetLength(1);
    double[] result = new double[cols];
    for (int i = 0; i < cols; i++)
        result[i] = matrix[row, i];
    return result;
}

static double[] Concatenate(double[] a, double[] b)
{
    double[] result = new double[a.Length + b.Length];
    a.CopyTo(result, 0);
    b.CopyTo(result, a.Length);
    return result;
}

static double[] Softmax(double[] logits)
{
    double maxLogit = logits.Max();
    double[] exps = logits.Select(x => Math.Exp(x - maxLogit)).ToArray();
    double sum = exps.Sum();
    return exps.Select(x => x / sum).ToArray();
}
static void SaveMatrix(string filePath, double[,] matrix)
{
    using (StreamWriter writer = new StreamWriter(filePath))
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        for (int i = 0; i < rows; i++)
        {
            string line = "";
            for (int j = 0; j < cols; j++)
            {
                line += matrix[i, j].ToString("F6") + " ";
            }
            writer.WriteLine(line.Trim());
        }
    }
}
static void SaveAllWeights()
{
    string folder = "_MODEL_WEIGHTS";
    Directory.CreateDirectory(folder);

    SaveMatrix(Path.Combine(folder, "inputEmbedding.txt"), inputEmbedding);
    SaveMatrix(Path.Combine(folder, "outputEmbedding.txt"), outputEmbedding);
    SaveMatrix(Path.Combine(folder, "rnnWeights.txt"), rnnWeights);
    SaveMatrix(Path.Combine(folder, "outputWeights.txt"), outputWeights);

    Console.WriteLine("💾 All weights saved to _MODEL_WEIGHTS folder.");
}

}
