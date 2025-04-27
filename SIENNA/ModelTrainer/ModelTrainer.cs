using System;
using System.Collections.Generic;
using System.IO;
using Python.Runtime;

class ModelTrainer
{
    static int vocabSize; // update if needed
    static int embeddingDim = 16;
    static int hiddenDim = 32;
    static int batchSize = 8;
    static int epochs = 1;
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
        Console.WriteLine("\n🧪 Enter test input (token IDs separated by space): ");
        string input = Console.ReadLine();
        int[] inputTokens = input.Split().Select(int.Parse).ToArray();

        List<int> response = Predict(inputTokens);
        Console.WriteLine("🤖 Model response:");
        Console.WriteLine(string.Join(" ", response));

    }

    static void LoadData()
{
    // 1. Load all padded sequences
    foreach (var line in File.ReadAllLines("_PREP_DATA/padded_inputs.txt"))
        inputSequences.Add(Array.ConvertAll(line.Split(), int.Parse));

    foreach (var line in File.ReadAllLines("_PREP_DATA/padded_outputs.txt"))
        outputSequences.Add(Array.ConvertAll(line.Split(), int.Parse));

    // 2. Set vocab size manually by vocab file size
    vocabSize = File.ReadLines("_VOCAB/vocab.txt").Count(); // 🔥
    Console.WriteLine($"✅ Vocab size loaded from vocab.txt: {vocabSize}");
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
            inputBatches.Add(inputs.GetRange(i, Math.Min(batchSize, inputs.Count - i)).ToArray());
            outputBatches.Add(targets.GetRange(i, Math.Min(batchSize, targets.Count - i)).ToArray());
        }

        return (inputBatches, outputBatches);
    }

   static double TrainStep(int[][] inputs, int[][] targets)
{
    double totalLoss = 0;

    // Initialize gradient accumulators
    double[,] dRnnWeights = new double[hiddenDim, embeddingDim + hiddenDim];
    double[,] dOutputWeights = new double[vocabSize, hiddenDim];

    for (int b = 0; b < inputs.Length; b++)
    {
        int[] inputSeq = inputs[b];
        int[] targetSeq = targets[b];

        int T = inputSeq.Length;
        double[][] hiddenStates = new double[T + 1][]; // include h0
        hiddenStates[0] = new double[hiddenDim]; // initial h0 = zeros

        double[][] logitsPerTimestep = new double[T][];
        double[][] probsPerTimestep = new double[T][];

        // FORWARD PASS 🔽
        for (int t = 0; t < T; t++)
        {
            double[] x = GetRow(inputEmbedding, inputSeq[t]);
            double[] combined = Concatenate(x, hiddenStates[t]); // [embedding + hidden]
            double[] h = Tanh(MatVecMul(rnnWeights, combined));

            hiddenStates[t + 1] = h;

            double[] logits = MatVecMul(outputWeights, h);
            double[] probs = Softmax(logits);

            logitsPerTimestep[t] = logits;
            probsPerTimestep[t] = probs;

            int target = targetSeq[t];
            totalLoss += -Math.Log(probs[target] + 1e-9);
        }

        // BACKWARD PASS 🔼
        double[] dhNext = new double[hiddenDim]; // gradient from next timestep

        for (int t = T - 1; t >= 0; t--)
        {
            int target = targetSeq[t];
            double[] probs = probsPerTimestep[t];

            // ∂L/∂logits
            double[] dLogits = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++)
                dLogits[i] = probs[i];
            dLogits[target] -= 1; // derivative of softmax + cross-entropy

            // ∂L/∂outputWeights += dLogits ⊗ h^T
            double[] h = hiddenStates[t + 1];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < hiddenDim; j++)
                    dOutputWeights[i, j] += dLogits[i] * h[j];

            // ∂L/∂h
            double[] dh = new double[hiddenDim];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < hiddenDim; j++)
                    dh[j] += dLogits[i] * outputWeights[i, j];

            for (int j = 0; j < hiddenDim; j++)
                dh[j] += dhNext[j]; // add gradient from future time step

            // tanh derivative
            for (int j = 0; j < hiddenDim; j++)
                dh[j] *= (1 - h[j] * h[j]);

            // ∂L/∂rnnWeights
            double[] prevH = hiddenStates[t];
            double[] x = GetRow(inputEmbedding, inputSeq[t]);
            double[] combined = Concatenate(x, prevH);

            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < combined.Length; j++)
                    dRnnWeights[i, j] += dh[i] * combined[j];

            // ∂L/∂prevHidden (for next timestep backprop)
            dhNext = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
                for (int i = 0; i < hiddenDim; i++)
                    dhNext[j] += dh[i] * rnnWeights[i, embeddingDim + j];
        }
    }


    // Gradient Clipping (before weight updates)
    ClipGradients(dRnnWeights, 1.0);
    ClipGradients(dOutputWeights, 1.0);

    // Apply SGD updates ✍️
    for (int i = 0; i < rnnWeights.GetLength(0); i++)
        for (int j = 0; j < rnnWeights.GetLength(1); j++)
            rnnWeights[i, j] -= learningRate * dRnnWeights[i, j];

    for (int i = 0; i < outputWeights.GetLength(0); i++)
        for (int j = 0; j < outputWeights.GetLength(1); j++)
            outputWeights[i, j] -= learningRate * dOutputWeights[i, j];

    int totalTokens = inputs.Sum(seq => seq.Length);
    return totalLoss / totalTokens;

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
    // Create directory if it doesn't exist
    Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);

    using (StreamWriter writer = new StreamWriter(filePath, false)) // overwrite mode
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
static void ClipGradients(double[,] gradients, double clipValue)
{
    for (int i = 0; i < gradients.GetLength(0); i++)
        for (int j = 0; j < gradients.GetLength(1); j++)
            gradients[i, j] = Math.Max(Math.Min(gradients[i, j], clipValue), -clipValue);
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



static int  maxPredictionLength = 30; // max tokens to generate

static List<int> Predict(int[] inputTokens)
{
    List<int> generated = new();

    double[] hidden = new double[hiddenDim];
    
    // Feed the input tokens through the RNN to get the final hidden state
    foreach (int token in inputTokens)
    {
        double[] x = GetRow(inputEmbedding, token);
        double[] combined = Concatenate(x, hidden);
        hidden = Tanh(MatVecMul(rnnWeights, combined));
    }

    int currentToken = inputTokens.Last();

    for (int step = 0; step < maxPredictionLength; step++)
    {
        // Embed the current token
        double[] x = GetRow(outputEmbedding, currentToken);
        double[] combined = Concatenate(x, hidden);
        hidden = Tanh(MatVecMul(rnnWeights, combined));

        // Get next token probabilities
        double[] logits = MatVecMul(outputWeights, hidden);
        double[] probs = Softmax(logits);

        // Greedy decode: pick the highest-probability token
        int nextToken = ArgMax(probs);

        generated.Add(nextToken);

        currentToken = nextToken;

        // Optional: break on some special token like <eos>
        // if (nextToken == EOS_TOKEN_ID) break;
    }

    return generated;
}

static int ArgMax(double[] array)
{
    int maxIndex = 0;
    double maxVal = array[0];

    for (int i = 1; i < array.Length; i++)
    {
        if (array[i] > maxVal)
        {
            maxVal = array[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}
}