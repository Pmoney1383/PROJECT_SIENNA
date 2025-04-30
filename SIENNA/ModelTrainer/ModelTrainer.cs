using System;
using System.Collections.Generic;
using System.IO;
using Python.Runtime;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;


class ModelTrainer
{
    static int vocabSize; // update if needed
    static int embeddingDim = 64;
    static int hiddenDim = 128;
    static int batchSize = 32;
    static int epochs = 5;
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
    var stopwatchTotal = Stopwatch.StartNew();

    int B = inputs.Length;
    int T = inputs[0].Length;
    int correctCount = 0;
    int totalCount = 0;

    double forwardTime = 0;
    double backwardTime = 0;
    double updateTime = 0;
    double totalLoss = 0;

    // Initialize gradient accumulators
    double[,] dRnnWeights = new double[hiddenDim, embeddingDim + hiddenDim];
    double[,] dOutputWeights = new double[vocabSize, hiddenDim];

    // Hidden states per sequence
    double[][][] hiddenStates = new double[B][][];
    for (int b = 0; b < B; b++)
    {
        hiddenStates[b] = new double[T + 1][];
        hiddenStates[b][0] = new double[hiddenDim];
    }

    double[][][] logitsPerSeq = new double[B][][];
    double[][][] probsPerSeq = new double[B][][];

    // 🔽 FORWARD PASS (parallel per sequence)
    var swForward = Stopwatch.StartNew();
    Parallel.For(0, B, new ParallelOptions { MaxDegreeOfParallelism = 12 }, b =>
    {
        double[][] hidden = hiddenStates[b];
        double[][] logitsPerTime = new double[T][];
        double[][] probsPerTime = new double[T][];

        for (int t = 0; t < T; t++)
        {
            double[] x = GetRow(inputEmbedding, inputs[b][t]);
            double[] combined = Concatenate(x, hidden[t]);
            double[] h = Tanh(MatVecMul(rnnWeights, combined));
            hidden[t + 1] = h;

            double[] logits = MatVecMul(outputWeights, h);
            double[] probs = Softmax(logits);

            logitsPerTime[t] = logits;
            probsPerTime[t] = probs;

            double loss = -Math.Log(probs[targets[b][t]] + 1e-9);
            if (ArgMax(probs) == targets[b][t])
            {
                Interlocked.Increment(ref correctCount);
            }
            Interlocked.Increment(ref totalCount);


            lock (typeof(ModelTrainer))
            {
                totalLoss += loss;
            }
        }

        logitsPerSeq[b] = logitsPerTime;
        probsPerSeq[b] = probsPerTime;
    });
    swForward.Stop();
    forwardTime = swForward.Elapsed.TotalMilliseconds;

    // 🔼 BACKWARD PASS (parallel per sequence)
    var swBackward = Stopwatch.StartNew();
    Parallel.For(0, B, new ParallelOptions { MaxDegreeOfParallelism = 12 }, () => (
        new double[hiddenDim, embeddingDim + hiddenDim],
        new double[vocabSize, hiddenDim]
    ),
    (b, _, localGrads) =>
    {
        double[,] localRnnGrad = localGrads.Item1;
        double[,] localOutGrad = localGrads.Item2;

        double[][] hidden = hiddenStates[b];
        double[] dhNext = new double[hiddenDim];

        for (int t = T - 1; t >= 0; t--)
        {
            int target = targets[b][t];
            double[] probs = probsPerSeq[b][t];

            double[] dLogits = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++) dLogits[i] = probs[i];
            dLogits[target] -= 1;

            double[] h = hidden[t + 1];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < hiddenDim; j++)
                    localOutGrad[i, j] += dLogits[i] * h[j];

            double[] dh = new double[hiddenDim];
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < hiddenDim; j++)
                    dh[j] += dLogits[i] * outputWeights[i, j];

            for (int j = 0; j < hiddenDim; j++) dh[j] += dhNext[j];
            for (int j = 0; j < hiddenDim; j++) dh[j] *= (1 - h[j] * h[j]);

            double[] prevH = hidden[t];
            double[] x = GetRow(inputEmbedding, inputs[b][t]);
            double[] combined = Concatenate(x, prevH);

            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < combined.Length; j++)
                    localRnnGrad[i, j] += dh[i] * combined[j];

            dhNext = new double[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
                for (int i = 0; i < hiddenDim; i++)
                    dhNext[j] += dh[i] * rnnWeights[i, embeddingDim + j];
        }

        return (localRnnGrad, localOutGrad);
    },
    localGrads =>
    {
        // Safely combine local gradients into global ones
        lock (dRnnWeights)
        {
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < embeddingDim + hiddenDim; j++)
                    InterlockedAdd(ref dRnnWeights[i, j], localGrads.Item1[i, j]);

        }

        lock (dOutputWeights)
        {
            for (int i = 0; i < vocabSize; i++)
                for (int j = 0; j < hiddenDim; j++)
                    dOutputWeights[i, j] += localGrads.Item2[i, j];
        }
    });
    swBackward.Stop();
    backwardTime = swBackward.Elapsed.TotalMilliseconds;

    // 💥 Apply updates
    var swUpdate = Stopwatch.StartNew();
    ClipGradients(dRnnWeights, 1.0);
    ClipGradients(dOutputWeights, 1.0);

    for (int i = 0; i < rnnWeights.GetLength(0); i++)
        for (int j = 0; j < rnnWeights.GetLength(1); j++)
            rnnWeights[i, j] -= learningRate * dRnnWeights[i, j];

    for (int i = 0; i < outputWeights.GetLength(0); i++)
        for (int j = 0; j < outputWeights.GetLength(1); j++)
            outputWeights[i, j] -= learningRate * dOutputWeights[i, j];
    swUpdate.Stop();
    updateTime = swUpdate.Elapsed.TotalMilliseconds;

    stopwatchTotal.Stop();
    double totalTime = stopwatchTotal.Elapsed.TotalMilliseconds;

    Console.Write($" \r ⏱️ Batch profiling | Tokens: {T * B} | " +
              $"Forward: {forwardTime:F2} ms | " +
              $"Backward: {backwardTime:F2} ms | " +
              $"Update: {updateTime:F2} ms | " +
              $"Total: {totalTime:F2} ms       ");

    double accuracy = 100.0 * correctCount / totalCount;
    Console.WriteLine($" \r | Accuracy: {accuracy:F2}%");

    return totalLoss / (T * B);
}

static void InterlockedAdd(ref double target, double value)
{
    long initialValue, computedValue;
    do
    {
        initialValue = BitConverter.DoubleToInt64Bits(target);
        double initialDouble = BitConverter.Int64BitsToDouble(initialValue);
        double newDouble = initialDouble + value;
        computedValue = BitConverter.DoubleToInt64Bits(newDouble);
    }
    while (Interlocked.CompareExchange(
        ref Unsafe.As<double, long>(ref target),
        computedValue, initialValue) != initialValue);
}


static double[][] MatMulBatch(double[,] matrix, double[][] batch)
{
    int rows = matrix.GetLength(0);      // Output dimension (e.g., hiddenDim or vocabSize)
    int cols = matrix.GetLength(1);      // Input dimension (embeddingDim + hiddenDim)
    int batchSize = batch.Length;

    // Preallocate result
    double[][] result = new double[batchSize][];
    for (int b = 0; b < batchSize; b++)
        result[b] = new double[rows];

    // Transpose batch for better memory access
    double[][] transposedBatch = new double[cols][];
    for (int i = 0; i < cols; i++)
        transposedBatch[i] = new double[batchSize];

    for (int b = 0; b < batchSize; b++)
        for (int j = 0; j < cols; j++)
            transposedBatch[j][b] = batch[b][j];

    // Perform matrix multiplication
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double weight = matrix[i, j];
            double[] transposedCol = transposedBatch[j];

            for (int b = 0; b < batchSize; b++)
                result[b][i] += weight * transposedCol[b];
        }
    }

    return result;
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

static double[] Tanh(double[] input)
{
    int len = input.Length;
    double[] result = new double[len];

    int simdLength = Vector<double>.Count;
    int i = 0;

    // SIMD portion
    for (; i <= len - simdLength; i += simdLength)
    {
        var v = new Vector<double>(input, i);

        // Approximate tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        Vector<double> two = new Vector<double>(2.0);
        Vector<double> v2x = v * two;

        double[] exp2x = new double[simdLength];
        for (int j = 0; j < simdLength; j++)
            exp2x[j] = ExpLookup.FastExp(v2x[j]);

        for (int j = 0; j < simdLength; j++)
            result[i + j] = (exp2x[j] - 1) / (exp2x[j] + 1);
    }

    // Scalar tail
    for (; i < len; i++)
    {
        double x2 = 2 * input[i];
        double exp2x = ExpLookup.FastExp(x2);
        result[i] = (exp2x - 1) / (exp2x + 1);
    }

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
static class ExpLookup
{
    static readonly double[] expTable;
    const double minX = -50.0;
    const double maxX = 50.0;
    const int resolution = 10000;
    const double step = (maxX - minX) / resolution;

    static ExpLookup()
    {
        expTable = new double[resolution + 1];
        for (int i = 0; i <= resolution; i++)
        {
            double x = minX + i * step;
            expTable[i] = Math.Exp(x);
        }
    }

    public static double FastExp(double x)
    {
        if (x < minX) return 0;
        if (x > maxX) return Math.Exp(x);
        int index = (int)((x - minX) / step);
        return expTable[index];
    }
}

static double[] Softmax(double[] logits)
{
    double maxLogit = logits.Max();
    double[] exps = new double[logits.Length];
    double sum = 0;

    for (int i = 0; i < logits.Length; i++)
    {
        exps[i] = ExpLookup.FastExp(logits[i] - maxLogit);
        sum += exps[i];
    }

    for (int i = 0; i < logits.Length; i++)
        exps[i] /= sum;

    return exps;
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
    int rows = gradients.GetLength(0);
    int cols = gradients.GetLength(1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double val = gradients[i, j];
            if (Math.Abs(val) > clipValue)
                gradients[i, j] = Math.Max(Math.Min(val, clipValue), -clipValue);
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