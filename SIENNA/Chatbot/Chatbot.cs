using System;
using System.IO;
using System.Linq;

class Chatbot
{
    static int vocabSize = 0;
    static int embeddingDim = 64;
    static int hiddenDim = 128;

    static double[,] inputEmbedding;
    static double[,] rnnWeights;
    static double[,] outputWeights;

    static Tokenizer tokenizer;
    
    static void Main()
    {
        tokenizer = new Tokenizer("_VOCAB/vocab.txt");
        vocabSize = tokenizer.WordToId.Count;

        Console.WriteLine("💾 Loading model weights...");
        inputEmbedding = LoadMatrix("_MODEL_WEIGHTS/inputEmbedding.txt");
        rnnWeights = LoadMatrix("_MODEL_WEIGHTS/rnnWeights.txt");
        outputWeights = LoadMatrix("_MODEL_WEIGHTS/outputWeights.txt");

        Console.WriteLine("🤖 Chatbot is ready. Type something!");

        while (true)
{
    Console.Write("\nYou: ");
    string userInput = Console.ReadLine()?.Trim();
    if (string.IsNullOrEmpty(userInput)) continue;

    int[] inputTokens = tokenizer.Encode(userInput);
    Console.WriteLine("🔢 Encoded tokens: " + string.Join(" ", inputTokens));

    List<int> generatedTokens = GenerateFullSentence(inputTokens);

    string reply = tokenizer.Decode(generatedTokens.ToArray());
    Console.WriteLine($"AI: {reply}");
}

    }

    static int PredictNextToken(int[] inputTokens)
{
    double[] hidden = new double[hiddenDim];
    foreach (int token in inputTokens)
    {
        double[] x = GetRow(inputEmbedding, token);
        double[] combined = Concatenate(x, hidden);
        hidden = Tanh(MatVecMul(rnnWeights, combined));
    }

    double[] logits = MatVecMul(outputWeights, hidden);

    int predicted = -1;
    int retryCount = 0;
    const int maxRetries = 5; // avoid infinite loops, just in case

    while ((predicted == -1 || predicted == 0) && retryCount < maxRetries)
    {
        double[] probs = Softmax(logits);

        // 🚨 Force UNK (id 0) probability to 0
        probs[0] = 0;

        predicted = SampleFromProbs(probs);


        if (predicted == 0)
        {
            Console.WriteLine("⚠️ Predicted UNK (id 0), retrying...");
            retryCount++;

            // Optional: Add a tiny random noise to logits to change prediction
            Random rand = new Random();
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] += (rand.NextDouble() - 0.5) * 0.01; // tiny noise
            }
        }
    }

    if (predicted == 0)
    {
        Console.WriteLine("❗Still UNK after retries. Forcing fallback token.");
        predicted = 1; // fallback: force token 1 (assuming it’s a valid token)
    }

    Console.WriteLine("🧠 Final probabilities (first 10): " + string.Join(", ", logits.Take(10)));
    Console.WriteLine("🎯 Final predicted token: " + predicted);

    return predicted;
}


static int SampleFromProbs(double[] probs)
{
    Random rand = new Random();
    double r = rand.NextDouble();
    double cumulative = 0.0;

    for (int i = 0; i < probs.Length; i++)
    {
        cumulative += probs[i];
        if (r < cumulative)
            return i;
    }

    // fallback (should never happen if probs sum to 1)
    return probs.Length - 1;
}

static List<int> GenerateFullSentence(int[] inputTokens)
{
    List<int> generatedTokens = new List<int>();
    int maxTokens = 20; // Max words in the reply
    int eosId = 3; // If you have an <eos> token, use its ID here. Otherwise ignore.

    int[] context = inputTokens;

    for (int i = 0; i < maxTokens; i++)
    {
        int predictedToken = PredictNextToken(context);

        // 🚨 Optional: if you have an End-Of-Sentence token, stop early
        if (predictedToken == eosId)
            break;

        generatedTokens.Add(predictedToken);

        // Context for next prediction: feed last token back
        context = new int[] { predictedToken };
    }

    return generatedTokens;
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

    static double[] GetRow(double[,] matrix, int row)
    {
        int cols = matrix.GetLength(1);
        double[] result = new double[cols];
        for (int i = 0; i < cols; i++)
            result[i] = matrix[row, i];
        return result;
    }

    static double[] Tanh(double[] vector)
    {
        double[] result = new double[vector.Length];
        for (int i = 0; i < vector.Length; i++)
            result[i] = Math.Tanh(vector[i]);
        return result;
    }

    static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        double[] exps = logits.Select(x => Math.Exp(x - max)).ToArray();
        double sum = exps.Sum();
        return exps.Select(x => x / sum).ToArray();
    }

    static int ArgMax(double[] array)
    {
        int best = 0;
        double max = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > max)
            {
                max = array[i];
                best = i;
            }
        }
        return best;
    }

    static double[,] LoadMatrix(string path)
    {
        var lines = File.ReadAllLines(path);
        int rows = lines.Length;
        int cols = lines[0].Split().Length;
        double[,] matrix = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            var parts = lines[i].Split().Select(double.Parse).ToArray();
            for (int j = 0; j < cols; j++)
                matrix[i, j] = parts[j];
        }

        return matrix;
    }
    static double[] Concatenate(double[] a, double[] b)
{
    double[] result = new double[a.Length + b.Length];
    a.CopyTo(result, 0);
    b.CopyTo(result, a.Length);
    return result;
}

}
