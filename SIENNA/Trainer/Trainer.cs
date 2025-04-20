using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

class Trainer
{
    static void Main(string[] args)
    {
        string tokenizedPath = "_CLEAN_DATA/tokenized_movie_lines_clean.txt";
        string vocabPath = "_VOCAB/vocab.txt";

        // 1. Load vocabulary
        Dictionary<string, int> vocab = File.ReadAllLines(vocabPath)
            .Select(line => line.Split(' '))
            .ToDictionary(parts => parts[0], parts => int.Parse(parts[1]));

        int vocabSize = vocab.Count + 1; // +1 for padding (0)

        // 2. Load tokenized input/output pairs
        List<List<int>> inputs = new();
        List<List<int>> outputs = new();

        foreach (var line in File.ReadLines(tokenizedPath).Take(1000)) // Limit to 1000 for now
        {
            var parts = line.Split("|||");
            if (parts.Length != 2) continue;

            var inputTokens = parts[0].Trim().Split(' ').Select(int.Parse).ToList();
            var outputTokens = parts[1].Trim().Split(' ').Select(int.Parse).ToList();

            inputs.Add(inputTokens);
            outputs.Add(outputTokens);
        }

        Console.WriteLine($"✅ Loaded {inputs.Count} pairs from tokenized data");

        // 3. Pad sequences to fixed length
        int maxInputLength = inputs.Max(seq => seq.Count);
        int maxOutputLength = outputs.Max(seq => seq.Count);

        var paddedInputs = PadSequences(inputs, maxInputLength);
        var paddedOutputs = PadSequences(outputs, maxOutputLength);

        Console.WriteLine($"📏 Max input length: {maxInputLength}, Max output length: {maxOutputLength}");

        // Pause here – model coming next!
        Console.WriteLine("🚧 Data prep complete – ready for model init!");
    }

    static List<List<int>> PadSequences(List<List<int>> sequences, int maxLength)
    {
        return sequences.Select(seq =>
        {
            var padded = new List<int>(seq);
            while (padded.Count < maxLength)
                padded.Insert(0, 0); // pad with zeros at the beginning

            return padded;
        }).ToList();
    }
}
