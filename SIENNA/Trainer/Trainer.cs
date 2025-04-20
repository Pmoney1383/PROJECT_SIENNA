using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

class Trainer
{
    static void Main(string[] args)
    {
         string inputPath = "_PREP_DATA/padded_inputs.txt";
        string outputPath = "_PREP_DATA/padded_outputs.txt";
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

        
int rawCount = 0;    // total lines attempted
int validCount = 0;  // successfully added to training data
int skipped = 0;
int tokenCount = 0;

var rawLines = File.ReadLines(tokenizedPath).Take(1000).ToList();
int totalLines = rawLines.Count;

foreach (var line in rawLines)
{
    rawCount++;

    var parts = line.Split("|||");
    if (parts.Length != 2) { skipped++; continue; }

    var inputTokens = parts[0].Trim()
        .Split(' ', StringSplitOptions.RemoveEmptyEntries)
        .Select(int.Parse)
        .ToList();

    var outputTokens = parts[1].Trim()
        .Split(' ', StringSplitOptions.RemoveEmptyEntries)
        .Select(int.Parse)
        .ToList();

    if (inputTokens.Count == 0 || outputTokens.Count == 0)
    {
        skipped++;
        continue;
    }

    inputs.Add(inputTokens);
    outputs.Add(outputTokens);

    validCount++;
    tokenCount += inputTokens.Count + outputTokens.Count;

    double percent = (rawCount / (double)totalLines) * 100;
    Console.Write($"\r📦 Preparing data: {rawCount}/{totalLines} lines | {tokenCount} tokens | {percent:F2}%");
}



        Console.WriteLine($"✅ Loaded {inputs.Count} pairs from tokenized data");

        // 3. Pad sequences to fixed length
        int maxInputLength = inputs.Max(seq => seq.Count);
        int maxOutputLength = outputs.Max(seq => seq.Count);

        var paddedInputs = PadSequences(inputs, maxInputLength);
        var paddedOutputs = PadSequences(outputs, maxOutputLength);
        
        string prepFolder = "_PREP_DATA";
        Directory.CreateDirectory(prepFolder); // create if not exists

        string inputFilePath = Path.Combine(prepFolder, "padded_inputs.txt");
        string outputFilePath = Path.Combine(prepFolder, "padded_outputs.txt");

        // Append if files already exist
        using (StreamWriter inputWriter = new StreamWriter(inputFilePath, append: true))
        using (StreamWriter outputWriter = new StreamWriter(outputFilePath, append: true))
        {
            for (int i = 0; i < paddedInputs.Count; i++)
            {
                string inputLine = string.Join(" ", paddedInputs[i]);
                string outputLine = string.Join(" ", paddedOutputs[i]);

                inputWriter.WriteLine(inputLine);
                outputWriter.WriteLine(outputLine);
            }
        }

        Console.WriteLine($"📏 Max input length: {maxInputLength}, Max output length: {maxOutputLength}");
        Console.WriteLine($"\n🚫 Skipped {skipped} malformed lines");
        // Pause here – model coming next!
        Console.WriteLine("🚧 Data prep complete – ready for model init!");
        CheckPaddedData.PrintSanityCheck(inputPath, outputPath);
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
