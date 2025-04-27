namespace Tokenizer 
{
    


using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

class Tokenizer
{
    static void Main(string[] args)
    {
        string inputPath = "_CLEAN_DATA/movie_lines_clean.txt";
        string vocabPath = "_VOCAB/vocab.txt";
        string tokenizedPath = "_CLEAN_DATA/tokenized_movie_lines_clean.txt";
        int totalLines = File.ReadLines(inputPath).Count();
        int lineCounter = 0;
        int tokenCounter = 0;

        // Holds word → ID mapping
        Dictionary<string, int> vocab = new Dictionary<string, int>();

        // Holds processed lines
        List<string> tokenizedLines = new List<string>();

        int currentId = 1;

        foreach (var line in File.ReadLines(inputPath))
        {
            var parts = line.Split("|||");
            if (parts.Length != 2) continue;

            string input = parts[0].ToLower().Trim();
            string response = parts[1].ToLower().Trim();

            List<int> inputTokens = Tokenize(input, ref vocab, ref currentId);
            List<int> responseTokens = Tokenize(response, ref vocab, ref currentId);

            tokenizedLines.Add($"{string.Join(" ", inputTokens)}|||{string.Join(" ", responseTokens)}");
            lineCounter++;
            tokenCounter += inputTokens.Count + responseTokens.Count;

            // Update console progress (overwrite same line)
            double percent = (lineCounter / (double)totalLines) * 100;
            Console.Write($"\r🔁 Tokenizing: {lineCounter}/{totalLines} lines | {tokenCounter} tokens | {percent:F2}%");

        }

        // Save vocab
        File.WriteAllLines(vocabPath, vocab.Select(kvp => $"{kvp.Key} {kvp.Value}"));

        // Save tokenized data
        File.WriteAllLines(tokenizedPath, tokenizedLines);

        Console.WriteLine("✅ Tokenization complete!");
        Console.WriteLine($"Saved vocab: {vocab.Count} words");
        Console.WriteLine($"Tokenized lines: {tokenizedLines.Count}");
    }

    // Tokenize a sentence, add to vocab if needed
    static List<int> Tokenize(string sentence, ref Dictionary<string, int> vocab, ref int currentId)
{
    // Normalize: remove punctuation, split dashes
    var cleaned = sentence
        .Replace("--", " ")     // Split stuck words
        .Replace(",", "")
        .Replace(".", "")
        .Replace("?", "")
        .Replace("!", "")
        .Replace("\"", "")
        .Replace("*", "")       // Remove asterisk formatting
        .ToLower();

    var rawTokens = cleaned.Split(" ", StringSplitOptions.RemoveEmptyEntries);

    List<int> tokenIds = new List<int>();

    foreach (var token in rawTokens)
    {
        // Only keep words made of letters (e.g., skip 123, *lost*, said--i, etc.)
        if (!token.All(char.IsLetter)) continue;

        if (!vocab.ContainsKey(token))
        {
            vocab[token] = currentId++;
        }

        tokenIds.Add(vocab[token]);
    }

    return tokenIds;
}

}
}
