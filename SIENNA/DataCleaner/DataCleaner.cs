using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

class DataCleaner
{
    static void Main(string[] args)
    {
        string inputPath = "_DATA/movie_lines.txt";
        string outputPath = "_CLEAN_DATA/movie_lines_clean.txt";

        // List to store (lineID, utterance)
        List<(int LineID, string Utterance)> lines = new List<(int, string)>();

        // Load and extract utterances
        // Step 1: Count total lines
int totalLines = File.ReadLines(inputPath).Count();
int lineCounter = 0;

File.Create(outputPath).Close();

// Step 2: Load and extract utterances
foreach (var line in File.ReadLines(inputPath))
{
    lineCounter++;

    var parts = line.Split(new string[] { "+++$+++" }, StringSplitOptions.None);
    if (parts.Length >= 5)
    {
        string idStr = parts[0].Replace("L", "");
        if (int.TryParse(idStr, out int lineId))
        {
            string utterance = parts[4].Trim();
            if (!string.IsNullOrEmpty(utterance))
            {
                lines.Add((lineId, utterance));
            }
        }
    }

    // Print progress every 1000 lines
    if (lineCounter % 1000 == 0 || lineCounter == totalLines)
    {
        double percent = (lineCounter / (double)totalLines) * 100;
        Console.WriteLine($"Reading... {lineCounter}/{totalLines} lines ({percent:F2}%)");
    }
}

        // Sort by LineID
        lines = lines.OrderBy(l => l.LineID).ToList();

        // Create input-response pairs
        List<string> outputLines = new List<string>();
        for (int i = 0; i < lines.Count - 1; i++)
        {
            string input = lines[i].Utterance;
            string response = lines[i + 1].Utterance;
            outputLines.Add($"{input}|||{response}");
        }

        // Write to file
        File.WriteAllLines(outputPath, outputLines);

        Console.WriteLine("Cleaned data written to: " + outputPath);
    }
}
