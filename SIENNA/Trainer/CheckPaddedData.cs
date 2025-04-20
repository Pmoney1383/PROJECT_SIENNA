using System;
using System.IO;

public static class CheckPaddedData
{
    public static void PrintSanityCheck(string inputFilePath, string outputFilePath)
    {
        if (!File.Exists(inputFilePath))
        {
            Console.WriteLine($"❌ Input file not found: {inputFilePath}");
            return;
        }

        if (!File.Exists(outputFilePath))
        {
            Console.WriteLine($"❌ Output file not found: {outputFilePath}");
            return;
        }

        string[] inputLines = File.ReadAllLines(inputFilePath);
        string[] outputLines = File.ReadAllLines(outputFilePath);

        int totalLines = Math.Min(inputLines.Length, outputLines.Length);
        bool testPassed = true;

        Console.WriteLine($"🧠 Total Lines Compared: {totalLines}\n");

        for (int i = 0; i < totalLines; i++)
        {
            string input = inputLines[i].Trim();
            string output = outputLines[i].Trim();

            

            if (string.IsNullOrWhiteSpace(input) || string.IsNullOrWhiteSpace(output))
            {
                Console.WriteLine("⚠️ MISSING DATA on this line!");
                testPassed = false;
            }

           
        }

        if (inputLines.Length != outputLines.Length)
        {
            Console.WriteLine("⚠️ Input and Output files have DIFFERENT number of lines!");
            testPassed = false;
        }

        Console.WriteLine(testPassed ? "\n✅ SANITY CHECK PASSED!" : "\n❌ SANITY CHECK FAILED!");
    }
}
