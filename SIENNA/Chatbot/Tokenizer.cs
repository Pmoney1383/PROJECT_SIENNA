using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class Tokenizer
{
    public Dictionary<string, int> WordToId = new();
    public Dictionary<int, string> IdToWord = new();

    public Tokenizer(string vocabPath)
    {
        var lines = File.ReadAllLines(vocabPath);
        foreach (var line in lines)
        {
            var parts = line.Trim().Split(' ', 2); // split into [word, id]
            if (parts.Length == 2 && int.TryParse(parts[1], out int id))
            {
                WordToId[parts[0]] = id;
                IdToWord[id] = parts[0];
            }
        }

        if (!WordToId.ContainsKey("<unk>"))
        {
            Console.WriteLine("⚠️ Warning: <unk> token not found in vocab. Unknown words will default to ID 0.");
            WordToId["<unk>"] = 0;
            IdToWord[0] = "<unk>";
        }
    }

    public int[] Encode(string sentence)
    {
        return sentence.ToLower().Split().Select(word =>
            WordToId.ContainsKey(word) ? WordToId[word] : WordToId["<unk>"]
        ).ToArray();
    }

    public string Decode(int[] tokens)
    {
        return string.Join(" ", tokens.Select(t =>
            IdToWord.ContainsKey(t) ? IdToWord[t] : "<unk>"
        ));
    }
}
