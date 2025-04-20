using System;
using TorchSharp;
using static TorchSharp.torch;

class CudaTest
{
    static void Main()
    {
        try
        {
            var device = new Device("cuda");
            var tensor = randn(new long[] { 2, 2 }, device: device);

            Console.WriteLine("🚀 CUDA is available!");
            Console.WriteLine("Tensor:");
            Console.WriteLine(tensor);
        }
        catch (Exception ex)
        {
            Console.WriteLine("❌ CUDA is NOT available.");
            Console.WriteLine($"Reason: {ex.Message}");
        }
    }
}
