using Microsoft.Extensions.AI;
using UglyToad.PdfPig;
using System.Collections.Concurrent;

namespace MilestoneSearch.RAG;

public class PdfIngestionService
{
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private const int SemaphoreCount = 40;

    public PdfIngestionService(IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        _embeddingGenerator = embeddingGenerator;
    }

    public async Task<IList<Topic>> ProcessPdfAsync(string pdfPath)
    {
        var topics = new ConcurrentBag<Topic>();
        var fileName = Path.GetFileName(pdfPath);

        using var document = PdfDocument.Open(pdfPath);

        var semaphore = new SemaphoreSlim(SemaphoreCount);

        var allChunks = new List<(int Page, string Chunk)>();

        // First pass: collect all chunks
        foreach (var page in document.GetPages())
        {
            var text = page.Text;
            if (string.IsNullOrWhiteSpace(text))
                continue;

            var chunks = ChunkText(text, maxChunkSize: 800);

            foreach (var chunk in chunks)
                allChunks.Add((page.Number, chunk));
        }

        int total = allChunks.Count;
        int completed = 0;

        Console.WriteLine($"Total chunks to embed: {total}");

        // Second pass: embed in parallel
        var tasks = allChunks.Select(async item =>
        {
            await semaphore.WaitAsync();
            try
            {
                var embedding = await _embeddingGenerator.GenerateAsync(item.Chunk);

                topics.Add(new Topic
                {
                    Title = $"Page {item.Page} - {fileName}",
                    Content = item.Chunk,
                    Source = $"{fileName} (Page {item.Page})",
                    FileName = fileName,
                    ContentEmbedding = embedding.Vector
                });

                // Thread-safe increment
                int done = Interlocked.Increment(ref completed);

                double percent = (double)done / total * 100;

                char[] spinner = { '|', '/', '-', '\\' };
                int spinIndex = done % spinner.Length;

                Console.Write($"\r{spinner[spinIndex]} Progress: {done}/{total} ({percent:F1}%)");
            }
            finally
            {
                semaphore.Release();
            }
        });

        await Task.WhenAll(tasks);

        Console.WriteLine("Embedding complete.");

        return topics.ToList();
    }

    private static List<string> ChunkText(string text, int maxChunkSize)
    {
        var chunks = new List<string>();
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        var current = new List<string>();

        foreach (var word in words)
        {
            current.Add(word);

            if (current.Count >= maxChunkSize)
            {
                chunks.Add(string.Join(" ", current));
                current.Clear();
            }
        }

        if (current.Count > 0)
            chunks.Add(string.Join(" ", current));

        return chunks;
    }
}