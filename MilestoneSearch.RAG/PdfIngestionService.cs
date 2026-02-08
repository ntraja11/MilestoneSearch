using Microsoft.Extensions.AI;
using UglyToad.PdfPig;

namespace MilestoneSearch.RAG;

public class PdfIngestionService
{
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;

    public PdfIngestionService(IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator)
    {
        _embeddingGenerator = embeddingGenerator;
    }

    public async Task<List<Topic>> ProcessPdfAsync(string pdfPath)
    {
        var topics = new List<Topic>();
        var fileName = Path.GetFileName(pdfPath);

        using var document = PdfDocument.Open(pdfPath);

        foreach (var page in document.GetPages())
        {
            var text = page.Text;

            if (string.IsNullOrWhiteSpace(text))
                continue;

            var chunks = ChunkText(text, maxChunkSize: 800);

            foreach (var chunk in chunks)
            {
                var embedding = await _embeddingGenerator.GenerateAsync(chunk);

                var topic = new Topic
                {
                    Title = $"Page {page.Number} - {fileName}",
                    Content = chunk,
                    Source = $"{fileName} (Page {page.Number})",
                    FileName = fileName,
                    ContentEmbedding = embedding.Vector
                };

                topics.Add(topic);
            }
        }

        return topics;
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