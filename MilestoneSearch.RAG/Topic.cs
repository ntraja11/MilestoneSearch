using Microsoft.Extensions.VectorData;

namespace MilestoneSearch.RAG;

public class Topic
{
    [VectorStoreKey]
    public Guid Key { get; set; } = Guid.NewGuid();

    [VectorStoreData]
    public string Title { get; set; } = string.Empty;

    [VectorStoreData]
    public string Content { get; set; } = string.Empty;

    [VectorStoreData]
    public string Source { get; set; } = string.Empty;

    [VectorStoreData]
    public string FileName { get; set; } = string.Empty;

    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> ContentEmbedding { get; set; } = Array.Empty<float>();

}
