using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.Qdrant;
using OllamaSharp;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using System.Text;

namespace MilestoneSearch.RAG;

internal class Program
{
    static async Task Main(string[] args)
    {
        var ollamaEndpoint = new Uri("http://localhost:11434");
        var qdrantEndpoint = new Uri("http://localhost:6334");

        const string chatModelId = "phi3:mini";
        const string embeddingModel = "nomic-embed-text";
        const string collectionName = "milestone_topics";
        const int searchLimit = 5;

        IChatClient ollamaChatClient = new OllamaChatClient(ollamaEndpoint, chatModelId);

        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator = 
            new OllamaEmbeddingGenerator(ollamaEndpoint, embeddingModel);

        var qdrantClient = new QdrantClient(qdrantEndpoint);

        var vectorStore = new QdrantVectorStore(qdrantClient, true, new QdrantVectorStoreOptions 
        { 
            EmbeddingGenerator = embeddingGenerator
        });


        var collections = await qdrantClient.ListCollectionsAsync();
        var exists = collections.Contains(collectionName);

        if (!exists)
        {
            Console.WriteLine("Creating Qdrant collection...");

            await qdrantClient.CreateCollectionAsync(collectionName, new VectorParams 
            { 
                Size = 768,
                Distance = Distance.Cosine
            });

            Console.WriteLine("Collection created.");
        }

        var topics = vectorStore.GetCollection<Guid, Topic>(collectionName);

        var ingestion = new PdfIngestionService(embeddingGenerator);
                
        var pdfFolder = @"C:\Users\Raja Anand\Downloads\milestone-pdfs";

        foreach (var pdf in Directory.GetFiles(pdfFolder, "*.pdf"))
        {
            var fileName = Path.GetFileName(pdf);

            if (await PdfAlreadyIngestedAsync(topics, fileName))
            {
                Console.WriteLine($"Skipping {fileName} (already ingested)");
                continue;
            }

            Console.WriteLine($"Processing conversion for :: {pdf}...");

            var topicsToInsert = await ingestion.ProcessPdfAsync(pdf);

            foreach (var topic in topicsToInsert)
            {
                await topics.UpsertAsync(topic);
            }

            Console.WriteLine("New content added for vector search.");
        }

        
        

        Console.WriteLine("Milestone RAG Ready! Ask questions or type 'quit' to exit.");

        var memory = new ChatMemory();

        while (true)
        {
            Console.Write("\nYour question: ");
            var query = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(query))
                continue;

            if (query.ToLower() == "quit")
            {
                Console.WriteLine("Goodbye!");
                break;
            }
                     

            
            var embeddingResult = await embeddingGenerator.GenerateAsync(query);
            var queryEmbedding = embeddingResult.Vector;


            var results = topics.SearchAsync(queryEmbedding, searchLimit,
                new VectorSearchOptions<Topic>
                {
                    VectorProperty = t => t.ContentEmbedding
                });

            //remove this after testing --------------------

            Console.WriteLine("\nRetrieved context:");
            await foreach (var result in results)
            {
                var topic = result.Record;
                Console.WriteLine($"- {topic.Title} (score: {result.Score})");
            }
            Console.WriteLine();


            var contextBuilder = new StringBuilder();
            var references = new List<string>();

            await foreach (var result in results)
            {
                if (result.Score < 0.50)
                    continue;

                var topic = result.Record;
                var score = result.Score ?? 0;
                var percent = (score * 100).ToString("F2");

                contextBuilder.AppendLine($"[{topic.Title}] {topic.Content}");
                references.Add($"[{percent}%] {topic.Source}");
            }

            var context = contextBuilder.ToString();
            var previousMessages = string.Join(Environment.NewLine, memory.GetMessages());

            var prompt = $"""
                You are a helpful assistant specialized in Milestone Systems documentation.

                Context:
                {context}

                Previous conversation:
                {previousMessages}

                Rules:
                - Use the context to answer.
                - If the user refers to earlier conversation, use memory.
                - If you don't know, say you don't know.

                User question: {query}

                Answer:
                """;

            memory.AddMessage(query.Trim());

            // Chat completion
            var responseText = new StringBuilder();

            await foreach(var update in ollamaChatClient.GetStreamingResponseAsync(
                new ChatMessage(ChatRole.User, prompt)))
            {
                if (!string.IsNullOrEmpty(update.Text))
                {
                    Console.Write(update.Text);
                    responseText.Append(update.Text);
                }
            }


            memory.AddMessage(responseText.ToString().Trim());


            if (references.Count > 0)
            {
                Console.WriteLine("\n\nReferences used:");
                foreach (var reference in references)
                {
                    Console.WriteLine($"- {reference}");
                }
            }

            Console.WriteLine();
        }
    }


    static async Task<bool> PdfAlreadyIngestedAsync(
        VectorStoreCollection<Guid, Topic> topics, string fileName)
    {
        var results = topics.SearchAsync(
            new ReadOnlyMemory<float>(new float[768]),
            1,
            new VectorSearchOptions<Topic>
            {
                Filter = f => f.FileName == fileName
            });

        await foreach (var _ in results)
            return true;

        return false;
    }

}