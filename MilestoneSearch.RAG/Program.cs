using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.Qdrant;
using OllamaSharp;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using System.Text;
using System.Diagnostics;
using System.Threading;

namespace MilestoneSearch.RAG;

internal class Program
{
    static async Task Main(string[] args)
    {
        var ollamaEndpoint = new Uri("http://localhost:11434");
        var qdrantEndpoint = new Uri("http://localhost:6334");

        const string chatModelId = "phi3:mini"; // lower-latency model
        // const string chatModelId = "gemma2:2b";        
        const string embeddingModel = "nomic-embed-text";
        const string collectionName = "milestone_topics";
        const int searchLimit = 5;

        IChatClient ollamaChatClient = new OllamaChatClient(ollamaEndpoint, chatModelId);

        // after creating ollamaChatClient, warm the model once (fire-and-forget small prompt)
        _ = Task.Run(async () =>
        {
            try
            {
                using var ctsWarm = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                var warmStream = ollamaChatClient.GetStreamingResponseAsync(new ChatMessage(ChatRole.User, "Hi"));
                var enumerator = warmStream.GetAsyncEnumerator(ctsWarm.Token);
                if (await enumerator.MoveNextAsync()) { /* consume first token */ }
                await enumerator.DisposeAsync();
                //Console.WriteLine("[info] Ollama model warmed.");
            }
            catch { /* ignore warm failures */ }
        });

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
                     

            var swOverall = Stopwatch.StartNew();

            var sw = Stopwatch.StartNew();
            var embeddingResult = await embeddingGenerator.GenerateAsync(query);
            sw.Stop();
            
            Console.WriteLine($"\n[timing] Embedding generation: {sw.ElapsedMilliseconds} ms");

            var queryEmbedding = embeddingResult.Vector;


            var results = topics.SearchAsync(queryEmbedding, searchLimit,
                new VectorSearchOptions<Topic>
                {
                    VectorProperty = t => t.ContentEmbedding
                });

            // Materialize results once so we don't re-run the search twice
            var matches = new List<VectorSearchResult<Topic>>();
            sw.Restart();
            await foreach (var r in results)
            {
                matches.Add(r);
            }
            sw.Stop();
            Console.WriteLine($"[timing] Vector search: {sw.ElapsedMilliseconds} ms (results: {matches.Count})");

            Console.WriteLine();

            var contextBuilder = new StringBuilder();
            var references = new List<string>();

            foreach (var result in matches)
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

            // estimate tokens and truncate context before building final prompt
            int maxContextChars = 700; // tune this
            string contextTruncated = context.Length > maxContextChars
                ? context.Substring(0, maxContextChars) + " ..."
                : context;

            // build bounded previous messages (keep last 5 messages, then tail to maxPrevChars)
            var prevList = memory.GetMessages().ToList();
            var lastN = string.Join(Environment.NewLine, prevList.Skip(Math.Max(0, prevList.Count - 5)));
            int maxPrevChars = 2500;
            string previousMessagesTruncated = lastN.Length > maxPrevChars
                ? lastN.Substring(lastN.Length - maxPrevChars) // keep tail (most recent context)
                : lastN;

            // compose prompt using contextTruncated and previousMessagesTruncated
            var prompt = $"""
                You are a helpful assistant specialized in Milestone Systems documentation.

                Context:
                {contextTruncated}

                Previous conversation:
                {previousMessagesTruncated}

                Rules:
                - Use the context to answer.
                - If the user refers to earlier conversation, use memory.
                - If you don't know, say you don't know.
                - Keep your responses short and sharp avoid long messages.

                User question: {query}

                Answer:
                """;

            // diagnostics: log prompt size and rough token estimate
            Console.WriteLine($"[debug] Prompt chars: {prompt.Length}, est tokens: {Math.Max(1, prompt.Length / 4)}\n");

            var responseText = new StringBuilder();

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(250));
            var streaming = ollamaChatClient.GetStreamingResponseAsync(new ChatMessage(ChatRole.User, prompt));

            var enumerator = streaming.GetAsyncEnumerator(cts.Token);
            var chatSw = Stopwatch.StartNew();
            try
            {
                while (await enumerator.MoveNextAsync())
                {
                    var update = enumerator.Current;
                    if (!string.IsNullOrEmpty(update.Text))
                    {
                        Console.Write(update.Text);
                        responseText.Append(update.Text);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("\n\n[Chat request timed out]");
                responseText.Append(" [Timed out]");
            }
            finally
            {
                await enumerator.DisposeAsync();
            }

            chatSw.Stop();
            Console.WriteLine($"\n\n[timing] Chat generation: {chatSw.ElapsedMilliseconds} ms");

            swOverall.Stop();
            Console.WriteLine($"[timing] Total query processing: {swOverall.ElapsedMilliseconds} ms");

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