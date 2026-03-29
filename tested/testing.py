from rag.pipeline import RAGPipeline

rag = RAGPipeline()
session_id = "user_1"

print("RAG Chatbot ready. Type 'quit' to exit.\n")

while True:
    query = input("Ask: ").strip()

    if not query:
        continue

    if query.lower() in ("quit", "exit"):
        print("Exiting.")
        break

    result = rag.run(query, session_id=session_id)

    print("\n" + "─" * 60)
    print(f"[Rewritten Query]  {result['rewritten_query']}")
    print(f"[Was Rewritten]    {result['was_rewritten']}")
    print(f"[Retrieval OK]     {result['retrieval_ok']}")
    print(f"[Fallback Used]    {result['fallback_used']}")
    print(f"\n[Context]\n{result['context']}")
    print(f"\n[Answer]\n{result['answer']}")
    print("─" * 60 + "\n")