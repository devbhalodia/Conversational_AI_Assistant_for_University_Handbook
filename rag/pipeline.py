import logging
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm import get_llm
from generation.prompt_template import get_chat_prompt
from generation.output_parser import get_output_parser
from rag.query_rewriter import QueryRewriter
from memory.session_manager import SessionManager

logger = logging.getLogger(__name__)

MIN_DOCS_THRESHOLD = 1

RELEVANCE_CHECK_PROMPT = """
You are a relevance checker for a RAG system.

Given a search query and a set of retrieved document chunks, determine whether 
the chunks contain enough information to meaningfully answer the query.

Respond with ONLY one word: YES or NO.
YES = chunks are relevant and sufficient to answer the query.
NO = chunks are irrelevant, off-topic, or clearly insufficient.

Query: {query}

Retrieved chunks:
{context}

Are the chunks relevant?
"""


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.llm = get_llm()
        self.prompt = get_chat_prompt()
        self.parser = get_output_parser()
        self.rewriter = QueryRewriter()
        self.session_manager = SessionManager()

        self.chain = self.prompt | self.llm | self.parser

    # ── private helpers ──────────────────────────────────────────────────────

    def _format_context(self, docs) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    def _format_history(self, history: list[tuple[str, str]]) -> list[dict]:
        """
        Converts (role, msg) tuples to LLM-friendly dicts.
        Normalizes both 'human' and 'user' → 'user' for the LLM prompt.
        Normalizes both 'assistant' and 'ai' → 'assistant'.
        """
        role_map = {
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "ai": "assistant"
        }
        return [
            {"role": role_map.get(role, "user"), "content": msg}
            for role, msg in history
        ]

    def _is_retrieval_relevant(self, query: str, context: str) -> bool:
        """
        Lightweight relevance check — asks the LLM if the retrieved chunks
        actually address the query before proceeding to answer generation.
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(RELEVANCE_CHECK_PROMPT)
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({"query": query, "context": context})
        return result.strip().upper().startswith("YES")

    def _try_fallback_retrieval(self, query: str) -> tuple[list, str]:
        """
        Fallback strategy when primary retrieval fails relevance check:
        1. Broaden the query by stripping trailing constraints (basic heuristic)
        2. Retry retrieval once

        Returns (docs, context) — empty if fallback also fails.
        """
        # simple broadening: take first 6 words of the query
        # replace this with a smarter LLM-based broadening if needed
        broadened = " ".join(query.split()[:6])
        logger.info(f"Fallback retrieval with broadened query: '{broadened}'")

        docs = self.retriever.retrieve(broadened)
        context = self._format_context(docs)
        return docs, context

    # ── public interface ─────────────────────────────────────────────────────

    def run(self, query: str, session_id: str = "default") -> dict:
        """
        Full RAG pipeline run for a single user turn.

        Args:
            query:      Raw user message.
            session_id: Identifies the conversation session.

        Returns:
            dict with keys:
                answer          - LLM generated response
                context         - chunks used for generation
                rewritten_query - query used for retrieval
                was_rewritten   - whether rewriting was triggered
                retrieval_ok    - whether retrieval passed relevance check
                fallback_used   - whether fallback retrieval was triggered
        """
        memory = self.session_manager.get_memory(session_id)
        chat_history = memory.get_history()

        # ── step 1: query rewriting ──────────────────────────────────────────
        rewritten_query, was_rewritten = self.rewriter.rewrite(query, chat_history)

        if was_rewritten:
            logger.info(f"Query rewritten: '{query}' → '{rewritten_query}'")
        else:
            logger.info(f"Query used as-is: '{query}'")

        # ── step 2: retrieval ────────────────────────────────────────────────
        retrieval_ok = False
        fallback_used = False

        try:
            docs = self.retriever.retrieve(rewritten_query)

            if len(docs) < MIN_DOCS_THRESHOLD:
                logger.warning(f"Retrieval returned no docs for: '{rewritten_query}'")
                docs, context = self._try_fallback_retrieval(rewritten_query)
                fallback_used = True
            else:
                docs = self.reranker.rerank(rewritten_query, docs)
            
            context = self._format_context(docs)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            docs, context = [], ""

        # ── step 3: relevance check ──────────────────────────────────────────
        if context:
            retrieval_ok = self._is_retrieval_relevant(rewritten_query, context)

            if not retrieval_ok and not fallback_used:
                logger.warning("Relevance check failed — attempting fallback retrieval")
                docs, context = self._try_fallback_retrieval(rewritten_query)
                fallback_used = True
                
                #re-ranking
                if docs:
                    docs = self.reranker.rerank(rewritten_query, docs)
                    context = self._format_context(docs)
                # re-check relevance on fallback result
                if context:
                    retrieval_ok = self._is_retrieval_relevant(rewritten_query, context)

        # ── step 4: answer generation ────────────────────────────────────────
        # NOTE: original `query` (not rewritten) is passed here intentionally.
        # The rewritten query is only for retrieval — the LLM should respond
        # to what the user actually said, using chat history for conversational tone.
        try:
            formatted_history = self._format_history(chat_history)

            response = self.chain.invoke({
                "chat_history": formatted_history,
                "context": context,
                "user_query": query      # ← original query, intentional
            })

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = "I encountered an error generating a response. Please try again."

        # ── step 5: update memory (only after successful generation) ─────────
        memory.add_user_message(query)
        memory.add_ai_message(response)

        # ── step 6: return full diagnostic payload ───────────────────────────
        return {
            "answer": response,
            "context": context,
            "rewritten_query": rewritten_query,
            "was_rewritten": was_rewritten,
            "retrieval_ok": retrieval_ok,
            "fallback_used": fallback_used
        }