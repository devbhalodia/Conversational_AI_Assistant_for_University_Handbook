from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from generation.llm import get_llm


SLIDING_WINDOW = 4

CLASSIFIER_SYSTEM = """
You are a query classifier for a RAG chatbot.
Determine if the user's latest message is self-contained — meaning it can be 
understood and searched without any prior conversation context.

A query is NOT self-contained if ONLY it contains:
- Pronouns referring to prior context (it, that, this, they, its, those)
- Continuation phrases (what about, and for, how about, same for, now filter)
- Implicit topic carry-over with no explicit subject stated

Examples of self-contained queries (answer YES):
- "who is the major advisor of btech?"
- "what is the hostel fee for first year students?"
- "how do I apply for admission?"
- "what are the graduation requirements?"

Examples of NOT self-contained queries (answer NO):
- "what about its fees?" (its = unclear)
- "and for masters?" (continuation of prior topic)
- "how do I contact them?" (them = unclear)
- "what is the deadline for that?" (that = unclear)

Respond with ONLY one word: YES or NO.
YES = already self-contained, no rewriting needed.
NO = depends on prior context, rewriting needed.
"""

CLASSIFIER_HUMAN = """
Chat History (last {window} turns):
{chat_history}

Latest message: {query}

Is this message self-contained?
"""

REWRITER_SYSTEM = """
You are a search query rewriter for a RAG system.

Your job is to rewrite the user's latest message into a single, 
self-contained search query that captures the full retrieval intent.

Rules:
1. Resolve all pronouns and references (it, that, this, they) using chat history
2. Carry forward relevant entity names, topics, and constraints from prior turns
3. Handle these 3 cases explicitly:
   - Reference resolution: "What about its pricing?" → replace pronoun with entity
   - Topic continuation: "Give me more detail" → repeat prior topic with expansion signal
   - Constraint addition: "Now filter that for India" → prior query + new constraint
4. Do NOT bleed in context that is irrelevant to the current question
5. Output ONLY the rewritten query — no explanation, no preamble, nothing else
6. If somehow the message is already self-contained, return it unchanged
"""

REWRITER_HUMAN = """
Chat History (last {window} turns):
{chat_history}

Latest message: {query}

Rewritten query:
"""


class QueryRewriter:

    def __init__(self, sliding_window: int = SLIDING_WINDOW):
        self.llm = get_llm()
        self.parser = StrOutputParser()
        self.sliding_window = sliding_window

        classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFIER_SYSTEM),
            ("human", CLASSIFIER_HUMAN)
        ])
        self.classifier_chain = classifier_prompt | self.llm | self.parser

        rewriter_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM),
            ("human", REWRITER_HUMAN)
        ])
        self.rewriter_chain = rewriter_prompt | self.llm | self.parser

    def _get_windowed_history(self, chat_history: list[tuple[str, str]]) -> str:
        """
        Slice to last N turns and format as:
            Human: <msg>
            Assistant: <msg>
        """
        windowed = chat_history[-self.sliding_window:]
        return "\n".join(
            f"{'Human' if role == 'human' else 'Assistant'}: {msg}"
            for role, msg in windowed
        )

    def _is_self_contained(self, query: str, history_text: str) -> bool:
        """
        Classifier chain — returns True if query needs no rewriting.
        Defaults to True (i.e. skip rewrite) on any unexpected output.
        Only returns False if model explicitly responds with NO.
        """
        result = self.classifier_chain.invoke({
            "chat_history": history_text,
            "query": query,
            "window": self.sliding_window
        })
        return not result.strip().upper().startswith("NO")

    def _rewrite(self, query: str, history_text: str) -> str:
        """
        Rewriter chain — returns the standalone query.
        Falls back to original query if output is empty.
        """
        result = self.rewriter_chain.invoke({
            "chat_history": history_text,
            "query": query,
            "window": self.sliding_window
        })
        rewritten = result.strip()
        return rewritten if rewritten else query

    def rewrite(
        self,
        query: str,
        chat_history: list[tuple[str, str]]
    ) -> tuple[str, bool]:
        """
        Main entry point.

        Args:
            query:        The user's latest raw message.
            chat_history: List of (role, message) tuples. role = 'human' | 'assistant'

        Returns:
            (final_query, was_rewritten)
            - final_query:    The query to use for retrieval (rewritten or original)
            - was_rewritten:  Boolean flag — useful for logging and debugging
        """

        # no history → nothing to resolve → return as-is
        if not chat_history:
            return query, False

        history_text = self._get_windowed_history(chat_history)

        # gate 1 — is it already self-contained?
        if self._is_self_contained(query, history_text):
            return query, False

        # gate 2 — rewrite
        rewritten_query = self._rewrite(query, history_text)
        return rewritten_query, True