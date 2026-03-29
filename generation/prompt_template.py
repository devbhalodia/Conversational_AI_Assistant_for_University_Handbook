'''
from langchain_core.prompts import ChatPromptTemplate


def get_chat_prompt():

    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a university assistant. Answer ONLY using the provided context.
            Do NOT infer or assume anything.
            If the context is a table, reproduce ALL rows exactly.
            If multiple tables exist, use only the most relevant one.
            If answer is not present, say:
            "I don't have complete information about this." 
            """
        ),
        (
            "human",
            """
            Context: 
            {context}

            Question:
            {user_query}

            Answer strictly based on the context."""
        )
    ])
'''

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_chat_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a university assistant. Answer ONLY using the provided context.

            Do NOT infer or assume anything.

            If the context is a table, reproduce ALL rows exactly.

            If answer is not present, say:
            "I don't have complete information about this."
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {user_query}
            """
        )
    ])