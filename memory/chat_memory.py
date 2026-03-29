from typing import List, Tuple


class ChatMemory:
    def __init__(self):
        self.history: List[Tuple[str, str]] = []

    def add_user_message(self, message: str):
        self.history.append(("user", message))

    def add_ai_message(self, message: str):
        self.history.append(("ai", message))

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []