from memory.chat_memory import ChatMemory


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_memory(self, session_id: str) -> ChatMemory:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMemory()
        return self.sessions[session_id]