from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()


class HistoryMongoTool:
    def __init__(self):
        self._mongo_url = os.getenv("MONGO_URL")
        self._db_name = os.getenv("MONGO_DB_NAME")

        # 链接
        self._client = MongoClient(self._mongo_url)

        # 索引(库)
        self.db = self._client[self._db_name]

        # 集合(表)
        self.chat_message = self.db["chat_message"]
        self.chat_message.create_index([("session_id", 1), ("ts", -1)])


_history_mongo_tool = HistoryMongoTool()


def get_history_mongo_tool():
    global _history_mongo_tool
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    return _history_mongo_tool
