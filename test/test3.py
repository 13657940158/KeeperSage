from utils.mongo_test_utils import get_history_mongo_tool

# 通过工具获得mongo的客户端
mongo_client = get_history_mongo_tool()

# 向chat_message集合插入一个doc
mongo_client.chat_message.insert_one({"session_id": 1, "message": "hello", "ts": 1234567890})
