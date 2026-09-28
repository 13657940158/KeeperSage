import os
#单例模式，不用反复创建对象，如果对象为空，则创建一个对象。4个utils都是一样的效果
from pymilvus import MilvusClient

from config.milvus_config import milvus_config

_milvus_client = None
def get_milvus_client():
    """
    获取全局单例的MilvusClient对象
    :return: MilvusClient实例
    """
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(uri=milvus_config.milvus_url)

    return _milvus_client