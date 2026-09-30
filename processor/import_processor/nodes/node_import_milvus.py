import json
import logging

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """
    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        # 1 数据校验
        chunks_json_data, vector_dimension = self._step_1_check_inputs(state)

        # 2 结构准备(字段+索引)
        milvus_client = self._step_2_prepare_collection(vector_dimension)

        # 3 清理可能的冗余数据(幂等性)
        self._step_3_clean_old_data(milvus_client, chunks_json_data)

        # 4 数据入库,返回数据库主键
        update_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)

        # 5 更新状态
        state["chunks"] = update_chunks

        return state

    def _step_1_check_inputs(self, state):
        print("node_import_milvus: 步骤1 数据校验 开始了")
        # 模拟返回：数据列表 和 向量维度（比如1024）
        chunks_json_data = state.get("chunks", [])
        vector_dimension = 1024
        return chunks_json_data, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension):
        print("node_import_milvus: 步骤2 结构准备 开始了")
        # 模拟返回一个 Milvus 客户端对象（随便给个字符串代替）
        milvus_client = "mock_milvus_client"
        return milvus_client

    def _step_3_clean_old_data(self, milvus_client, chunks_json_data):
        print("node_import_milvus: 步骤3 清理冗余数据 开始了")
        # 这个方法在 process 里没有接收返回值，所以返回 None 即可
        return None

    def _step_4_insert_data(self, milvus_client, chunks_json_data):
        print("node_import_milvus: 步骤4 数据入库 开始了")
        # 模拟返回插入数据库后生成的主键列表
        update_chunks = ["mock_id_1", "mock_id_2"]
        return update_chunks

if __name__ == "__main__":
    setup_logging()

    json_path = r"D:\doc\output\H3C\H3C_new_3_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))