import json
import logging
from typing import List, Dict, Any

from pymilvus import DataType, milvus_client
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.state import ImportGraphState


# 如果 get_milvus_client / MilvusError 在别的地方，改成你实际路径
from processor.import_processor.exceptions import ImportProcessError as MilvusError, StateFieldError
from utils.milvus_utils import get_milvus_client, escape_milvus_string


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """
    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        # 1 数据校验
        chunks_json_data, vector_dimension = self._step_1_check_inputs(state)

        # 2 准备客户端和集合名
        milvus_client = self._step_2_prepare_collection(vector_dimension)

        # 3 清理可能的冗余数据(幂等性)
        self._step_3_clean_old_data(milvus_client, chunks_json_data)

        # 4 数据入库,返回数据库主键(字段+索引)
        update_chunks = self._step_4_insert_data(milvus_client, chunks_json_data)
        for chunk in update_chunks:
            print(f"chunk_id:{chunk['chunk_id']}")
        # 5 更新状态
        state["chunks"] = update_chunks

        return state

    def _step_1_check_inputs(self, state):
        print("node_import_milvus: 步骤1 数据校验 开始了")
        #处理好的文本块
        chunks_json_data = state.get("chunks")
        if not chunks_json_data:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)
        #向量维度
        dimension=len(chunks_json_data[0].get("dense_vector"))
        print(f"向量维度:{dimension}")

        return chunks_json_data, dimension
    #客户端和集合准备
    def _step_2_prepare_collection(self, vector_dimension:int):
        print("node_import_milvus: 步骤2 结构准备 开始了")

        # 先拿 client
        milvus_client = get_milvus_client()
        collection_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collection_name):
            self.create_chunks_collection(collection_name, milvus_client, vector_dimension)

        return milvus_client


    def _step_3_clean_old_data(self,client, chunks_json_data):
        print("node_import_milvus: 清理旧数据开始了 开始了")
        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self.clear_chunks_by_file_title(client, file_title)



    def _step_4_insert_data(self,milvus_client, chunks_json_data:List[Dict[str,Any]])->List[Dict[str,Any]]:
        print("node_import_milvus: 步骤4 数据入库 开始了")
        #数据处理
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()

            # 补充 part 字段
            if "part" not in item_copy:
                item_copy["part"] = 0

            # 添加到待插入列表
            data_to_insert.append(item_copy)

        #批量插入
        insert_result =  milvus_client.insert(collection_name=milvus_config.chunks_collection, data=data_to_insert)
        insert_count = insert_result.get("insert_count")#插入数据的条数
        print(f"insert_count:{insert_count}")
        #回写主键
        insert_ids = insert_result.get("ids")
        if insert_ids:
            for idx, item in enumerate(chunks_json_data):
                item['chunk_id'] = str(insert_ids[idx])
        #返回带有主键的chunks
        return chunks_json_data

    def create_chunks_collection(self, collections_name, milvus_client, vector_dimension):
        print("node_import_milvus:  创建集合 开始了")
        # 1. 创建 schema
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 2. 创建列
        #字段申明
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)#1024维

        # 3. 创建索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"},
        )

        # 4. 创建集合
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params,
        )

    #清理旧数据
    def clear_chunks_by_file_title(self, client, file_title):
        print("node_import_milvus: 步骤3方法1，清理旧数据")
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")


if __name__ == "__main__":
    # setup_logging()

    json_path = r"D:\doc\output\H3C\H3C_new_3_chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)
    init_state = {"chunks": state}

    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))