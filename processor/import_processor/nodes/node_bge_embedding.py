import json
from typing import List, Dict

from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        # 1 参数校验
        chunks = self._step_1_validate_paths(state)

        # 2 数据向量化
        output_data = self._step_generate_embeddings(chunks)
        for item in output_data:
            item_name=item.get("item_name")
            sparse_vector=item.get("sparse_vector")
            content=item.get("content")
            print(f"item_name:{item_name}")
            print(f"sparse_vector:{sparse_vector}")
        # 回填数据(item_name -> chunks)
        # path = r"D:\doc\output\H3C\H3C_new_3_chunks.json"
        # with open(path, "w", encoding="utf-8") as f:
        #     json.dump(
        #         output_data,
        #         f,
        #         ensure_ascii=False,
        #         indent=2
        #     )
        # 3 返回结果
        state["chunks"] = output_data

        return state

    # 步骤1
    def _step_1_validate_paths(self, state: ImportGraphState) -> List[Dict]:
        print("node_bge_embedding:步骤1:参数校验")
        # 校验参数
        chunks = state.get("chunks")
        if not chunks:
            raise ValueError("参数错误:chunks")
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)
        return chunks

    # 步骤2
    def _step_generate_embeddings(self, chunks: List[Dict[str,str]]) -> List[Dict[str,str]]:
        print("node_bge_embedding:步骤2:数据向量化")
        output_data = []
        batch_size = 5
        for i in range(0,len(chunks),batch_size):
            vector_texts = []
            batch_text = chunks[i:i+batch_size]#第一次从0块取到4块，一共取五块
            for doc in batch_text:
                item_name = doc["item_name"]
                content = doc["content"]
                vector_texts.append(f"{item_name}\n{content}" if item_name else content)

            embeddings=generate_embeddings(vector_texts)
            for j,doc in enumerate(batch_text):
                item =doc.copy()
                dense= embeddings["dense"][j]
                item["dense_vector"] = dense
                sparse = embeddings["sparse"][j]
                item["sparse_vector"] = sparse
                output_data.append(item)

        return output_data


if __name__ == "__main__":
    # 1. 实例化节点
    node = NodeBGEEmbedding()

    # 2. 读取并解析 JSON 文件
    file_path = "D:\\doc\\output\\H3C\\H3C_new_2_chunks.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks_content = f.read()
        json_state = json.loads(chunks_content)
    except FileNotFoundError:
        print(f"找不到文件，请检查路径: {file_path}")
        exit(1)

    # 3. 构造初始 state (根据你的 json 结构，确保 chunks 是列表)
    init_state = {
        "chunks": json_state
    }

    # 4. 调用 process 方法（而不是直接用类名构造！）
    print("\n--- 开始执行 process ---")
    try:
        response = node.process(init_state)

        # 5. 序列化输出（去掉 encoding，加上缩进方便看）
        # dumps = json.dumps(response, ensure_ascii=False, indent=4)
        # print("\n--- 执行成功，输出结果 ---")
        # 如果 _step_generate_embeddings 里还是 pass 并且返回空列表，这里会打印出 "chunks": []
        # 可以截取前 500 个字符看看结构
        # print(dumps)

    except Exception as e:
        print(f"\n执行失败，错误信息: {e}")
