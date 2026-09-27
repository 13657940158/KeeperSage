import json
from typing import  Dict, List

from click import prompt
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai.types.beta.threads import message

from config.lm_config import lm_config
from processor.import_processor.base import BaseNode
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        """
              LangGraph 核心节点：商品主体名称识别
              流程总览：
                  1. 提取输入
                  2. 构建大模型上下文
                  3. 调用大模型识别商品名称
                  4. 回填商品名称到状态和切片
                  5. 生成商品名称的稠密/稀疏向量
                  6. 将数据存入Milvus向量数据库

              必要参数：task_id、file_title, chunks
              更新参数：item_name

              :param state: 工作流状态对象
              :return: 更新后的状态对象
              """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别商品名称
        item_name = self._step_3_call_llm(file_title, context)
        print(f"itemname:{item_name}")
        # 步骤4：回填商品名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤5：为商品名称生成稠密/稀疏向量
        dense_vector, sparse_vector = self._step_5_generate_vectors(item_name)
        print(f"dense_vector:{dense_vector}")
        print(f"sparse_vector:{sparse_vector}")
        # 步骤6：将数据存入Milvus向量数据库
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector, sparse_vector)

        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} ---")

        return state

    def _step_1_get_inputs(self, state) ->(str,List[Dict]) :
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        print("node_item_name_recognition:步骤2：上下文拼接")
        #限制上下文的片数
        k = self.config.item_name_chunk_k
        #限制上下文的长度
        chunk_size = self.config.item_name_chunk_size
        parts: List[Dict] = []
        total_chars = 0
        for index,chunk in  enumerate(chunks[:k],start=1):
            chunk_title = chunk.get("title","").strip()
            chunk_content = chunk.get("content","").strip()
            piece = f"【切片{index}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)
            #计算长度
            total_chars += len(piece)
            if total_chars >= chunk_size:
                break
        #截断处理
        context = "\n\n".join(parts).strip()#让每一段中间有空行
        final_context = context[:chunk_size]
        return final_context

    def _step_3_call_llm(self,file_title,context)->str:
        print("node_item_name_recognition:步骤3：模型识别")
        if not context:
            return file_title#如果没有内容上下文，就直接返回这个文件的名称
        #llm客户端
        llm_ai = ChatOpenAI(
            model=lm_config.llm_model,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            temperature=lm_config.llm_temperature,
            extra_body={"enable_thinking": False},
        )
        prompt=f"""
                请从以下信息中识别出商品名称与型号：
                文件名：{file_title}
                
                正文切片（用于辅助识别）：
                {context}
                
                要求：
                1. 返回内容为字符串形式，最好是带品牌、型号和名称的完整商品名称。比如：苏伯尓5000W大功率电磁炉；
                2. 返回结果应该只包含商品名称，不要添加任何解释或其他内容；
                3. 如果无法识别商品名称,请返回空字符串。
        """
        #提示词
        #调用
        #解析

        message = [
            SystemMessage("你是一个专业的商品分析识别模型，请根据提供的信息，识别商品的名称，最好不要超过20个字"),
            HumanMessage(content=prompt)
        ]

        response=llm_ai.invoke(message)
        item_name = getattr(response,"content","").strip()
        item_name = (item_name.replace(" ", "")
                     .replace("\n", "")
                     .replace("\t", "")
                     .replace("\r", ""))

        if not item_name:
            return file_title
        return item_name


    def _step_4_update_chunks(self,state,chunks,item_name):
        print("node_item_name_recognition:步骤4：回填数据")
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        return state

    def _step_5_generate_vectors(self, item_name: str):
        """
        步骤 5: 为商品名称生成 BGE-M3 稠密 + 稀疏双向量（Milvus 向量检索核心）
        核心说明：
            - 稠密向量（dense_vector）：BGE-M3 固定 1024 维，记录文本深层语义信息
            - 稀疏向量（sparse_vector）：变长键值对，记录文本关键词/特征位置信息
        依赖工具：
            generate_embeddings：封装 BGE-M3 模型，批量生成双向量，兼容单条/批量输入
        参数：
            item_name: 步骤 3 识别的商品名称（非空，空值时直接返回空向量）
        返回值：
            Tuple[List, Dict]: (稠密向量列表，稀疏向量字典)
        """

        # 1. 商品名称为空 → 直接返回空向量
        if not item_name:
            return None, None

        # 2、调用向量生成工具：传入列表支持批量生成，单条数据仍用列表保证格式统一
        vectors = generate_embeddings([item_name])
        return vectors["dense"][0], vectors["sparse"][0]

    def _step_6_save_to_milvus(self, state, file_title, item_name, dense_vector, sparse_vector):
        print("node_item_name_recognition:步骤6存入milvus向量库")
        pass

if __name__ == "__main__":
    node = NodeItemNameRecognition()
    path = "D:\doc\output\H3C\H3C_new_chunks.json"

    with open(path,"r",encoding="utf-8") as f:
        chunks_json_data = f.read()

    init_state = {
        "file_title":"H3C",
        "chunks":json.loads(chunks_json_data)
    }

    process = node.process(init_state)
    for chunk in process["chunks"]:
        print(f'{chunk.get("title")}章节：{chunk.get("item_name")}')
    print(process)