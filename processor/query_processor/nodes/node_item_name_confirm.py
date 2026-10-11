import json
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.milvus_client import milvus_client
from requests import options
from sympy import content

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names


class NodeItemNameConfirm(NodeBase):
    """
    节点功能：确认用户问题中的核心商品名称。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        logger.info(f"【{self.name}】节点逻辑")
        #1 校验参数 #需要session_id ,mongo的会话记忆,和original_query 都是从前端传递过来的
        session_id,original_query = self._step_1_validate_param(state)#mongo_db的记忆
        #2 获取历史会话step_2
        history = get_recent_messages(session_id)
        state["history"] = history
        #3 保存用户信息#这步骤保存的是原始内容step_3
        message_id = save_chat_message(session_id,"user",original_query)#返回的主键ID,把主键的ID拿过来

        #4 模型提取主体：主体识别
        #rewritten_query(改写的问题),item_names(模型识别到的设备主体)
        extract_res = self._step_4_extract_info(original_query,history)
        item_names = extract_res["item_names"]#可能的设备主题名称
        rewritten_query = extract_res["rewritten_query"]#模型改写后的问题
        state["rewritten_query"] = rewritten_query
        state["item_names"] = item_names
        print("extract_res:",extract_res)

        #5 向量搜索（搜索知识库）#6 收索结果对齐（整理）
        align_result = {}
        if len(item_names) > 0:
            #vector
            query_results = self._step_5_vectorize_and_query(item_names)#检索米尔沃思#字典列表
            # print(f"query_results根据提取道德item_names的匹配结果:")
            # for query_result in query_results:
            #     print(query_result)
            align_result = self._step_6_align_item_names(query_results)#结果整理,需要存入state#字典

            # print("milvus根据提取到的item_names的匹配结果,并且经过了分数整理(结果对齐):")

            # names_list = align_result["confirmed_item_names"]
            # options = align_result["options"]
            # print(names_list)
            # print("------------------------------")
            # print(options)
        else:
            logger.info("Node: 未提取到商品名，跳过向量检索")


        #7 状态state信息整理（确认状态）将第六步的结果配置到state当中
        state = self._step_7_check_confirmation(state,align_result,history)

        #8 写入历史会话（将识别完成的内容保存到记忆）
        self._step_8_write_history(state,session_id,rewritten_query,message_id)

        return state
    #参数校验
    def _step_1_validate_param(self, state):
        print("step_1:参数校验")
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数session_id缺失")

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数original_query缺失")

        return session_id, original_query
    #步骤4 模型提取意图主体
    def _step_4_extract_info(self, original_query, history)-> Dict:
        print("step_4:模型提取意图主题")
        #llm客户端
        ai_client =get_llm_client()
        #拼接上下文（history+original_query）,prompt
        history_text = ""
        for msg in history:
            role = msg.get("text")
            history_text += f"{role}:{content}\n"
        user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query,
        )
        #调用大模型
        messages = [
            SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        response = ai_client.invoke(messages)
        response_content=response.content
        #结果解析
        result = json.loads(response_content)
        print("result:",result)
        if "item_names" not in result:
            result["item_names"] = []
        if "rewritten_query" not in result:
            result["rewritten_query"] = original_query
        if result["item_names"]:
            result["item_names"] = [name for name in result["item_names"]]
        return result
    #步骤五 向量化并检索
    def _step_5_vectorize_and_query(self, item_names)->List[Dict]:
        print("step_5:向量化检索，检索出来对应item_name的向量库中的相似的商品名称的列表")
        results: List[Dict] = []#(大模型识别的可能名称:milvus的匹配结果集合)k:

        #获得milvus的客户端
        milvus_client = get_milvus_client()
        collection_name=milvus_config.item_name_collection
        #条件向量化
        embeddings=generate_embeddings(item_names)
        #相似性匹配
        for i in range(len(item_names)):
            dense_vector = embeddings.get("dense")[i]
            sparse_vector = embeddings.get("sparse")[i]
            reqs = create_hybrid_search_requests(dense_vector=dense_vector,sparse_vector=sparse_vector)
            search_res=hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                reqs=reqs,
                ranker_weights=(0.8,0.2),
                norm_score=True,
                output_fields=["item_name"],
            )
        #返回
            matches = []
        if search_res and len(search_res)>0:
            print("意图识别的向量搜索结果search_res:",search_res)
            for hit in search_res[0]:
                #将search_res中的匹配结果封装到matches中
                matches.append({
                    "item_name": hit.entity.get("item_name"),
                    "score": hit.get("distance"),
                })
        #这两个值共同封装为result
            results.append({
                "extracted_name":item_names[i],#大模型提取的名称
                "matches":matches,#大模型在milvus中提取的信息列表
            })

        return results
    #步骤六 结果对齐
    def _step_6_align_item_names(self, query_results: List[Dict])->Dict:
        print("step_6:对齐结果:")
        confirmed_item_names: List[str] = []#高于0.8
        options:List[str]=[]#高于0.6低于0.8
        #高于0.8为高度匹配，直接封装结果你直接变成要搜索的item_name
        for res in query_results:
            extracted_name = (res.get("extracted_name","")or "").strip()#模型识别的商品名称
            matches = res.get("matches",[]) or []
            if not matches:
                continue
            high_results = [m for m in matches if m.get("score",0)>=0.8]#高分结果
            mid_results = [m for m in matches if m.get("score", 0) >= 0.6]#中分结果
            ###################################################################################################################
            #特殊，只有一条结果高于0.8
            if len(high_results) == 1:
                confirmed_item_names.append(high_results[0].get("item_name"))
                continue
            #有多条匹配结果，找出最匹配的
            if len(high_results) > 1:
                picked = None
                if extracted_name:
                    for hr in high_results:
                        if hr.get("item_name") == extracted_name:
                            picked = hr
                            break
                #如果没有和大模型识别的，则取分数最高的
                if not picked:
                    picked = high_results[0]

                #确认名称
                confirmed_item_names.append(picked.get("item_name"))
                continue#确认名称不止一个
            ####################################################################################################
            #高于0.6低于0.8为可能匹配，返回answer，有可选项
            if len(mid_results) > 0:
                for mr in mid_results[:5]:
                    options.append(mr.get("item_name"))
            #####################################################################################################
        #低于0.6无匹配结果,都是空


        return {
            "confirmed_item_names":list(set(confirmed_item_names)),#确认后的高分商品名称（>0.8）
            "options": list(set(options))#可能低分商品名称（>0.6）
        }
    def _step_7_check_confirmation(self, state,align_result,history)->Dict:
        print("step_7:状态stat信息,根据step_6高分低分对齐结果整理")
        confirmed = align_result.get("confirmed_item_names")
        options = align_result.get("options")
        #1 有命中(>0.8)
        if confirmed:
            #更新会话信息,将于本次命中结果更新到与本次有关的所有的会话中（session_id,_id）
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    mid =msg.get("_id")
                    if mid :
                        ids_to_update.append(str(mid))
            if ids_to_update:
                update_message_item_names(ids_to_update,confirmed)

            #封装结果
            state["item_names"] = confirmed
            state["answer"]=""

        #2 没命中，有备选(>0.6)
        if options:
            state["item_names"] = []
            options_str = ",".join(options)
            state["answer"]=f"请选择一个商品,您想问的是这个产品吗？{options_str}请明确一下型号"

        #3 没命中
        if not confirmed and not options:
            state["answer"]="抱歉，我没有找到你想要的产品"#如果有高于0.6的可选项options(反问用户，你想问的是xxx设备吗或者低于0.6（未找到产品）
            state["item_names"]=[]#有高于0.8就封装
            #back to state
        return state



    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        print("step_8:写入历史会话，更新")
        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            save_chat_message(
                session_id=session_id,  # 会话ID，关联所属会话
                role="assistant",  # 消息角色：助手
                text=state["answer"],  # 消息内容：向用户确认的提示语/无结果提示语
                rewritten_query="",  # 助手消息无需改写查询，设为空
                item_names=state.get("item_names", [])  # 关联的商品名列表（分支B/C均为空）
            )

        # 强制更新本次用户原始问题的关联信息（核心：补充改写查询、商品名）
        save_chat_message(
            session_id=session_id,  # 会话ID，关联所属会话
            role="user",  # 消息角色：用户
            text=state["original_query"],  # 消息内容：用户原始查询
            rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
            item_names=state.get("item_names", []),  # 补充关联的商品名列表
            message_id=message_id  # 消息ID，指定更新已存在的用户消息（而非新增）
        )

        # 返回最终会话状态，供下游节点使用
        return state


if __name__ == "__main__":

    # 初始化图状态
    init_state = {
        "original_query": "h3c是干啥子的？",
        "session_id": 1,
    }

    # 创建节点对象
    node_item_name_confirm = NodeItemNameConfirm()
    # 执行节点的单元测试
    result = node_item_name_confirm(init_state)
    # 将返回的图状态进行json序列化
    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)