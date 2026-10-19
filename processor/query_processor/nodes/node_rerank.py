import json
from typing import List, Dict, Any

from sympy import content

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.reranker_http_utils import rerank_documents


class NodeRerank(NodeBase):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rerank"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        执行重排序
        流程: 合并多源文档 → Reranker 计算相关性 → 断崖检测动态截断
        :param state: 需包含 rrf_chunks、web_search_docs、rewritten_query
        :return: 更新后的 state，包含 reranked_docs
        """

        # 1. 合并多源文档
        merged_multi_docs: List[Dict[str, Any]] = self._step_1_merge_multi_source_docs(state)
        print(merged_multi_docs)
        # 2. Rerank 精排(精排打分)
        reranked_docs: List[Dict[str, Any]] = self._step_2_rerank_merged_docs(state, merged_multi_docs)

        # 3. 动态 Top_K 截取(断崖检测)去掉匹配较低的结果
        cutoff_docs = self._step_3_cliff_cutoff(reranked_docs)

        # 4. 更新state
        state['reranked_docs'] = cutoff_docs

        # 5. 返回state
        return state

    def _step_1_merge_multi_source_docs(self, state):
        print("步骤1: 将web_doc和rrf_chunks节点的数据合并")
        final_docs = []
        rrf_chunks = state.get('rrf_chunks')
        web_docs = state.get('web_search_docs')
        for rrf_doc in rrf_chunks:
            format_rrf_doc = {
                "content": rrf_doc.get('content'),
                "title": rrf_doc.get('title'),
                "chunk_id": rrf_doc.get('chunk_id'),
                "url":None,
                "source" :"local"
            }
            final_docs.append(format_rrf_doc)
        for web_doc in web_docs:
            format_web_doc = {
                "content": web_doc.get('snippet') or web_doc.get('content'),
                "title": web_doc.get('title'),
                "chunk_id": None,
                "url": web_doc.get("url"),
                "source": "web"
            }
            final_docs.append(format_web_doc)
        return final_docs

        # 步骤2: 调用排序模型对合并后结果进行排序

    def _step_2_rerank_merged_docs(self, state, merged_multi_docs):
        print("步骤2: 调用排序模型对合并后结果进行排序")
        rewritten_query = state.get('rewritten_query')

        # 过滤掉没有正文的文档，避免接口参数校验失败
        valid_docs = [doc for doc in merged_multi_docs if doc.get("content")]
        if not valid_docs:
            logger.warning("没有可重排的文档内容，跳过精排")
            return []

        contents = [doc["content"] for doc in valid_docs]
        rerank_scores = rerank_documents(rewritten_query, contents)

        scored_score_docs = [{**doc, "score": score} for doc, score in zip(valid_docs, rerank_scores)]
        return sorted(scored_score_docs, key=lambda x: x['score'], reverse=True)


        # 步骤3: 断崖测试(分数差距, 分数比率)去掉过低的匹配结果

    def _step_3_cliff_cutoff(self, reranked_docs):
        print("步骤3: 断崖测试(分数差距, 分数比率)去掉过低的匹配结果")
        
        return None


if __name__ == "__main__":

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {
                "chunk_id": "local_1",
                "title": "主板维修手册",
                "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"
            },
            {
                "chunk_id": "local_2",
                "title": "闲聊",
                "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"
            },
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/repair",
                "title": "短路查修指南",
                "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"
            },
            {
                "url": "https://example.com/news",
                "title": "科技新闻",
                "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"
            },
        ],
    }

    node_rerank = NodeRerank()
    result = node_rerank(mock_state)
    print(result)
    # logger.info(json.dumps(result, indent=4))