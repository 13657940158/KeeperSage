import dashscope
from agents.voice import result
from dashscope.rerank import text_rerank
from dotenv import load_dotenv
from config.reranker_config import reranker_config

load_dotenv()

def rerank_documents(query: str, documents: list[str]) -> list[float]:#直接返回分,没有排序
    text_rerank_api_key=reranker_config.text_rerank_api_key
    text_rerank_model=reranker_config.text_rerank_model
    #调用排序模型对文档的相关度打分
    response = dashscope.TextReRank.call(
        model=text_rerank_model,
        api_key=text_rerank_api_key,
        query=query,
        documents=documents,
        top_n=len(documents),
        instruct=reranker_config.text_rerank_instruct,
    )

    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")

    rerank_scores = [0.0] * len(documents)
    results = response.get("output").get("results")
    for index,result in enumerate(results):
        rerank_scores[index] = result.get("relevance_score")
    print(f"rerank scores: {rerank_scores}")
    return rerank_scores