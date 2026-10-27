from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse
from fastapi import Request
from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import get_recent_messages
from utils.sse_utils import create_sse_queue, sse_generator
from utils.task_utils import update_task_status, TASK_STATUS_PROCESSING, get_task_result

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    session_id: str
    is_stream: bool
@app.get("/chat.html")
async def chat():
    current_dir_parent_path= Path(__file__).absolute().parent.parent
    chat_html_path = current_dir_parent_path / "page" / "chat.html"
    return FileResponse(chat_html_path)

@app.post("/query")
async def query(query: QueryRequest,background_tasks: BackgroundTasks):
    user_query = query.query
    session_id = query.session_id
    is_stream = query.is_stream

    if is_stream:
        create_sse_queue(session_id)

    update_task_status(session_id,TASK_STATUS_PROCESSING,is_stream)#进度
    if is_stream:
        background_tasks.add_task(run_query_graph, session_id,user_query,is_stream)#调用工作流
        return {"message":"任务开始，请耐心等待","session_id":session_id}
    else:
        run_query_graph(session_id,user_query,is_stream)
        answer=get_task_result(session_id,"answer","")
        return {
            "message":"处理完成",
            "session_id":session_id,
            "answer":answer
        }
def run_query_graph(session_id, user_query, is_stream):
    print("调用搜索工作流")
    init_state={
        "original_query":user_query,
        "session_id":session_id,
            "is_stream":is_stream
    }
    workflow = KBQueryWorkflow()
    for chunk in workflow.run(init_state,stream=is_stream):
        pass
@app.get("/stream/{session_id}")
async def stream(session_id: str,request: Request):
    print("追踪生成结果.....")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """
    查询当前会话历史记录
    """
    records = get_recent_messages(session_id, limit=limit)
    items = []
    for r in records:
        items.append({
            "_id": str(r.get("_id")) if r.get("_id") is not None else "",
            "session_id": r.get("session_id", ""),
            "role": r.get("role", ""),
            "text": r.get("text", ""),
            "rewritten_query": r.get("rewritten_query", ""),
            "item_names": r.get("item_names", []),
            "ts": r.get("ts")
        })
    return {"session_id": session_id, "items": items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)