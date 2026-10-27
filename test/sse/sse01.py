import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()

# ✅ 必须加这段，否则8001的前端请求8000会被浏览器拦截
app.add_middleware(CORSMiddleware, allow_origins=["*"])

async def event_generator():
    for i in range(10):
        # SSE 协议规定：数据必须以 "data: " 开头，以 "\n\n" 结尾
        yield f"data: 这是第 {i} 条流式消息\n\n"
        await asyncio.sleep(1)
#后端直接建立长连接，不通过前端
@app.get("/simple_stream")
async def simple_stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)