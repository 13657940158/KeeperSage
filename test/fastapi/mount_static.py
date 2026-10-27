from pathlib import Path

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
print(BASE_DIR)
static_dir = BASE_DIR / "static"
print(static_dir)

app = FastAPI()

# 挂载前端资源
app.mount(path="/static", app=StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)