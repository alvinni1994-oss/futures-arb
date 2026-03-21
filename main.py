"""
入口文件：挂载静态文件 + API
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, uvicorn

from api.index import app

# 挂载前端静态文件
static_dir = os.path.join(os.path.dirname(__file__), "public")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 根路径返回 index.html
@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
