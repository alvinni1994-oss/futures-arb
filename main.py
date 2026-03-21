"""
入口文件：挂载静态文件 + API
  /          → BR/NR 橡胶套利主页
  /multi     → 多策略套利监控
  /api/...   → 橡胶套利 API
  /multi/api/... → 多策略 API
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, uvicorn

from api.index import app

# ── 多策略路由 ──────────────────────────────────
from multi.api.index import router as multi_router
app.include_router(multi_router, prefix="/multi")

# ── 静态文件 ────────────────────────────────────
_base = os.path.dirname(__file__)
static_dir = os.path.join(_base, "public")
multi_static_dir = os.path.join(_base, "multi", "public")

# /multi/static/* → multi 前端资源
app.mount("/multi/static", StaticFiles(directory=multi_static_dir), name="multi_static")

# /static/* → 主前端资源
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── 页面路由 ────────────────────────────────────
@app.get("/multi", include_in_schema=False)
@app.get("/multi/", include_in_schema=False)
def multi_index():
    return FileResponse(os.path.join(multi_static_dir, "index.html"))

@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8502))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
