"""
入口文件：挂载静态文件 + API
  /          → BR/NR 橡胶套利主页
  /multi     → 多策略套利监控
  /api/...   → 橡胶套利 API
  /multi/api/... → 多策略 API
"""
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, uvicorn, threading, time, urllib.request

from api.index import app

# ── 自 ping：防止 Render 免费套餐休眠 ──────────────────────────────
def _self_ping():
    """每14分钟 ping 自己，防止 Render 冷启动。"""
    time.sleep(30)  # 等服务完全启动
    service_url = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not service_url:
        service_name = os.environ.get("RENDER_SERVICE_NAME", "")
        if service_name:
            service_url = f"https://{service_name}.onrender.com"
        else:
            service_url = "https://futures-arb.onrender.com"
    ping_url = f"{service_url}/api"
    while True:
        try:
            req = urllib.request.Request(ping_url, headers={"User-Agent": "self-ping/1.0"})
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass
        time.sleep(14 * 60)  # 14分钟 ping 一次

# 仅在 Render 环境启动自 ping 后台线程
if os.environ.get("RENDER") or os.environ.get("SELF_PING", "").lower() == "true":
    threading.Thread(target=_self_ping, daemon=True, name="self-ping").start()

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
