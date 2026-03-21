"""
期货套利分析 API
支持 Vercel Python Serverless Functions
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI(title="期货套利分析 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FUTURES_DICT = {
    "BR": "BR0", "RU": "RU0", "NR": "NR0",
    "BU": "BU0", "FU": "FU0", "SC": "SC0",
    "L": "L0",   "PP": "PP0", "EB": "EB0",
    "PG": "PG0", "V":  "V0",  "MA": "MA0",
    "TA": "TA0", "EG": "EG0",
    "CU": "CU0", "AL": "AL0", "ZN": "ZN0",
    "AU": "AU0", "AG": "AG0",
    "I":  "I0",  "J":  "J0",  "JM": "JM0",
    "M":  "M0",  "Y":  "Y0",  "P":  "P0",
    "OI": "OI0", "C":  "C0",  "CF": "CF0",
    "SR": "SR0", "RM": "RM0",
}

RECOMMENDED_PAIRS = [
    {"a": "BR", "b": "NR", "note": "橡胶替代"},
    {"a": "BR", "b": "RU", "note": "橡胶替代"},
    {"a": "RU", "b": "NR", "note": "橡胶替代"},
    {"a": "L",  "b": "PP", "note": "烯烃价差"},
    {"a": "J",  "b": "JM", "note": "焦化价差"},
    {"a": "M",  "b": "RM", "note": "蛋白替代"},
    {"a": "Y",  "b": "P",  "note": "油脂价差"},
    {"a": "AU", "b": "AG", "note": "贵金属比价"},
    {"a": "MA", "b": "EG", "note": "化工联动"},
    {"a": "CU", "b": "AL", "note": "有色比价"},
    {"a": "EB", "b": "L",  "note": "化工裂解"},
    {"a": "SC", "b": "FU", "note": "油品裂解"},
    {"a": "I",  "b": "J",  "note": "钢铁产业链"},
    {"a": "Y",  "b": "OI", "note": "油脂价差"},
    {"a": "CF", "b": "SR", "note": "软商品"},
]

import functools, time

_cache: dict = {}
CACHE_TTL = 3600  # 历史数据1小时缓存

# 实时行情缓存（30秒）
_rt_cache: dict = {}
CACHE_RT_TTL = 30

# 品种名称映射 (用于 futures_zh_realtime)
REALTIME_NAME_MAP = {
    "BR": "丁二烯橡胶",
    "NR": "20号胶",
    "RU": "橡胶",
    "BU": "沥青",
    "FU": "燃油",
    "SC": "原油",
    "AU": "黄金",
    "AG": "白银",
    "CU": "沪铜",
    "AL": "沪铝",
    "ZN": "沪锌",
    "L":  "塑料",
    "PP": "PP",
    "EB": "苯乙烯",
    "MA": "郑醇",
    "EG": "乙二醇",
    "I":  "铁矿石",
    "J":  "焦炭",
    "JM": "焦煤",
    "M":  "豆粕",
    "Y":  "豆油",
    "P":  "棕榈",
    "OI": "菜油",
    "CF": "棉花",
    "SR": "白糖",
    "RM": "菜粕",
}

def get_realtime_price(symbol: str) -> dict | None:
    """获取实时行情（主力合约最新价），30秒缓存"""
    import akshare as ak
    now = time.time()
    if symbol in _rt_cache:
        data, ts = _rt_cache[symbol]
        if now - ts < CACHE_RT_TTL:
            return data
    
    name = REALTIME_NAME_MAP.get(symbol)
    if not name:
        return None
    
    try:
        df = ak.futures_zh_realtime(symbol=name)
        # 优先取持仓量最大的具体月份合约（如 BR2605），排除 BR0 通用代码
        specific = df[df["symbol"].str.match(f"^{symbol}\\d{{4}}$")]
        if not specific.empty:
            main = specific.sort_values("position", ascending=False).head(1)
        else:
            # 回退到通用主力代码（BR0）
            main = df[df["symbol"] == f"{symbol}0"]
        if main.empty:
            return None
        row = main.iloc[0]
        result = {
            "symbol": str(row["symbol"]),
            "price": float(row["trade"]),
            "open": float(row["open"]) if pd.notna(row["open"]) else None,
            "high": float(row["high"]) if pd.notna(row["high"]) else None,
            "low": float(row["low"]) if pd.notna(row["low"]) else None,
            "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            "position": int(row["position"]) if pd.notna(row["position"]) else None,
            "source": "realtime",
            "ts": int(now),
        }
        _rt_cache[symbol] = (result, now)
        return result
    except Exception:
        return None


def load_data(symbol: str, years: int = 3) -> pd.DataFrame:
    cache_key = f"{symbol}_{years}"
    now = time.time()
    if cache_key in _cache:
        df, ts = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return df

    import akshare as ak
    df = ak.futures_main_sina(symbol=symbol)
    df.columns = ["date", "open", "high", "low", "close", "volume", "oi", "settle"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df["date"] >= cutoff].reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    _cache[cache_key] = (df, now)
    return df


def compute_pair(sym_a: str, sym_b: str, years: int = 3) -> dict:
    da = load_data(FUTURES_DICT[sym_a], years)
    db = load_data(FUTURES_DICT[sym_b], years)

    merged = pd.merge(
        da[["date", "close"]].rename(columns={"close": "ca"}),
        db[["date", "close"]].rename(columns={"close": "cb"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    # 尝试用实时行情更新最后一行（盘中/夜盘实时价格）
    rt_a = get_realtime_price(sym_a)
    rt_b = get_realtime_price(sym_b)
    realtime_updated = False
    if rt_a and rt_b and rt_a["price"] > 0 and rt_b["price"] > 0:
        today = pd.Timestamp.now().normalize()
        last_date = merged["date"].iloc[-1]
        if last_date < today:
            # 夜盘/盘中：追加今日实时行情作为新行
            new_row = pd.DataFrame([{"date": today, "ca": rt_a["price"], "cb": rt_b["price"]}])
            merged = pd.concat([merged, new_row], ignore_index=True)
        else:
            # 当日：用实时价覆盖最后一行（更准确）
            merged.loc[merged.index[-1], "ca"] = rt_a["price"]
            merged.loc[merged.index[-1], "cb"] = rt_b["price"]
        realtime_updated = True

    if len(merged) < 20:
        raise ValueError("数据不足")

    ratio = merged["ca"] / merged["cb"]
    diff = merged["ca"] - merged["cb"]
    mean_r = float(ratio.mean()); std_r = float(ratio.std())
    cur_r = float(ratio.iloc[-1])
    z_r = (cur_r - mean_r) / std_r if std_r > 0 else 0
    pct_r = float(stats.percentileofscore(ratio.dropna(), cur_r))

    mean_d = float(diff.mean()); std_d = float(diff.std())
    cur_d = float(diff.iloc[-1])
    z_d = (cur_d - mean_d) / std_d if std_d > 0 else 0
    pct_d = float(stats.percentileofscore(diff.dropna(), cur_d))

    boll_mid_r = ratio.rolling(20).mean()
    boll_up_r = boll_mid_r + 2 * ratio.rolling(20).std()
    boll_dn_r = boll_mid_r - 2 * ratio.rolling(20).std()
    boll_mid_d = diff.rolling(20).mean()
    boll_up_d = boll_mid_d + 2 * diff.rolling(20).std()
    boll_dn_d = boll_mid_d - 2 * diff.rolling(20).std()

    corr = {}
    now = merged["date"].max()
    for label, days in [("1M", 21), ("3M", 63), ("1Y", 252), ("3Y", 756)]:
        sub = merged[merged["date"] >= now - pd.Timedelta(days=days)]
        if len(sub) >= 5:
            corr[label] = round(float(sub["ca"].corr(sub["cb"])), 3)

    def safe(lst):
        return [round(float(x), 4) if pd.notna(x) else None for x in lst]

    return {
        "sym_a": sym_a, "sym_b": sym_b,
        "dates": merged["date"].dt.strftime("%Y-%m-%d").tolist(),
        "ca": safe(merged["ca"]),
        "cb": safe(merged["cb"]),
        "ratio": safe(ratio),
        "diff": safe(diff),
        "boll_mid_r": safe(boll_mid_r),
        "boll_up_r": safe(boll_up_r),
        "boll_dn_r": safe(boll_dn_r),
        "boll_mid_d": safe(boll_mid_d),
        "boll_up_d": safe(boll_up_d),
        "boll_dn_d": safe(boll_dn_d),
        "corr": corr,
        "ratio_stats": {
            "cur": round(cur_r, 4), "mean": round(mean_r, 4),
            "std": round(std_r, 4), "z": round(z_r, 3),
            "pct": round(pct_r, 1),
            "p5": round(float(np.percentile(ratio.dropna(), 5)), 4),
            "p95": round(float(np.percentile(ratio.dropna(), 95)), 4),
        },
        "diff_stats": {
            "cur": round(cur_d, 2), "mean": round(mean_d, 2),
            "std": round(std_d, 2), "z": round(z_d, 3),
            "pct": round(pct_d, 1),
            "p5": round(float(np.percentile(diff.dropna(), 5)), 2),
            "p95": round(float(np.percentile(diff.dropna(), 95)), 2),
        },
        "ca_last": round(float(merged["ca"].iloc[-1]), 2),
        "cb_last": round(float(merged["cb"].iloc[-1]), 2),
        "last_date": merged["date"].iloc[-1].strftime("%Y-%m-%d"),
        "count": len(merged),
        "realtime": realtime_updated,
        "rt_sym_a": rt_a["symbol"] if rt_a else None,
        "rt_sym_b": rt_b["symbol"] if rt_b else None,
    }


@app.get("/api")
def root():
    return {"status": "ok", "message": "期货套利分析 API"}


@app.get("/api/symbols")
def get_symbols():
    return {
        "symbols": list(FUTURES_DICT.keys()),
        "recommended_pairs": RECOMMENDED_PAIRS,
    }


@app.get("/api/pair")
def get_pair(a: str, b: str, years: int = 3):
    a = a.upper(); b = b.upper()
    if a not in FUTURES_DICT:
        return JSONResponse({"error": f"未知品种: {a}"}, status_code=400)
    if b not in FUTURES_DICT:
        return JSONResponse({"error": f"未知品种: {b}"}, status_code=400)
    if a == b:
        return JSONResponse({"error": "品种 A 和 B 不能相同"}, status_code=400)
    try:
        result = compute_pair(a, b, years)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/realtime")
def get_realtime(a: str = "BR", b: str = "NR"):
    """快速实时行情接口（30秒缓存），不重算历史统计"""
    a = a.upper(); b = b.upper()
    rt_a = get_realtime_price(a)
    rt_b = get_realtime_price(b)
    if not rt_a or not rt_b:
        return JSONResponse({"error": "实时行情获取失败"}, status_code=503)
    return {
        "a": a, "b": b,
        "ca": rt_a["price"], "cb": rt_b["price"],
        "sym_a": rt_a["symbol"], "sym_b": rt_b["symbol"],
        "ratio": round(rt_a["price"] / rt_b["price"], 4) if rt_b["price"] else None,
        "diff": round(rt_a["price"] - rt_b["price"], 2),
        "ts": int(time.time()),
    }


@app.get("/api/batch")
def get_batch(years: int = 3):
    """批量获取所有推荐对的摘要"""
    results = []
    for pair in RECOMMENDED_PAIRS:
        try:
            d = compute_pair(pair["a"], pair["b"], years)
            results.append({
                "a": pair["a"], "b": pair["b"], "note": pair["note"],
                "ratio_z": d["ratio_stats"]["z"],
                "ratio_cur": d["ratio_stats"]["cur"],
                "ratio_pct": d["ratio_stats"]["pct"],
                "corr_1y": d["corr"].get("1Y"),
                "ca_last": d["ca_last"],
                "cb_last": d["cb_last"],
                "last_date": d["last_date"],
            })
        except Exception as e:
            results.append({"a": pair["a"], "b": pair["b"], "note": pair["note"], "error": str(e)})
    return results


# ===== 邮件发送接口（Resend HTTP API） =====
import urllib.request
import json as _json
from pydantic import BaseModel

class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str

@app.post("/api/send-email")
async def send_email_api(req: EmailRequest):
    """
    使用 Resend HTTP API 发送邮件（兼容 Render 免费版网络限制）。
    环境变量：RESEND_API_KEY
    """
    api_key = os.environ.get("RESEND_API_KEY", "")

    if not api_key:
        return JSONResponse({
            "ok": False,
            "error": "邮件服务未配置，请在 Render 环境变量中设置 RESEND_API_KEY"
        }, status_code=503)

    try:
        payload = _json.dumps({
            "from": "BR/NR套利监控 <onboarding@resend.dev>",
            "to": [req.to],
            "subject": req.subject,
            "text": req.body,
        }).encode("utf-8")

        request = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=10) as resp:
            result = _json.loads(resp.read().decode())
            return {"ok": True, "message": f"邮件已发送至 {req.to}", "id": result.get("id")}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        return JSONResponse({"ok": False, "error": f"Resend API 错误: {err_body}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ===== 图表图片生成接口 =====
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests as _requests

# 图片缓存（5分钟）
_chart_cache: dict = {}
CHART_CACHE_TTL = 300

def upload_to_imgbb(img_bytes: bytes) -> str | None:
    """上传图片到 imgbb，返回直链 URL"""
    api_key = os.environ.get("IMGBB_API_KEY", "")
    if not api_key:
        return None
    try:
        b64 = base64.b64encode(img_bytes).decode()
        r = _requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": api_key, "image": b64, "expiration": 600},  # 10分钟有效
            timeout=15
        )
        j = r.json()
        return j["data"]["url"] if j.get("success") else None
    except Exception:
        return None

def generate_chart(sym_a: str = "BR", sym_b: str = "NR", days: int = 180) -> bytes:
    """生成价差走势图 + 两品种价格走势图，返回 PNG bytes"""
    import akshare as ak

    da = load_data(FUTURES_DICT[sym_a])
    db = load_data(FUTURES_DICT[sym_b])
    merged = pd.merge(
        da[["date","close"]].rename(columns={"close":"ca"}),
        db[["date","close"]].rename(columns={"close":"cb"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    # 接入实时价格
    rt_a = get_realtime_price(sym_a)
    rt_b = get_realtime_price(sym_b)
    if rt_a and rt_b and rt_a["price"] > 0 and rt_b["price"] > 0:
        today = pd.Timestamp.now().normalize()
        last_date = merged["date"].iloc[-1]
        if last_date < today:
            new_row = pd.DataFrame([{"date": today, "ca": rt_a["price"], "cb": rt_b["price"]}])
            merged = pd.concat([merged, new_row], ignore_index=True)
        else:
            merged.loc[merged.index[-1], "ca"] = rt_a["price"]
            merged.loc[merged.index[-1], "cb"] = rt_b["price"]

    # 取最近 N 天
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = merged[merged["date"] >= cutoff].copy()
    diff = df["ca"] - df["cb"]
    boll_mid = diff.rolling(20).mean()
    boll_up  = boll_mid + 2 * diff.rolling(20).std()
    boll_dn  = boll_mid - 2 * diff.rolling(20).std()
    mean_all = float((merged["ca"] - merged["cb"]).mean())
    cur_diff = float(diff.iloc[-1])
    cur_br   = float(df["ca"].iloc[-1])
    cur_nr   = float(df["cb"].iloc[-1])

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=120,
                                    gridspec_kw={"height_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#f8fafc")

    # --- 上图：两品种价格走势 ---
    ax1.set_facecolor("#f8fafc")
    ax1.plot(df["date"], df["ca"], color="#ef4444", linewidth=1.5, label=f"BR  latest={cur_br:,.0f}")
    ax1.plot(df["date"], df["cb"], color="#3b82f6", linewidth=1.5, label=f"NR  latest={cur_nr:,.0f}")
    ax1.set_title(f"BR / NR  Price Trend  (latest {days}d)", fontsize=13, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(labelsize=8)
    for spine in ax1.spines.values(): spine.set_alpha(0.3)

    # --- 下图：价差+布林带 ---
    ax2.set_facecolor("#f8fafc")
    ax2.fill_between(df["date"], boll_dn, boll_up, alpha=0.12, color="#6366f1", label="Boll Band (±2σ)")
    ax2.plot(df["date"], boll_mid, color="#6366f1", linewidth=1, linestyle="--", alpha=0.7, label=f"20d MA  {boll_mid.iloc[-1]:,.0f}")
    ax2.plot(df["date"], diff, color="#f59e0b", linewidth=1.8, label=f"Spread  cur={cur_diff:+,.0f}")
    ax2.axhline(mean_all, color="#94a3b8", linewidth=1, linestyle=":", alpha=0.8, label=f"3Y Mean  {mean_all:,.0f}")

    # 标注当前价差
    ax2.scatter([df["date"].iloc[-1]], [cur_diff], color="#ef4444", s=60, zorder=5)
    ax2.annotate(f" {cur_diff:+,.0f}", xy=(df["date"].iloc[-1], cur_diff),
                 fontsize=9, color="#ef4444", fontweight="bold")

    ax2.set_title(f"BR-NR Spread  3Y mean={mean_all:,.0f}  cur={cur_diff:+,.0f}", fontsize=12, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(labelsize=8)
    for spine in ax2.spines.values(): spine.set_alpha(0.3)
    ax2.axhline(0, color="#94a3b8", linewidth=0.5, alpha=0.5)

    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.get("/api/chart-image")
def get_chart_image(a: str = "BR", b: str = "NR", days: int = 180):
    """
    生成价差+价格走势图，上传 imgbb 并返回 URL。
    需要环境变量 IMGBB_API_KEY。
    返回 {"url": "...", "base64": "..."（无imgbb时）}
    """
    cache_key = f"{a}_{b}_{days}"
    now = time.time()
    if cache_key in _chart_cache:
        data, ts = _chart_cache[cache_key]
        if now - ts < CHART_CACHE_TTL:
            return data

    try:
        img_bytes = generate_chart(a.upper(), b.upper(), days)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # 尝试上传 imgbb
    url = upload_to_imgbb(img_bytes)
    b64 = base64.b64encode(img_bytes).decode()

    result = {
        "ok": True,
        "url": url,
        "base64": f"data:image/png;base64,{b64}",
        "has_url": url is not None,
    }
    _chart_cache[cache_key] = (result, now)
    return result
