"""
期货套利分析 API
支持 Vercel Python Serverless Functions
"""
from fastapi import FastAPI, HTTPException
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

# ===== 策略配置表 =====
STRATEGIES = [
    # ── 橡胶套利 ──
    {
        "id": "br_nr", "name": "BR/NR 橡胶套利", "category": "rubber",
        "unit": "元/吨", "legs": ["BR", "NR"],
        "formula": "BR - NR",
        "ratio": [2, 1],
        "lot_sizes": [5, 10],
        "open_hi": 1500, "strong_hi": 2500,
        "open_lo": -800, "strong_lo": -1500,
        "tp": 500, "sl": 4500,
        "note": "2手BR(5t×2=10t) : 1手NR(10t×1=10t)"
    },
    # ── 股指期货 ──
    {
        "id": "im_ic", "name": "IM-IC 价差", "category": "equity_index",
        "unit": "点", "legs": ["IM", "IC"],
        "formula": "IM - IC",
        "ratio": [1, 1],
        "lot_sizes": [1, 1],
        "open_hi": 500, "strong_hi": 800,
        "open_lo": -200, "strong_lo": -400,
        "tp": 150, "sl": 1200,
        "note": "1手IM : 1手IC"
    },
    # ── 加工费套利 ──
    {
        "id": "ta_px", "name": "TA加工费", "category": "processing",
        "unit": "元/吨", "legs": ["TA", "PX"],
        "formula": "TA - PX * 0.655",
        "ratio": [3, 2],
        "lot_sizes": [5, 5],
        "open_hi": 200, "strong_hi": 400,
        "open_lo": -100, "strong_lo": -200,
        "tp": 100, "sl": 600,
        "note": "多3TA空2PX；加工费=TA−PX×0.655"
    },
    {
        "id": "pr_bottle", "name": "瓶片加工费", "category": "processing",
        "unit": "元/吨", "legs": ["PR", "TA", "EG"],
        "formula": "PR - TA * 0.855 - EG * 0.335",
        "ratio": [2, 5, 1],
        "lot_sizes": [5, 5, 10],
        "open_hi": 1200, "strong_hi": 1500,
        "open_lo": 500, "strong_lo": 300,
        "tp": 200, "sl": 800,
        "note": "空2PR多5TA多1EG；正常区间800-1200元/吨"
    },
    {
        "id": "pr_pf", "name": "PR-PF 相对加工费", "category": "processing",
        "unit": "元/吨", "legs": ["PR", "PF", "TA", "EG"],
        "formula": "(PR - TA * 0.855 - EG * 0.332) - (PF - TA * 0.855 - EG * 0.335)",
        "ratio": [1, 3, 0, 0],
        "lot_sizes": [5, 5, 5, 10],
        "open_hi": 1500, "strong_hi": 2000,
        "open_lo": 500, "strong_lo": 300,
        "tp": 300, "sl": 2500,
        "note": "空1PR多3PF；正常区间800-1200，>1500空PR多PF，<500多PR空PF"
    },
    # ── 跨品种套利 ──
    {
        "id": "l_pp", "name": "L-PP 跨品种", "category": "cross_product",
        "unit": "元/吨", "legs": ["L", "PP"],
        "formula": "L - PP",
        "ratio": [1, 1],
        "lot_sizes": [5, 5],
        "open_hi": 300, "strong_hi": 500,
        "open_lo": -300, "strong_lo": -500,
        "tp": 100, "sl": 700,
        "note": "1手L : 1手PP"
    },
    {
        "id": "rm_oi", "name": "菜粕-菜油", "category": "cross_product",
        "unit": "元（加权）", "legs": ["RM", "OI"],
        "formula": "RM * 4 - OI",
        "ratio": [4, 1],
        "lot_sizes": [10, 10],
        "open_hi": 500, "strong_hi": 800,
        "open_lo": -200, "strong_lo": -400,
        "tp": 150, "sl": 1000,
        "note": "4手RM空1手OI；加权价差=RM×4−OI"
    },
    {
        "id": "m_rm", "name": "豆粕-菜粕", "category": "cross_product",
        "unit": "元/吨", "legs": ["M", "RM"],
        "formula": "M - RM",
        "ratio": [1, 1],
        "lot_sizes": [10, 10],
        "open_hi": 200, "strong_hi": 350,
        "open_lo": -200, "strong_lo": -350,
        "tp": 80, "sl": 500,
        "note": "1手M : 1手RM"
    },
    {
        "id": "rm_c", "name": "菜粕-玉米", "category": "cross_product",
        "unit": "元/吨", "legs": ["RM", "C"],
        "formula": "RM - C",
        "ratio": [1, 1],
        "lot_sizes": [10, 5],
        "open_hi": 300, "strong_hi": 500,
        "open_lo": 100, "strong_lo": -50,
        "tp": 100, "sl": 700,
        "note": "1手RM : 1手C"
    },
    {
        "id": "lu_fu", "name": "LU-FU 跨品种", "category": "cross_product",
        "unit": "元/吨", "legs": ["LU", "FU"],
        "formula": "LU - FU",
        "ratio": [1, 1],
        "lot_sizes": [10, 10],
        "open_hi": 300, "strong_hi": 500,
        "open_lo": -300, "strong_lo": -500,
        "tp": 100, "sl": 700,
        "note": "1手LU : 1手FU"
    },
]

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
    # 补充缺失品种
    "IM": "IM0",   # 中证1000
    "IC": "IC0",   # 中证500
    "PX": "PX0",   # PX
    "PR": "PR0",   # 涤纶短纤（郑商所）
    "PF": "SF0",   # 短纤（akshare代码SF0）
    "LU": "LU0",   # 低硫燃料油
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
    "IM": "中证1000",
    "IC": "中证500",
    "PX": "对二甲苯",
    "PR": "涤纶短纤",
    "PF": "短纤",
    "LU": "低硫燃料油",
    "TA": "PTA",
    "C":  "玉米",
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


def compute_spread(strategy_id: str, years: int = 3) -> dict:
    """通用多腿价差计算"""
    strat = next((s for s in STRATEGIES if s["id"] == strategy_id), None)
    if not strat:
        raise ValueError(f"未知策略: {strategy_id}")

    legs = strat["legs"]
    formula = strat["formula"]

    # 加载各腿数据
    dfs = {}
    for sym in legs:
        ak_symbol = FUTURES_DICT.get(sym)
        if not ak_symbol:
            raise ValueError(f"未知品种代码: {sym}")
        df = load_data(ak_symbol, years)
        dfs[sym] = df[["date", "close"]].rename(columns={"close": sym})

    # 合并（inner join on date）
    merged = dfs[legs[0]]
    for sym in legs[1:]:
        merged = pd.merge(merged, dfs[sym], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 20:
        raise ValueError("数据不足，无法计算统计信息")

    # 尝试更新实时行情
    realtime_updated = False
    rt_data = {}
    for sym in legs:
        rt = get_realtime_price(sym)
        if rt and rt["price"] > 0:
            rt_data[sym] = rt
    if len(rt_data) == len(legs):
        today = pd.Timestamp.now().normalize()
        last_date = merged["date"].iloc[-1]
        new_vals = {"date": today}
        for sym in legs:
            new_vals[sym] = rt_data[sym]["price"]
        if last_date < today:
            merged = pd.concat([merged, pd.DataFrame([new_vals])], ignore_index=True)
        else:
            for sym in legs:
                merged.loc[merged.index[-1], sym] = new_vals[sym]
        realtime_updated = True

    # 计算价差：先尝试 pd.eval，失败则用手动计算
    try:
        spread = merged.eval(formula)
    except Exception:
        # 手动计算兜底
        local_vars = {sym: merged[sym] for sym in legs}
        spread = eval(formula, {"__builtins__": {}}, local_vars)

    spread = pd.Series(spread.values if hasattr(spread, "values") else spread, name="spread")

    mean_d = float(spread.mean())
    std_d = float(spread.std())
    cur_d = float(spread.iloc[-1])
    z_d = (cur_d - mean_d) / std_d if std_d > 0 else 0
    pct_d = float(stats.percentileofscore(spread.dropna(), cur_d))

    boll_mid = spread.rolling(20).mean()
    boll_up = boll_mid + 2 * spread.rolling(20).std()
    boll_dn = boll_mid - 2 * spread.rolling(20).std()

    # 相关性（用第1、2腿）
    sym_a, sym_b = legs[0], legs[1]
    corr = {}
    now_dt = merged["date"].max()
    for label, days in [("1M", 21), ("3M", 63), ("1Y", 252), ("3Y", 756)]:
        sub = merged[merged["date"] >= now_dt - pd.Timedelta(days=days)]
        if len(sub) >= 5:
            corr[label] = round(float(sub[sym_a].corr(sub[sym_b])), 3)

    def safe(lst):
        return [round(float(x), 2) if pd.notna(x) else None for x in lst]

    # 各腿最新价
    leg_prices = {sym: round(float(merged[sym].iloc[-1]), 2) for sym in legs}

    rt_syms = {}
    for sym in legs:
        if sym in rt_data:
            rt_syms[sym] = rt_data[sym]["symbol"]

    return {
        "id": strategy_id,
        "name": strat["name"],
        "category": strat["category"],
        "unit": strat["unit"],
        "formula": formula,
        "legs": legs,
        "leg_prices": leg_prices,
        "rt_syms": rt_syms,
        "dates": merged["date"].dt.strftime("%Y-%m-%d").tolist(),
        "spread": safe(spread),
        "boll_mid": safe(boll_mid),
        "boll_up": safe(boll_up),
        "boll_dn": safe(boll_dn),
        # 各腿价格数据（用于副图）
        "legs_data": {sym: safe(merged[sym]) for sym in legs},
        "corr": corr,
        "stats": {
            "cur": round(cur_d, 2), "mean": round(mean_d, 2),
            "std": round(std_d, 2), "z": round(z_d, 3),
            "pct": round(pct_d, 1),
            "p5": round(float(np.percentile(spread.dropna(), 5)), 2),
            "p95": round(float(np.percentile(spread.dropna(), 95)), 2),
        },
        "thresholds": {
            "open_hi": strat["open_hi"], "strong_hi": strat["strong_hi"],
            "open_lo": strat["open_lo"], "strong_lo": strat["strong_lo"],
            "tp": strat["tp"], "sl": strat["sl"],
        },
        "note": strat["note"],
        "last_date": merged["date"].iloc[-1].strftime("%Y-%m-%d"),
        "count": len(merged),
        "realtime": realtime_updated,
    }


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


@app.get("/api/strategies")
def get_strategies():
    """返回所有策略列表"""
    return [{"id": s["id"], "name": s["name"], "category": s["category"]} for s in STRATEGIES]


@app.get("/api/spread")
def get_spread(id: str, years: int = 3):
    """通用多腿价差计算接口"""
    try:
        return compute_spread(id, years)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ===== 跨设备设置同步 =====
import json as _json_sync

_SETTINGS_FILE = "/tmp/brnr_settings.json"
_settings_mem: dict = {}  # 内存缓存，重启丢失但至少当次会话有效

def _load_settings() -> dict:
    # 1. 优先内存
    if _settings_mem:
        return dict(_settings_mem)
    # 2. 从环境变量读（持久化）
    raw = os.environ.get("BRNR_SETTINGS", "")
    if raw:
        try:
            data = _json_sync.loads(raw)
            _settings_mem.update(data)
            return data
        except Exception:
            pass
    # 3. 从文件读（临时）
    try:
        with open(_SETTINGS_FILE) as f:
            data = _json_sync.load(f)
            _settings_mem.update(data)
            return data
    except Exception:
        return {}

def _save_settings(data: dict):
    _settings_mem.clear()
    _settings_mem.update(data)
    # 写文件（临时，重启丢失）
    try:
        with open(_SETTINGS_FILE, "w") as f:
            _json_sync.dump(data, f)
    except Exception:
        pass
    # 尝试通过 Render API 写环境变量（持久化）
    render_api_key = os.environ.get("RENDER_API_KEY", "")
    render_service_id = os.environ.get("RENDER_SERVICE_ID", "")
    if render_api_key and render_service_id:
        try:
            import urllib.request as _ur
            payload = _json_sync.dumps({"key": "BRNR_SETTINGS", "value": _json_sync.dumps(data)}).encode()
            req = _ur.Request(
                f"https://api.render.com/v1/services/{render_service_id}/env-vars",
                data=_json_sync.dumps([{"key": "BRNR_SETTINGS", "value": _json_sync.dumps(data)}]).encode(),
                headers={"Authorization": f"Bearer {render_api_key}", "Content-Type": "application/json"},
                method="PUT"
            )
            with _ur.urlopen(req, timeout=5):
                pass
        except Exception:
            pass

@app.get("/api/settings")
def get_settings(key: str = "default"):
    """获取用户设置（跨设备同步）"""
    data = _load_settings()
    return data.get(key, {})

class SettingsReq(BaseModel):
    key: str = "default"
    thresh: dict = {}
    alert: dict = {}

@app.post("/api/settings")
def save_settings(req: SettingsReq):
    """保存用户设置（跨设备同步）"""
    data = _load_settings()
    data[req.key] = {"thresh": req.thresh, "alert": req.alert, "ts": time.time()}
    _save_settings(data)
    return {"ok": True}


@app.get("/api/debug")
def debug_akshare():
    """诊断接口：测试akshare各接口可达性"""
    import traceback, time as _t
    results = {}
    
    # 测试历史数据
    try:
        t0 = _t.time()
        import akshare as ak
        df = ak.futures_main_sina(symbol="BR0")
        results["history_BR"] = {"ok": True, "rows": len(df), "ms": round((_t.time()-t0)*1000)}
    except Exception as e:
        results["history_BR"] = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-300:]}
    
    # 测试实时数据
    try:
        t0 = _t.time()
        df2 = ak.futures_zh_realtime(symbol="丁二烯橡胶")
        results["realtime_BR"] = {"ok": True, "rows": len(df2), "ms": round((_t.time()-t0)*1000)}
    except Exception as e:
        results["realtime_BR"] = {"ok": False, "error": str(e)}
    
    return results
