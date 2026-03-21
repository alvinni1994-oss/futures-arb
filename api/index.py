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
CACHE_TTL = 3600  # 1小时缓存

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
