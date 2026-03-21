"""
多策略期货套利监控后端 API
========================
技术栈: FastAPI + akshare + pandas + numpy + scipy

策略覆盖:
  分类一: 股指期货价差 (IM-IC)
  分类二: 加工费套利 (TA加工费, 瓶片加工费, PR-PF相对加工费差)
  分类三: 跨品种套利 (L-PP, 菜粕菜油, 豆粕菜粕, 菜粕玉米, LU-FU)
  分类四: 跨期价差 (PP, BU, FU)

数据来源: akshare futures_main_sina(symbol="XX0")
注意: akshare返回中文列名，需rename后使用
"""

import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import akshare as ak
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from scipy import stats

app = FastAPI(title="多策略期货套利监控", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 缓存系统
# ============================================================

# 历史数据缓存: {symbol: {"data": df, "ts": timestamp}}
_history_cache: Dict[str, Dict] = {}
HISTORY_CACHE_TTL = 3600  # 1小时

# 实时数据缓存: {"data": dict, "ts": timestamp}
_realtime_cache: Dict = {}
REALTIME_CACHE_TTL = 30  # 30秒

# 跨期数据缓存
_calendar_cache: Dict[str, Dict] = {}
CALENDAR_CACHE_TTL = 60  # 1分钟


# ============================================================
# 策略定义
# ============================================================

STRATEGIES = [
    # ---------- 分类一：股指期货价差 ----------
    {
        "id": "im_ic",
        "name": "IM-IC价差",
        "category": "equity_index",
        "category_name": "股指期货",
        "formula": "IM - IC",
        "description": "中证1000(IM) vs 中证500(IC) 价差套利",
        "symbols": ["IM0", "IC0"],
        "unit": "点",
        "signal_thresholds": {"high": 1000, "low": -500},
        "legs": [
            {"symbol": "IM0", "coef": 1.0, "label": "IM"},
            {"symbol": "IC0", "coef": -1.0, "label": "IC"},
        ],
    },
    # ---------- 分类二：加工费套利 ----------
    {
        "id": "ta_proc",
        "name": "TA加工费",
        "category": "processing",
        "category_name": "加工费套利",
        "formula": "TA - PX × 0.655",
        "description": "PTA加工费：多3手TA空2手PX。加工费低时买入，高时卖出。",
        "symbols": ["TA0", "PX0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 800, "low": 200},
        "legs": [
            {"symbol": "TA0", "coef": 1.0, "label": "TA"},
            {"symbol": "PX0", "coef": -0.655, "label": "PX"},
        ],
    },
    {
        "id": "bottle_proc",
        "name": "瓶片加工费",
        "category": "processing",
        "category_name": "加工费套利",
        "formula": "PR - TA × 0.855 - EG × 0.335",
        "description": "瓶片加工费：空2手PR多5手TA多1手EG。加工费高时做空。",
        "symbols": ["PR0", "TA0", "EG0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 600, "low": 100},
        "legs": [
            {"symbol": "PR0", "coef": 1.0, "label": "PR"},
            {"symbol": "TA0", "coef": -0.855, "label": "TA"},
            {"symbol": "EG0", "coef": -0.335, "label": "EG"},
        ],
    },
    {
        "id": "pr_pf_spread",
        "name": "PR-PF相对加工费差",
        "category": "processing",
        "category_name": "加工费套利",
        # 相对加工费差 = (PR - TA*0.855 - EG*0.332) - (PF - TA*0.855 - EG*0.335)
        # = PR - PF - EG*0.332 + EG*0.335 = PR - PF + EG*0.003
        # 简化展示: PR - SF(PF) - EG*(0.332-0.335) = PR - SF + 0.003*EG
        "formula": "(PR - TA×0.855 - EG×0.332) - (PF - TA×0.855 - EG×0.335)",
        "description": (
            "PR加工费 - PF加工费。正常区间800-1200元/吨。\n"
            ">1500 PR偏强(空1PR多3PF)；<500 PF偏强(多1PR空3PF)。\n"
            "注意：PF的akshare代码为SF0"
        ),
        "symbols": ["PR0", "SF0", "EG0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 1500, "low": 500},
        "normal_range": [800, 1200],
        # PR加工费 - PF加工费
        # = PR - TA*0.855 - EG*0.332 - (SF - TA*0.855 - EG*0.335)
        # = PR - SF + EG*(0.335 - 0.332) = PR - SF + EG*0.003
        "legs": [
            {"symbol": "PR0", "coef": 1.0, "label": "PR"},
            {"symbol": "SF0", "coef": -1.0, "label": "PF(SF)"},
            {"symbol": "EG0", "coef": 0.003, "label": "EG"},
        ],
    },
    # ---------- 分类三：跨品种套利 ----------
    {
        "id": "l_pp",
        "name": "L-PP价差",
        "category": "cross_product",
        "category_name": "跨品种套利",
        "formula": "L - PP",
        "description": "塑料(L) vs 聚丙烯(PP) 价差套利，1:1比例。",
        "symbols": ["L0", "PP0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 400, "low": -200},
        "legs": [
            {"symbol": "L0", "coef": 1.0, "label": "L"},
            {"symbol": "PP0", "coef": -1.0, "label": "PP"},
        ],
    },
    {
        "id": "rm_oi",
        "name": "菜粕菜油价差(4RM-OI)",
        "category": "cross_product",
        "category_name": "跨品种套利",
        "formula": "RM×4 - OI×1",
        "description": (
            "多4手菜粕(RM)空1手菜油(OI)，压榨利润套利。\n"
            "菜粕/菜油均为10吨/手。"
        ),
        "symbols": ["RM0", "OI0"],
        "unit": "元（加权）",
        "signal_thresholds": {"high": 20000, "low": 5000},
        "legs": [
            {"symbol": "RM0", "coef": 4.0, "label": "RM×4"},
            {"symbol": "OI0", "coef": -1.0, "label": "OI"},
        ],
    },
    {
        "id": "m_rm",
        "name": "豆粕-菜粕(M-RM)",
        "category": "cross_product",
        "category_name": "跨品种套利",
        "formula": "M - RM",
        "description": "豆粕(M) vs 菜粕(RM) 价差套利，1:1比例。",
        "symbols": ["M0", "RM0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 500, "low": 50},
        "legs": [
            {"symbol": "M0", "coef": 1.0, "label": "M"},
            {"symbol": "RM0", "coef": -1.0, "label": "RM"},
        ],
    },
    {
        "id": "rm_c",
        "name": "菜粕-玉米(RM-C)",
        "category": "cross_product",
        "category_name": "跨品种套利",
        "formula": "RM - C",
        "description": "菜粕(RM) vs 玉米(C) 价差套利，1:1比例。",
        "symbols": ["RM0", "C0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 1500, "low": 300},
        "legs": [
            {"symbol": "RM0", "coef": 1.0, "label": "RM"},
            {"symbol": "C0", "coef": -1.0, "label": "C"},
        ],
    },
    {
        "id": "lu_fu",
        "name": "LU-FU(低硫-燃料油)",
        "category": "cross_product",
        "category_name": "跨品种套利",
        "formula": "LU - FU",
        "description": "低硫燃油(LU) vs 燃料油(FU) 价差套利，1:1比例。",
        "symbols": ["LU0", "FU0"],
        "unit": "元/吨",
        "signal_thresholds": {"high": 2000, "low": 200},
        "legs": [
            {"symbol": "LU0", "coef": 1.0, "label": "LU"},
            {"symbol": "FU0", "coef": -1.0, "label": "FU"},
        ],
    },
    # ---------- 分类四：跨期价差 ----------
    {
        "id": "pp_calendar",
        "name": "PP跨期价差",
        "category": "calendar",
        "category_name": "跨期价差",
        "formula": "PP近月 - PP次近月",
        "description": "聚丙烯(PP)跨期价差，通过实时各月合约报价计算。",
        "symbols": ["PP0"],
        "unit": "元/吨",
        "is_calendar": True,
        "calendar_symbol": "PP",
    },
    {
        "id": "bu_calendar",
        "name": "BU跨期价差",
        "category": "calendar",
        "category_name": "跨期价差",
        "formula": "BU近月 - BU次近月",
        "description": "沥青(BU)跨期价差，通过实时各月合约报价计算。",
        "symbols": ["BU0"],
        "unit": "元/吨",
        "is_calendar": True,
        "calendar_symbol": "BU",
    },
    {
        "id": "fu_calendar",
        "name": "FU跨期价差",
        "category": "calendar",
        "category_name": "跨期价差",
        "formula": "FU近月 - FU次近月",
        "description": "燃料油(FU)跨期价差，通过实时各月合约报价计算。",
        "symbols": ["FU0"],
        "unit": "元/吨",
        "is_calendar": True,
        "calendar_symbol": "FU",
    },
]

# 建立id -> strategy 映射
STRATEGY_MAP = {s["id"]: s for s in STRATEGIES}


# ============================================================
# 数据获取工具函数
# ============================================================

def fetch_history(symbol: str, years: int = 3) -> pd.DataFrame:
    """
    获取主力合约历史数据（带1小时缓存）。
    akshare返回中文列名，这里统一rename为英文。
    """
    cache_key = f"{symbol}_{years}"
    now = time.time()

    if cache_key in _history_cache:
        cached = _history_cache[cache_key]
        if now - cached["ts"] < HISTORY_CACHE_TTL:
            return cached["data"].copy()

    try:
        df = ak.futures_main_sina(symbol=symbol)
        # akshare返回中文列名：日期/开盘价/最高价/最低价/收盘价/成交量/持仓量/动态结算价
        df = df.rename(columns={
            "日期": "date",
            "开盘价": "open",
            "最高价": "high",
            "最低价": "low",
            "收盘价": "close",
            "成交量": "volume",
            "持仓量": "open_interest",
            "动态结算价": "settle",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 只保留最近N年
        cutoff = datetime.now() - timedelta(days=years * 365)
        df = df[df["date"] >= cutoff].reset_index(drop=True)

        # 保证close是数值
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        _history_cache[cache_key] = {"data": df, "ts": now}
        return df.copy()

    except Exception as e:
        raise RuntimeError(f"akshare获取{symbol}历史数据失败: {e}")


def fetch_realtime_all() -> Dict[str, float]:
    """
    获取所有品种实时价格（带30秒缓存）。
    使用 ak.futures_zh_realtime 或逐个品种fallback。
    """
    now = time.time()
    if _realtime_cache and now - _realtime_cache.get("ts", 0) < REALTIME_CACHE_TTL:
        return _realtime_cache.get("data", {})

    result: Dict[str, float] = {}

    # 收集所有需要的主力合约代码
    needed_symbols = set()
    for s in STRATEGIES:
        if not s.get("is_calendar"):
            for leg in s.get("legs", []):
                needed_symbols.add(leg["symbol"])

    # 尝试逐个获取最新收盘价
    for sym in needed_symbols:
        try:
            df = ak.futures_main_sina(symbol=sym)
            df = df.rename(columns={"收盘价": "close", "日期": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            last = df["close"].dropna().iloc[-1]
            result[sym] = float(last)
        except Exception:
            pass

    _realtime_cache["data"] = result
    _realtime_cache["ts"] = now
    return result


def compute_spread_series(legs: List[Dict], dfs: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    根据腿定义计算价差序列。
    legs: [{"symbol": "TA0", "coef": 1.0}, ...]
    dfs: {symbol: df with date/close}
    """
    # 以第一条腿为基准，inner join其余
    base_sym = legs[0]["symbol"]
    merged = dfs[base_sym][["date", "close"]].rename(columns={"close": base_sym})

    for leg in legs[1:]:
        sym = leg["symbol"]
        if sym not in dfs:
            raise ValueError(f"缺少品种数据: {sym}")
        tmp = dfs[sym][["date", "close"]].rename(columns={"close": sym})
        merged = merged.merge(tmp, on="date", how="inner")

    # 计算加权价差
    spread = sum(leg["coef"] * merged[leg["symbol"]] for leg in legs)
    spread.index = merged["date"]
    return spread


def compute_statistics(spread: pd.Series) -> Dict:
    """
    计算价差统计指标：
    - 当前值
    - 历史分位 (percentile)
    - Z-Score
    - 布林带 (20日, 2σ)
    - 均值/标准差
    """
    arr = spread.dropna().values
    if len(arr) == 0:
        return {}

    current = float(arr[-1])
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    percentile = float(stats.percentileofscore(arr, current))
    zscore = float((current - mean) / std) if std > 0 else 0.0

    # 布林带 (20日移动平均 ± 2σ)
    window = min(20, len(arr))
    roll_mean = pd.Series(arr).rolling(window).mean()
    roll_std = pd.Series(arr).rolling(window).std()
    bb_upper = (roll_mean + 2 * roll_std).tolist()
    bb_lower = (roll_mean - 2 * roll_std).tolist()
    bb_mid = roll_mean.tolist()

    # 操作信号
    if percentile >= 80:
        signal = "偏高"
        signal_color = "red"
    elif percentile <= 20:
        signal = "偏低"
        signal_color = "green"
    else:
        signal = "中性"
        signal_color = "gray"

    return {
        "current": round(current, 2),
        "mean": round(mean, 2),
        "std": round(std, 2),
        "percentile": round(percentile, 1),
        "zscore": round(zscore, 3),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "signal": signal,
        "signal_color": signal_color,
        "bb_upper": [round(x, 2) if x is not None and not np.isnan(x) else None for x in bb_upper],
        "bb_lower": [round(x, 2) if x is not None and not np.isnan(x) else None for x in bb_lower],
        "bb_mid": [round(x, 2) if x is not None and not np.isnan(x) else None for x in bb_mid],
    }


def fetch_calendar_spread(symbol: str, years: int = 2) -> Dict:
    """
    获取跨期价差数据。
    尝试用 ak.futures_zh_realtime 获取各月合约实时报价，
    并用主力合约历史数据辅助展示。
    """
    cache_key = f"cal_{symbol}_{years}"
    now = time.time()

    if cache_key in _calendar_cache:
        cached = _calendar_cache[cache_key]
        if now - cached["ts"] < CALENDAR_CACHE_TTL:
            return cached["data"]

    result = {
        "symbol": symbol,
        "realtime_spreads": [],
        "history_note": "",
        "supported": False,
        "error": None,
    }

    # 交易所映射
    exchange_map = {
        "PP": "大商所",
        "BU": "上期所",
        "FU": "上期所",
    }
    exchange = exchange_map.get(symbol, "上期所")

    try:
        # 获取实时各月合约报价
        df_rt = ak.futures_zh_realtime(symbol=exchange)
        # akshare返回列: 名称/最新价/涨跌额/涨跌幅/买价/卖价/持仓量/成交量
        # 过滤出目标品种的合约（合约代码以symbol开头）
        # 列名可能是中文，先检查
        if "名称" in df_rt.columns:
            name_col = "名称"
        elif "symbol" in df_rt.columns:
            name_col = "symbol"
        else:
            name_col = df_rt.columns[0]

        # 根据名称过滤（如 "PP2501", "PP2505" 等）
        mask = df_rt[name_col].str.startswith(symbol, na=False)
        df_sym = df_rt[mask].copy()

        if len(df_sym) >= 2:
            # 尝试获取最新价列
            price_col = None
            for col in ["最新价", "现价", "price", "last"]:
                if col in df_sym.columns:
                    price_col = col
                    break
            if price_col is None:
                price_col = df_sym.columns[1]  # 默认取第二列

            df_sym = df_sym.sort_values(name_col).reset_index(drop=True)
            df_sym[price_col] = pd.to_numeric(df_sym[price_col], errors="coerce")
            df_sym = df_sym.dropna(subset=[price_col])

            contracts = df_sym[[name_col, price_col]].values.tolist()
            spreads = []
            for i in range(len(contracts) - 1):
                near_name, near_price = contracts[i]
                far_name, far_price = contracts[i + 1]
                spread_val = round(float(near_price) - float(far_price), 2)
                spreads.append({
                    "near": str(near_name),
                    "far": str(far_name),
                    "near_price": round(float(near_price), 2),
                    "far_price": round(float(far_price), 2),
                    "spread": spread_val,
                    "label": f"{near_name}-{far_name}",
                })

            result["realtime_spreads"] = spreads
            result["supported"] = True
            result["contracts"] = [
                {"name": str(c[0]), "price": round(float(c[1]), 2)}
                for c in contracts
            ]
        else:
            result["supported"] = False
            result["error"] = f"未找到足够的{symbol}合约数据（当前交易时段可能暂无报价）"

    except Exception as e:
        result["supported"] = False
        result["error"] = f"获取{symbol}跨期数据失败: {str(e)[:200]}"

    _calendar_cache[cache_key] = {"data": result, "ts": now}
    return result


# ============================================================
# API 路由
# ============================================================

@app.get("/api/strategies")
def get_strategies():
    """返回所有策略列表（不含历史数据）"""
    output = []
    for s in STRATEGIES:
        item = {
            "id": s["id"],
            "name": s["name"],
            "category": s["category"],
            "category_name": s["category_name"],
            "formula": s["formula"],
            "description": s["description"],
            "unit": s["unit"],
            "is_calendar": s.get("is_calendar", False),
        }
        if "normal_range" in s:
            item["normal_range"] = s["normal_range"]
        output.append(item)
    return {"strategies": output}


@app.get("/api/spread")
def get_spread(
    id: str = Query(..., description="策略ID"),
    years: int = Query(3, ge=1, le=10, description="历史年数"),
):
    """
    返回指定策略的历史价差数据及统计指标。
    
    返回格式:
    {
      "id": "ta_proc",
      "name": "TA加工费",
      "dates": [...],
      "spreads": [...],
      "stats": {current, percentile, zscore, signal, bb_*...}
    }
    """
    if id not in STRATEGY_MAP:
        raise HTTPException(status_code=404, detail=f"策略不存在: {id}")

    strategy = STRATEGY_MAP[id]

    if strategy.get("is_calendar"):
        raise HTTPException(
            status_code=400,
            detail="跨期价差请使用 /api/calendar_spread 接口",
        )

    legs = strategy["legs"]

    # 收集所有品种的历史数据
    dfs: Dict[str, pd.DataFrame] = {}
    errors = []
    for leg in legs:
        sym = leg["symbol"]
        if sym not in dfs:
            try:
                dfs[sym] = fetch_history(sym, years)
            except Exception as e:
                errors.append(str(e))

    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))

    # 计算价差序列
    try:
        spread_series = compute_spread_series(legs, dfs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算价差失败: {e}")

    # 统计指标
    stats_dict = compute_statistics(spread_series)

    # 格式化日期和价差列表
    dates = [d.strftime("%Y-%m-%d") for d in spread_series.index]
    spreads = [round(float(v), 2) if not np.isnan(v) else None for v in spread_series.values]

    return {
        "id": id,
        "name": strategy["name"],
        "formula": strategy["formula"],
        "unit": strategy["unit"],
        "dates": dates,
        "spreads": spreads,
        "stats": stats_dict,
        "legs": [
            {"label": leg["label"], "coef": leg["coef"], "symbol": leg["symbol"]}
            for leg in legs
        ],
    }


@app.get("/api/realtime")
def get_realtime():
    """
    返回所有品种当前实时价格（最新主力合约收盘价）。
    同时计算每个策略的当前价差值。
    缓存30秒。
    """
    prices = fetch_realtime_all()

    # 计算各策略当前价差
    strategy_spreads = {}
    for s in STRATEGIES:
        if s.get("is_calendar"):
            continue
        try:
            legs = s["legs"]
            val = sum(
                leg["coef"] * prices[leg["symbol"]]
                for leg in legs
                if leg["symbol"] in prices
            )
            # 检查是否所有腿都有价格
            has_all = all(leg["symbol"] in prices for leg in legs)
            if has_all:
                strategy_spreads[s["id"]] = round(float(val), 2)
        except Exception:
            pass

    return {
        "prices": prices,
        "strategy_spreads": strategy_spreads,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/calendar_spread")
def get_calendar_spread(
    symbol: str = Query(..., description="品种代码，如PP/BU/FU"),
    years: int = Query(2, ge=1, le=5, description="历史参考年数"),
):
    """
    返回跨期价差数据（实时各月合约报价）。
    """
    valid_symbols = {"PP", "BU", "FU"}
    if symbol.upper() not in valid_symbols:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的品种: {symbol}，支持: {', '.join(valid_symbols)}",
        )

    data = fetch_calendar_spread(symbol.upper(), years)
    return data


@app.get("/api/health")
def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}


# ============================================================
# 静态文件服务（前端）
# ============================================================

import os

_public_dir = os.path.join(os.path.dirname(__file__), "..", "public")
if os.path.isdir(_public_dir):
    app.mount("/", StaticFiles(directory=_public_dir, html=True), name="static")
