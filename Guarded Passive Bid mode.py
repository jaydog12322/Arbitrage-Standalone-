#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Arbitrage / Guarded Passive Bid Backtester (with traceability)
- (A) Original LOCF arbitrage detector (percent/bps logic)
- (B) Guarded Passive Bid backtest using 주식체결(fid_10,fid_11)

UPGRADES:
- Input: Excel (.xlsx) OR Parquet (.parquet). Mixed stream supported:
    * real_type == '주식호가잔량' → fid_41/fid_51/fid_61/fid_71 (quotes)
    * real_type == '주식체결'   → fid_10 (±price; sign=side), fid_11 (size)
- Outputs:
    * Arbitrage detector → <stem>_arbitrage_windows_bps.{csv,parquet,xlsx}
    * Guarded Passive    → <stem>_guarded_passive.{csv,parquet,xlsx}
- VI(변동성완화장치) filter preserved; pre-open cutoff preserved (drop before 09:00:30).
- NEW traceability:
    * Input gets 'row_id' after sort.
    * Output rows include:
        entry_dt_iso / fill_dt_iso / hedge_dt_iso
        entry_row_id / fill_row_id / hedge_row_id

Dependencies:
    pip install PySide6 pandas numpy pyyaml openpyxl
    # Parquet:
    pip install pyarrow  # or: pip install fastparquet
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# GUI
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QCheckBox, QGridLayout, QMessageBox
)
from PySide6.QtCore import Qt

# Optional YAML config
try:
    import yaml
except Exception:
    yaml = None

# ---------- Parquet engine detection ----------
try:
    import pyarrow  # noqa: F401
    _PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        _PARQUET_ENGINE = None


def _ensure_parquet_engine_or_raise():
    if _PARQUET_ENGINE is None:
        raise RuntimeError(
            "Parquet engine not found. Please install one of:\n"
            "  pip install pyarrow\n"
            "     or\n"
            "  pip install fastparquet"
        )


# ---------------- Tick size -------------------

def get_tick_size(price: float) -> int:
    """KRX tick size schedule."""
    if price < 2000:
        return 1
    elif price < 5000:
        return 5
    elif price < 20000:
        return 10
    elif price < 50000:
        return 50
    elif price < 200000:
        return 100
    elif price < 500000:
        return 500
    else:
        return 1000


# ---------------- Data structures -------------

@dataclass
class QuoteSnapshot:
    krx_bid: int = 0
    krx_ask: int = 0
    krx_bid_size: int = 0
    krx_ask_size: int = 0
    nxt_bid: int = 0
    nxt_ask: int = 0
    nxt_bid_size: int = 0
    nxt_ask_size: int = 0


@dataclass
class PassiveState:
    active: bool = False           # have we armed?
    symbol: str = ""
    buy_venue: str = ""            # 'KRX' or 'NXT'
    sell_venue: str = ""           # opposite venue
    bid_price: int = 0             # price we rest at
    pre_existing: int = 0          # bid size present when we armed
    qty: int = 0                   # order size (0.1×pre_existing; ≥1)
    cum_hit_bid: int = 0           # cumulative prints hitting bid @<= bid_price
    arm_ts: Optional[pd.Timestamp] = None
    fill_ts: Optional[pd.Timestamp] = None
    hedge_due_ts: Optional[pd.Timestamp] = None  # = fill_ts + latency
    filled: bool = False
    # NEW: trace to input rows
    arm_row_id: Optional[int] = None
    fill_row_id: Optional[int] = None
    hedge_row_id: Optional[int] = None


# ---------------- Config ----------------------

def load_yaml_config(path: Optional[Path]) -> dict:
    """
    Load YAML config if provided; otherwise return defaults.

    vi_filter:
        enabled: true
        default_end_minutes: 3

    guarded_passive_bid:
        enabled: true
        hedge_latency_ms: 200
        order_size_frac_of_visible_bid: 0.1
    """
    default_cfg = {
        "fees": {
            "krx": {"commission_bps": 1.5, "sell_tax_bps": 20},
            "nxt": {"commission_bps": 1.5, "sell_tax_bps": 20},
        },
        "spread_engine": {
            "edge_rule": {
                "min_net_ticks_after_fees": 1,      # require ≥ 1 tick after fees
                "also_require_min_visible_qty": 1,  # both venues have at least this visible
            }
        },
        "vi_filter": {
            "enabled": True,
            "default_end_minutes": 3
        },
        "guarded_passive_bid": {
            "enabled": True,
            "hedge_latency_ms": 200,
            "order_size_frac_of_visible_bid": 0.1
        }
    }
    if path and yaml:
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}

            def merge(a, b):
                for k, v in b.items():
                    if isinstance(v, dict) and isinstance(a.get(k), dict):
                        merge(a[k], v)
                    else:
                        a[k] = v

            cfg = default_cfg.copy()
            merge(cfg, user_cfg)
            return cfg
        except Exception:
            return default_cfg
    return default_cfg


# ---------------- Helpers ---------------------

def is_quote_valid(q: QuoteSnapshot, min_visible: int) -> bool:
    krx_ok = (
        q.krx_bid > 0 and q.krx_ask > 0 and q.krx_ask > q.krx_bid and
        min(q.krx_bid_size, q.krx_ask_size) >= min_visible
    )
    nxt_ok = (
        q.nxt_bid > 0 and q.nxt_ask > 0 and q.nxt_ask > q.nxt_bid and
        min(q.nxt_bid_size, q.nxt_ask_size) >= min_visible
    )
    return krx_ok and nxt_ok


def calculate_direction_edge(
    symbol: str,
    buy_venue: str,
    buy_price: int,
    buy_size: int,
    sell_venue: str,
    sell_price: int,
    sell_size: int,
    fees_bps: Dict[str, Dict[str, float]],
) -> Optional[dict]:
    """
    Regular arbitrage edge (buy@ask, sell@bid). Fees in bps:
        buy_fee  = buy_price  * commission_bps / 10000
        sell_fee = sell_price * (commission_bps + sell_tax_bps) / 10000
    """
    if sell_price <= buy_price:
        return None
    gross_edge = sell_price - buy_price
    buy_fee_rate = fees_bps.get(buy_venue, {}).get("commission_bps", 0.0)
    sell_commission = fees_bps.get(sell_venue, {}).get("commission_bps", 0.0)
    sell_tax = fees_bps.get(sell_venue, {}).get("sell_tax_bps", 0.0)
    buy_fee = buy_price * buy_fee_rate / 10000.0
    sell_fee = sell_price * (sell_commission + sell_tax) / 10000.0
    total_fees = buy_fee + sell_fee
    net_edge = gross_edge - total_fees
    edge_bps = (net_edge / buy_price * 10000.0) if buy_price else 0.0
    max_qty = int(min(buy_size, sell_size))
    return {
        "symbol": symbol,
        "buy_venue": buy_venue, "sell_venue": sell_venue,
        "buy_price": buy_price, "sell_price": sell_price,
        "edge_krw": gross_edge, "total_fees_krw": total_fees,
        "net_edge_krw": net_edge, "edge_bps": edge_bps, "max_qty": max_qty
    }


def meets_tick_threshold(signal: dict, q: QuoteSnapshot, min_ticks_after_fees: int) -> bool:
    krx_mid = (q.krx_ask + q.krx_bid) / 2 if (q.krx_ask and q.krx_bid) else 0
    nxt_mid = (q.nxt_ask + q.nxt_bid) / 2 if (q.nxt_ask and q.nxt_bid) else 0
    ref_mid = (krx_mid + nxt_mid) / 2 if (krx_mid and nxt_mid) else (krx_mid or nxt_mid)
    tick = get_tick_size(ref_mid) if ref_mid else 10
    return signal["net_edge_krw"] >= tick * min_ticks_after_fees


# --------- I/O helpers (Excel OR Parquet in; Parquet out) ---------

def _read_input_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".parquet":
        _ensure_parquet_engine_or_raise()
        df = pd.read_parquet(path, engine=_PARQUET_ENGINE)
    else:
        raise RuntimeError(f"Unsupported input extension: {ext} (use .xlsx or .parquet)")

    # Required core columns
    core_required = ["timestamp", "symbol", "venue"]
    missing = [c for c in core_required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Normalize timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.replace("'", ""),
        errors="coerce",
        infer_datetime_format=True
    )
    df = df.dropna(subset=["timestamp", "symbol", "venue"]).copy()
    df["venue"] = df["venue"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()

    # Ensure columns exist; coerce numerics but DO NOT drop rows lacking quotes
    # NOTE: 체결가격 = fid_10, 체결량 = fid_15 (signed; sign determines side)
    for c in ["fid_41", "fid_51", "fid_61", "fid_71", "fid_10", "fid_15"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # real_type optional, but useful
    if "real_type" not in df.columns:
        df["real_type"] = ""

    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    # NEW: stable row id for traceability (0-based after sort)
    df["row_id"] = np.arange(len(df))
    return df


def _save_any_all_formats(
    df_raw: pd.DataFrame,
    df_out: pd.DataFrame,
    base_stem_path: Path,
    suffix_stem: str,         # e.g., "arbitrage_windows_bps" or "guarded_passive"
    write_xlsx: bool
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Save as CSV (legacy), Parquet (new), and optionally XLSX (legacy GUI checkbox).
    Returns (csv_path, parquet_path, xlsx_path_or_None)
    """
    out_csv = base_stem_path.with_name(f"{base_stem_path.name}_{suffix_stem}.csv")
    out_parq = base_stem_path.with_name(f"{base_stem_path.name}_{suffix_stem}.parquet")
    out_xlsx = base_stem_path.with_name(f"{base_stem_path.name}_{suffix_stem}.xlsx") if write_xlsx else None

    # CSV
    df_out.to_csv(out_csv, index=False)

    # Parquet
    _ensure_parquet_engine_or_raise()
    df_out.to_parquet(out_parq, index=False, engine=_PARQUET_ENGINE)

    # XLSX (optional) — ensure at least one visible sheet exists
    if out_xlsx:
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                wrote_any = False
                if df_raw is not None and (len(df_raw.columns) > 0):
                    df_raw.to_excel(w, index=False, sheet_name="data"); wrote_any = True
                if df_out is not None and (len(df_out.columns) > 0):
                    df_out.to_excel(w, index=False, sheet_name="results"); wrote_any = True
                if not wrote_any:
                    pd.DataFrame({"note": ["No data to write"]}).to_excel(
                        w, index=False, sheet_name="empty"
                    )
        except IndexError:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                pd.DataFrame({"note": ["Recovered: workbook had no visible sheets"]}).to_excel(
                    w, index=False, sheet_name="empty"
                )

    return out_csv, out_parq, out_xlsx


# ------------------------- VI filter utilities -------------------------

def _norm_name(s: str) -> str:
    return "".join(str(s).split()).lower()

def _norm_code(s: str) -> str:
    s = str(s).strip()
    if s.endswith(".0"):
        s = s[:-2]
    try:
        return f"{int(s):06d}"
    except Exception:
        return s

def _time_to_td(t) -> pd.Timedelta:
    if pd.isna(t):
        return pd.NaT
    if isinstance(t, pd.Timestamp):
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    if hasattr(t, "hour") and hasattr(t, "minute") and hasattr(t, "second"):
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    try:
        parts = str(t).strip().split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(float(parts[2]))
        return pd.Timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return pd.NaT

def _load_vi_intervals_any(path: Optional[Path], default_end_minutes: int) -> Optional[dict]:
    if not path:
        return None
    if not path.exists():
        raise RuntimeError(f"VI Trigger file not found: {path}")

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise RuntimeError(f"Unsupported VI Trigger file: {ext} (use .xlsx or .csv)")

    cols = {c: _norm_name(c) for c in df.columns}
    df = df.rename(columns=cols)

    code_col = next((c for c in df.columns if "종목코드" in c or c in ("code", "symbolcode", "ticker", "symbol_code")), None)
    if code_col is None and "symbol" in df.columns:
        code_col = "symbol"

    name_col = next((c for c in df.columns if " 종목명" in (" "+c) or c in ("name", "symbolname", "symbol_name")), None)
    if name_col is None and "종목명" in df.columns:
        name_col = "종목명"

    start_col = next((c for c in df.columns if "발동" in c and "시간" in c), None)
    if start_col is None:
        start_col = next((c for c in df.columns if c in ("vi_start", "start", "start_time", "starttime")), None)

    end_col = next((c for c in df.columns if "해지" in c and "시간" in c), None)
    if end_col is None:
        end_col = next((c for c in df.columns if c in ("vi_end", "end", "end_time", "endtime")), None)

    if start_col is None:
        raise RuntimeError("VI Trigger file: could not find start time column (e.g., '발동\\n시간' or 'vi_start').")

    by_code: Dict[str, List[Tuple[pd.Timedelta, pd.Timedelta]]] = {}
    by_name: Dict[str, List[Tuple[pd.Timedelta, pd.Timedelta]]] = {}
    default_td = pd.Timedelta(minutes=int(default_end_minutes))

    for r in df.itertuples(index=False):
        code_val = getattr(r, code_col) if code_col else None
        name_val = getattr(r, name_col) if name_col else None
        st_val = getattr(r, start_col)
        en_val = getattr(r, end_col) if end_col and hasattr(r, end_col) else None

        td_start = _time_to_td(st_val)
        td_end = _time_to_td(en_val) if en_val is not None else pd.NaT
        if pd.isna(td_start):
            continue
        if pd.isna(td_end):
            td_end = td_start + default_td

        if code_val is not None and not pd.isna(code_val):
            key = _norm_code(code_val)
            by_code.setdefault(key, []).append((td_start, td_end))
        if name_val is not None and not pd.isna(name_val):
            keyn = _norm_name(name_val)
            by_name.setdefault(keyn, []).append((td_start, td_end))

    for d in (by_code, by_name):
        for k in d:
            d[k].sort()

    return {"by_code": by_code, "by_name": by_name}


def _row_in_vi(symbol: str, ts: pd.Timestamp, vi_idx: dict) -> bool:
    if not vi_idx:
        return False
    tod = pd.Timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second) + pd.Timedelta(microseconds=ts.microsecond)
    s_sym = str(symbol).strip()

    s_code = _norm_code(s_sym)
    lst = []
    if vi_idx.get("by_code") and s_code in vi_idx["by_code"]:
        lst = vi_idx["by_code"][s_code]
    else:
        s_name = _norm_name(s_sym)
        if vi_idx.get("by_name") and s_name in vi_idx["by_name"]:
            lst = vi_idx["by_name"][s_name]

    if not lst:
        return False
    for st, en in lst:
        if st <= tod <= en:
            return True
    return False


# -------------------------- (A) Arbitrage detector (+ VI) --------------------------

def detect_arbitrage_locf_percent(
    input_path: Path,
    config_path: Optional[Path],
    write_xlsx: bool,
    vi_trigger_path: Optional[Path] = None
) -> Tuple[Path, Path, Optional[Path], int]:
    """Legacy detector: loads data, applies LOCF + bps logic, VI filtering, writes outputs next to input."""
    df = _read_input_any(input_path)
    cfg = load_yaml_config(config_path)

    fees_bps = {
        "KRX": {
            "commission_bps": float(cfg["fees"]["krx"]["commission_bps"]),
            "sell_tax_bps": float(cfg["fees"]["krx"].get("sell_tax_bps", 0.0)),
        },
        "NXT": {
            "commission_bps": float(cfg["fees"]["nxt"]["commission_bps"]),
            "sell_tax_bps": float(cfg["fees"]["nxt"].get("sell_tax_bps", 0.0)),
        },
    }
    min_ticks = int(cfg["spread_engine"]["edge_rule"]["min_net_ticks_after_fees"])
    min_visible = int(cfg["spread_engine"]["edge_rule"]["also_require_min_visible_qty"])

    vi_cfg_enabled = bool(cfg.get("vi_filter", {}).get("enabled", True))
    vi_default_end_min = int(cfg.get("vi_filter", {}).get("default_end_minutes", 3))
    vi_index = None
    if vi_cfg_enabled and vi_trigger_path:
        vi_index = _load_vi_intervals_any(vi_trigger_path, default_end_minutes=vi_default_end_min)

    base_stem = input_path.with_suffix("")

    snapshots: Dict[str, QuoteSnapshot] = {}
    opps: List[dict] = []

    for row in df.itertuples(index=False):
        # only quotes update snapshot
        if getattr(row, "real_type", "") != "주식호가잔량":
            continue

        symbol = str(row.symbol)
        venue = str(row.venue)

        snap = snapshots.get(symbol, QuoteSnapshot())
        ask = int(getattr(row, "fid_41") or 0)
        bid = int(getattr(row, "fid_51") or 0)
        ask_sz = int(getattr(row, "fid_61") or 0)
        bid_sz = int(getattr(row, "fid_71") or 0)

        if venue == "KRX":
            snap.krx_ask, snap.krx_bid = ask, bid
            snap.krx_ask_size, snap.krx_bid_size = ask_sz, bid_sz
        else:
            snap.nxt_ask, snap.nxt_bid = ask, bid
            snap.nxt_ask_size, snap.nxt_bid_size = ask_sz, bid_sz

        snapshots[symbol] = snap

        if not is_quote_valid(snap, min_visible):
            continue

        s1 = calculate_direction_edge(symbol, "KRX", snap.krx_ask, snap.krx_ask_size,
                                      "NXT", snap.nxt_bid, snap.nxt_bid_size, fees_bps)
        s2 = calculate_direction_edge(symbol, "NXT", snap.nxt_ask, snap.nxt_ask_size,
                                      "KRX", snap.krx_bid, snap.krx_bid_size, fees_bps)
        cands = [s for s in (s1, s2) if s]
        if not cands:
            continue

        best = max(cands, key=lambda s: s["net_edge_krw"])
        if not meets_tick_threshold(best, snap, min_ticks):
            continue

        ts_row = getattr(row, "timestamp")
        tod = pd.Timedelta(hours=ts_row.hour, minutes=ts_row.minute, seconds=ts_row.second,
                           microseconds=ts_row.microsecond)
        if tod < pd.Timedelta(hours=9, minutes=0, seconds=30):
            continue

        if vi_index and _row_in_vi(symbol, ts_row, vi_index):
            continue

        record = row._asdict() | {
            "buy_venue": best["buy_venue"],
            "sell_venue": best["sell_venue"],
            "buy_price": best["buy_price"],
            "sell_price": best["sell_price"],
            "edge_krw": best["edge_krw"],
            "total_fees_krw": best["total_fees_krw"],
            "net_edge_krw": best["net_edge_krw"],
            "edge_bps": best["edge_bps"],
            "max_qty": best["max_qty"],
        }
        opps.append(record)

    opp_df = pd.DataFrame(opps, columns=list(df.columns) + [
        "buy_venue", "sell_venue", "buy_price", "sell_price",
        "edge_krw", "total_fees_krw", "net_edge_krw", "edge_bps", "max_qty"
    ])

    out_csv, out_parq, out_xlsx = _save_any_all_formats(
        df_raw=df,
        df_out=opp_df,
        base_stem_path=base_stem,
        suffix_stem="arbitrage_windows_bps",
        write_xlsx=write_xlsx
    )

    return out_csv, out_parq, out_xlsx, len(opp_df)


# -------------- (B) Guarded Passive Bid backtest (체결 기반) --------------

def backtest_guarded_passive_bid(
    input_path: Path,
    config_path: Optional[Path],
    write_xlsx: bool,
    vi_trigger_path: Optional[Path] = None
) -> Tuple[Path, Path, Optional[Path], int]:
    """
    Guarded Passive Bid backtest:
    - Arm when regular edge >= threshold (and tick gate passes).
    - Rest a bid at top of book on the buy venue; qty = floor(0.1 * bid_size_at_arm). min 1.
    - Fill rule: cumulative 'hit-bid' prints at or below our bid price since arm
                 >= pre_existing + qty.
    - Hedge after latency (ms). If edge still open → HEDGED; else → ESCAPE. Either way we sell
      immediately at opposite venue best bid.

    Returns output paths and number of completed trades.
    """
    df = _read_input_any(input_path)
    cfg = load_yaml_config(config_path)

    fees_bps = {
        "KRX": {
            "commission_bps": float(cfg["fees"]["krx"]["commission_bps"]),
            "sell_tax_bps": float(cfg["fees"]["krx"].get("sell_tax_bps", 0.0)),
        },
        "NXT": {
            "commission_bps": float(cfg["fees"]["nxt"]["commission_bps"]),
            "sell_tax_bps": float(cfg["fees"]["nxt"].get("sell_tax_bps", 0.0)),
        },
    }
    min_ticks = int(cfg["spread_engine"]["edge_rule"]["min_net_ticks_after_fees"])
    min_visible = int(cfg["spread_engine"]["edge_rule"]["also_require_min_visible_qty"])

    gp_cfg = cfg.get("guarded_passive_bid", {})
    latency_ms = int(gp_cfg.get("hedge_latency_ms", 200))
    frac = float(gp_cfg.get("order_size_frac_of_visible_bid", 0.1))
    frac = min(max(frac, 0.0), 1.0)

    vi_cfg_enabled = bool(cfg.get("vi_filter", {}).get("enabled", True))
    vi_default_end_min = int(cfg.get("vi_filter", {}).get("default_end_minutes", 3))
    vi_index = None
    if vi_cfg_enabled and vi_trigger_path:
        vi_index = _load_vi_intervals_any(vi_trigger_path, default_end_minutes=vi_default_end_min)

    base_stem = input_path.with_suffix("")

    snapshots: Dict[str, QuoteSnapshot] = {}
    # one state per symbol (only one passive order at a time per symbol)
    states: Dict[str, PassiveState] = {}

    trades: List[dict] = []

    def _valid_time_and_vi(ts: pd.Timestamp, sym: str) -> bool:
        tod = pd.Timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second, microseconds=ts.microsecond)
        if tod < pd.Timedelta(hours=9, minutes=0, seconds=30):
            return False
        if vi_index and _row_in_vi(sym, ts, vi_index):
            return False
        return True

    for row in df.itertuples(index=False):
        ts = getattr(row, "timestamp")
        sym = str(row.symbol)
        venue = str(row.venue)
        rtype = str(getattr(row, "real_type", ""))

        # 1) Handle quote updates (update snapshots, potentially ARM / CANCEL, or HEDGE if due)
        if rtype == "주식호가잔량":
            snap = snapshots.get(sym, QuoteSnapshot())
            ask = int(getattr(row, "fid_41") or 0)
            bid = int(getattr(row, "fid_51") or 0)
            ask_sz = int(getattr(row, "fid_61") or 0)
            bid_sz = int(getattr(row, "fid_71") or 0)

            if venue == "KRX":
                snap.krx_ask, snap.krx_bid = ask, bid
                snap.krx_ask_size, snap.krx_bid_size = ask_sz, bid_sz
            else:
                snap.nxt_ask, snap.nxt_bid = ask, bid
                snap.nxt_ask_size, snap.nxt_bid_size = ask_sz, bid_sz
            snapshots[sym] = snap

            # Try hedge if we have a filled order waiting for latency
            st = states.get(sym)
            if st and st.active and st.filled and st.hedge_due_ts and ts >= st.hedge_due_ts:
                # We sell on opposite venue best bid immediately
                if st.sell_venue == "KRX":
                    sell_px = snapshots[sym].krx_bid
                else:
                    sell_px = snapshots[sym].nxt_bid

                # If sell px not available, we postpone until next quote arrives
                if sell_px > 0:
                    # Evaluate whether edge is still open (for reason label); hedge happens anyway
                    s1 = calculate_direction_edge(sym, "KRX", snap.krx_ask, snap.krx_ask_size,
                                                  "NXT", snap.nxt_bid, snap.nxt_bid_size, fees_bps)
                    s2 = calculate_direction_edge(sym, "NXT", snap.nxt_ask, snap.nxt_ask_size,
                                                  "KRX", snap.krx_bid, snap.krx_bid_size, fees_bps)
                    cands = [s for s in (s1, s2) if s]
                    still_open = False
                    if cands:
                        best_now = max(cands, key=lambda s: s["net_edge_krw"])
                        still_open = meets_tick_threshold(best_now, snap, min_ticks)

                    # Compute PnL (fees included)
                    buy_fee_rate = fees_bps[st.buy_venue]["commission_bps"]
                    sell_comm = fees_bps[st.sell_venue]["commission_bps"]
                    sell_tax = fees_bps[st.sell_venue]["sell_tax_bps"]

                    buy_fee = st.bid_price * buy_fee_rate / 10000.0
                    sell_fee = sell_px * (sell_comm + sell_tax) / 10000.0

                    pnl_per_share = (sell_px - st.bid_price) - (buy_fee + sell_fee)
                    pnl_total = pnl_per_share * st.qty
                    pnl_bps = (pnl_per_share / st.bid_price * 10000.0) if st.bid_price else 0.0

                    st.hedge_row_id = getattr(row, "row_id")

                    trades.append({
                        "symbol": st.symbol,
                        "buy_venue": st.buy_venue,
                        "sell_venue": st.sell_venue,
                        "entry_ts": st.arm_ts,
                        "fill_ts": st.fill_ts,
                        "hedge_ts": ts,
                        # ISO datetimes for unambiguous tracing
                        "entry_dt_iso": st.arm_ts.isoformat() if st.arm_ts is not None else None,
                        "fill_dt_iso": st.fill_ts.isoformat() if st.fill_ts is not None else None,
                        "hedge_dt_iso": ts.isoformat() if ts is not None else None,
                        # row pointers back to input
                        "entry_row_id": st.arm_row_id,
                        "fill_row_id": st.fill_row_id,
                        "hedge_row_id": st.hedge_row_id,
                        "entry_price": st.bid_price,
                        "hedge_price": sell_px,
                        "qty": st.qty,
                        "reason": "HEDGED" if still_open else "ESCAPE",
                        "net_pnl_krw": pnl_total,
                        "net_pnl_bps": pnl_bps
                    })

                    # reset state
                    states[sym] = PassiveState()  # inactive

            # Consider (ARM / CANCEL) only if quotes are valid, time≥09:00:30, and not in VI
            if not _valid_time_and_vi(ts, sym):
                continue
            if not is_quote_valid(snap, min_visible):
                continue

            # Compute regular edges to decide whether to ARM/CANCEL
            s1 = calculate_direction_edge(sym, "KRX", snap.krx_ask, snap.krx_ask_size,
                                          "NXT", snap.nxt_bid, snap.nxt_bid_size, fees_bps)
            s2 = calculate_direction_edge(sym, "NXT", snap.nxt_ask, snap.nxt_ask_size,
                                          "KRX", snap.krx_bid, snap.krx_bid_size, fees_bps)
            cands = [s for s in (s1, s2) if s]
            if not cands:
                continue
            best = max(cands, key=lambda s: s["net_edge_krw"])
            window_open = meets_tick_threshold(best, snap, min_ticks)

            st = states.get(sym)
            if not st or not st.active:
                # ARM if window is open
                if window_open:
                    buy_v = best["buy_venue"]
                    sell_v = best["sell_venue"]
                    if buy_v == "KRX":
                        bid_px = snap.krx_bid
                        bid_sz = snap.krx_bid_size
                    else:
                        bid_px = snap.nxt_bid
                        bid_sz = snap.nxt_bid_size
                    if bid_px > 0 and bid_sz > 0:
                        qty = max(1, int(np.floor(bid_sz * frac)))
                        st = PassiveState(
                            active=True, symbol=sym,
                            buy_venue=buy_v, sell_venue=sell_v,
                            bid_price=bid_px, pre_existing=bid_sz,
                            qty=qty, cum_hit_bid=0, arm_ts=ts,
                            arm_row_id=getattr(row, "row_id")  # NEW: entry pointer
                        )
                        states[sym] = st
                # else: stay idle
            else:
                # Already ARMED (but not yet filled): CANCEL if window closed
                if st.active and not st.filled and not window_open:
                    states[sym] = PassiveState()  # cancel → reset

            continue  # quote rows handled; move on

        # 2) Handle trade prints (체결): update fill progress for ARMED state
        if rtype == "주식체결":
            st = states.get(sym)
            if not st or not st.active or st.filled:
                continue

            # Count only prints on our BUY venue
            if venue != st.buy_venue:
                continue

            # 체결가격 = fid_10 (unsigned price), 체결량 = fid_15 (signed)
            trade_px = int(getattr(row, "fid_10") or 0)
            qty_signed = getattr(row, "fid_15")
            if (trade_px <= 0) or pd.isna(qty_signed):
                continue
            qty_signed = int(qty_signed)
            if qty_signed == 0:
                continue

            # Current top-of-book on the buy venue
            snap = snapshots.get(sym, QuoteSnapshot())
            if st.buy_venue == "KRX":
                curr_bid, curr_ask = snap.krx_bid, snap.krx_ask
            else:
                curr_bid, curr_ask = snap.nxt_bid, snap.nxt_ask

            # New rule:
            #   qty_signed < 0  → hit bid (FID_51) at the matching price
            #   qty_signed > 0  → hit ask (FID_41)  (ignored for passive BID fills)
            if qty_signed < 0:
                # Require price to match current best bid (“matching price”)
                if curr_bid <= 0 or trade_px != int(curr_bid):
                    continue
                if st.bid_price <= 0:
                    continue

                st.cum_hit_bid += abs(qty_signed)

                # Filled when cumulative hit-bid >= pre-existing + our qty
                if st.cum_hit_bid >= (st.pre_existing + st.qty):
                    st.filled = True
                    st.fill_ts = ts
                    st.fill_row_id = getattr(row, "row_id")
                    st.hedge_due_ts = ts + pd.to_timedelta(int(max(0, latency_ms)), unit="ms")
                    states[sym] = st

            # qty_signed > 0 (hit ask) is irrelevant for passive BID fills
            continue

        # ignore other real_type rows if any

    # Build output DataFrame
    cols = [
        "symbol","buy_venue","sell_venue",
        "entry_ts","fill_ts","hedge_ts",
        "entry_dt_iso","fill_dt_iso","hedge_dt_iso",   # NEW
        "entry_row_id","fill_row_id","hedge_row_id",   # NEW
        "entry_price","hedge_price","qty","reason","net_pnl_krw","net_pnl_bps"
    ]
    trades_df = pd.DataFrame(trades, columns=cols)

    out_csv, out_parq, out_xlsx = _save_any_all_formats(
        df_raw=df,
        df_out=trades_df,
        base_stem_path=base_stem,
        suffix_stem="guarded_passive",
        write_xlsx=write_xlsx
    )

    return out_csv, out_parq, out_xlsx, len(trades_df)


# ------------------------------- GUI ----------------------------------

class ArbGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arbitrage / Guarded Passive Bid Backtester")

        self.input_edit = QLineEdit()
        self.input_browse = QPushButton("Browse…")
        self.input_browse.clicked.connect(self.pick_input)

        self.config_edit = QLineEdit()
        self.config_browse = QPushButton("Config… (optional)")
        self.config_browse.clicked.connect(self.pick_config)

        # VI Trigger file
        self.vi_edit = QLineEdit()
        self.vi_browse = QPushButton("VI Trigger… (optional)")
        self.vi_browse.clicked.connect(self.pick_vi)

        self.write_xlsx = QCheckBox("Also write XLSX")
        self.write_xlsx.setChecked(True)

        # Mode toggle
        self.mode_guarded = QCheckBox("Run Guarded Passive Bid backtest (체결 기반)")
        self.mode_guarded.setChecked(True)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_detector)

        layout = QGridLayout(self)
        r = 0
        layout.addWidget(QLabel("Input Excel/Parquet (.xlsx / .parquet):"), r, 0)
        layout.addWidget(self.input_edit, r, 1)
        layout.addWidget(self.input_browse, r, 2); r += 1

        layout.addWidget(QLabel("Config YAML (fees/thresholds):"), r, 0)
        layout.addWidget(self.config_edit, r, 1)
        layout.addWidget(self.config_browse, r, 2); r += 1

        layout.addWidget(QLabel("VI Trigger file (.xlsx / .csv):"), r, 0)
        layout.addWidget(self.vi_edit, r, 1)
        layout.addWidget(self.vi_browse, r, 2); r += 1

        layout.addWidget(self.write_xlsx, r, 0, 1, 3); r += 1
        layout.addWidget(self.mode_guarded, r, 0, 1, 3); r += 1
        layout.addWidget(self.run_btn, r, 0, 1, 3)

        self.resize(820, 220)

    def pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            "",
            "Data Files (*.xlsx *.parquet);;Excel Files (*.xlsx);;Parquet Files (*.parquet);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def pick_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML config", "", "YAML Files (*.yml *.yaml);;All Files (*)"
        )
        if path:
            self.config_edit.setText(path)

    def pick_vi(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select VI Trigger file", "",
            "VI Files (*.xlsx *.csv);;Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.vi_edit.setText(path)

    def run_detector(self):
        in_path_str = self.input_edit.text().strip()
        if not in_path_str:
            QMessageBox.warning(self, "Missing input", "Please choose an input .xlsx or .parquet file.")
            return
        input_path = Path(in_path_str)
        if not input_path.exists():
            QMessageBox.critical(self, "File not found", f"Input file not found:\n{input_path}")
            return

        cfg_path = Path(self.config_edit.text().strip()) if self.config_edit.text().strip() else None
        if cfg_path and not cfg_path.exists():
            QMessageBox.critical(self, "Config not found", f"YAML config not found:\n{cfg_path}")
            return
        if cfg_path and yaml is None:
            QMessageBox.critical(self, "Missing dependency",
                                 "PyYAML is not installed. Run:\n\n    pip install pyyaml")
            return

        vi_path = Path(self.vi_edit.text().strip()) if self.vi_edit.text().strip() else None
        if vi_path and not vi_path.exists():
            QMessageBox.critical(self, "VI Trigger not found", f"VI file not found:\n{vi_path}")
            return

        # UI feedback
        self.setCursor(Qt.WaitCursor)
        self.run_btn.setEnabled(False)
        try:
            if self.mode_guarded.isChecked():
                out_csv, out_parq, out_xlsx, nrows = backtest_guarded_passive_bid(
                    input_path=input_path,
                    config_path=cfg_path,
                    write_xlsx=self.write_xlsx.isChecked(),
                    vi_trigger_path=vi_path
                )
                label = "Guarded Passive Bid backtest"
            else:
                out_csv, out_parq, out_xlsx, nrows = detect_arbitrage_locf_percent(
                    input_path=input_path,
                    config_path=cfg_path,
                    write_xlsx=self.write_xlsx.isChecked(),
                    vi_trigger_path=vi_path
                )
                label = "Arbitrage windows detector"

            msg = (
                f"{label} finished.\n\n"
                f"CSV saved:\n{out_csv}\n"
                f"Parquet saved:\n{out_parq}\n"
            )
            if out_xlsx:
                msg += f"\nXLSX saved:\n{out_xlsx}\n"
            msg += f"\nResults (rows): {nrows:,}"
            QMessageBox.information(self, "Completed", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")
        finally:
            self.run_btn.setEnabled(True)
            self.unsetCursor()


def main():
    app = QApplication(sys.argv)
    gui = ArbGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
