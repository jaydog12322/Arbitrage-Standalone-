#!/usr/bin/env python3
# -*- coding: utf-8 -*-.

"""
Minimal GUI wrapper (PySide6) for the LOCF arbitrage detector (percent/bps logic).

UPGRADE (non-breaking, GUI unchanged except one optional field):
- Input: now supports Excel (.xlsx) OR Parquet (.parquet).
- Output: always writes Parquet opportunities next to the input, in addition to the
          existing CSV (and optional XLSX when the checkbox is enabled).
- NEW: Optional VI (변동성완화장치) filter:
    * Browse a VI Trigger file (.xlsx or .csv) listing:
        - stock code/name
        - VI start time
        - VI end time
    * Any opportunity whose timestamp falls inside a VI window for its symbol is removed.

Features (unchanged otherwise):
- Browse for input data containing columns:
    timestamp, symbol, venue, fid_41 (ask), fid_51 (bid), fid_61 (ask size), fid_71 (bid size)
- Optional: browse for a YAML config (fees & thresholds); otherwise sensible defaults are used.
- Outputs are written to the SAME folder as the input:
    <input_stem>_arbitrage_windows_bps.csv
    <input_stem>_arbitrage_windows_bps.parquet
    <input_stem>_arbitrage_windows_bps.xlsx      (if "Write XLSX too" is checked)

Dependencies:
    pip install PySide6 pandas numpy pyyaml openpyxl
    # for Parquet:
    pip install pyarrow    # preferred
    # or:
    pip install fastparquet
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


# ---------------- Core arbitrage logic (percent/bps + LOCF) -------------------

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


def load_yaml_config(path: Optional[Path]) -> dict:
    """
    Load YAML config if provided; otherwise return defaults.
    Also allows setting VI filter behavior:
        vi_filter:
            default_end_minutes: 3
            enabled: true
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
            "enabled": True,               # gate on/off via YAML
            "default_end_minutes": 3       # used when VI end is missing
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
    Fees in bps (%*100), applied to each side:
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

    required = ["timestamp", "symbol", "venue", "fid_41", "fid_51", "fid_61", "fid_71"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # timestamp may have a leading single quote (from Excel-safe exports)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.replace("'", ""),
        errors="coerce",
        infer_datetime_format=True
    )
    df = df.dropna(subset=["timestamp", "symbol", "venue"]).copy()
    for c in ["fid_41", "fid_51", "fid_61", "fid_71"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fid_41", "fid_51", "fid_61", "fid_71"]).copy()
    df["venue"] = df["venue"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    return df


def _save_opportunities_all_formats(
    df_raw: pd.DataFrame,
    df_opp: pd.DataFrame,
    base_stem_path: Path,
    write_xlsx: bool
) -> Tuple[Path, Path, Optional[Path]]:
    """
    Save opportunities as CSV (legacy), Parquet (new), and optionally XLSX (legacy GUI checkbox).
    Returns (csv_path, parquet_path, xlsx_path_or_None)
    """
    out_csv = base_stem_path.with_name(f"{base_stem_path.name}_arbitrage_windows_bps.csv")
    out_parq = base_stem_path.with_name(f"{base_stem_path.name}_arbitrage_windows_bps.parquet")
    out_xlsx = base_stem_path.with_name(f"{base_stem_path.name}_arbitrage_windows_bps.xlsx") if write_xlsx else None

    # CSV (always)
    df_opp.to_csv(out_csv, index=False)

    # Parquet (always)
    _ensure_parquet_engine_or_raise()
    df_opp.to_parquet(out_parq, index=False, engine=_PARQUET_ENGINE)

    # XLSX (optional) — ensure at least one visible sheet exists
    if out_xlsx:
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                wrote_any = False

                # write raw if exists (even if empty with columns, it’s fine)
                if df_raw is not None and (len(df_raw.columns) > 0):
                    df_raw.to_excel(w, index=False, sheet_name="data")
                    wrote_any = True

                # write opps if exists (even if filtered to 0 rows)
                if df_opp is not None and (len(df_opp.columns) > 0):
                    df_opp.to_excel(w, index=False, sheet_name="opportunities")
                    wrote_any = True

                # if for any reason nothing was written, add a tiny placeholder
                if not wrote_any:
                    pd.DataFrame({"note": ["No data to write"]}).to_excel(
                        w, index=False, sheet_name="empty"
                    )

        except IndexError:
            # Fallback: create a guaranteed visible sheet
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
    # drop decimals if came from Excel (e.g., "12345.0")
    if s.endswith(".0"):
        s = s[:-2]
    # normalize leading zeros for consistent matching
    try:
        return f"{int(s):06d}"
    except Exception:
        return s  # fall back

def _time_to_td(t) -> pd.Timedelta:
    # accepts pd.Timestamp, datetime.time, or "HH:MM:SS"
    if pd.isna(t):
        return pd.NaT
    if isinstance(t, pd.Timestamp):
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    if hasattr(t, "hour") and hasattr(t, "minute") and hasattr(t, "second"):
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
    # string
    try:
        parts = str(t).strip().split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(float(parts[2]))
        return pd.Timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return pd.NaT

def _load_vi_intervals_any(path: Optional[Path], default_end_minutes: int) -> Optional[dict]:
    """
    Returns dict with:
      {
        "by_code": { "005930": [(td_start, td_end_or_default), ...], ... },
        "by_name": { "삼성전자": [(td_start, td_end_or_default), ...], ... }
      }
    td_* are Timedeltas since 00:00 for quick intra-day matching.

    Accepts .xlsx/.xls or .csv. Column flexibility:
      Korean: 종목코드, 종목명, 발동\n시간, 해지\n시간
      English: code/symbol, name, vi_start, vi_end
    """
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

    # Normalize columns
    cols = {c: _norm_name(c) for c in df.columns}
    df = df.rename(columns=cols)

    # Map possible names
    code_col = next((c for c in df.columns if "종목코드" in c or c in ("code", "symbolcode", "ticker", "symbol_code")), None)
    if code_col is None and "symbol" in df.columns:
        code_col = "symbol"  # if user provided numeric symbol in this column

    name_col = next((c for c in df.columns if " 종목명" in (" "+c) or c in ("name", "symbolname", "symbol_name")), None)
    if name_col is None and "종목명" in df.columns:
        name_col = "종목명"

    # Start/End time columns (prefer Korean headers if present)
    start_col = next((c for c in df.columns if "발동" in c and "시간" in c), None)
    if start_col is None:
        start_col = next((c for c in df.columns if c in ("vi_start", "start", "start_time", "starttime")), None)

    end_col = next((c for c in df.columns if "해지" in c and "시간" in c), None)
    if end_col is None:
        end_col = next((c for c in df.columns if c in ("vi_end", "end", "end_time", "endtime")), None)

    if start_col is None:
        raise RuntimeError("VI Trigger file: could not find start time column (e.g., '발동\\n시간' or 'vi_start').")

    # Build dictionaries
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

    # sort intervals per key
    for d in (by_code, by_name):
        for k in d:
            d[k].sort()

    return {"by_code": by_code, "by_name": by_name}


def _row_in_vi(symbol: str, ts: pd.Timestamp, vi_idx: dict) -> bool:
    """
    Check if row timestamp falls in any VI window for symbol (by code or name).
    We compare only the time-of-day (Timedelta since midnight), assuming VI file is for the same day.
    """
    if not vi_idx:
        return False

    tod = pd.Timedelta(hours=ts.hour, minutes=ts.minute, seconds=ts.second) + pd.Timedelta(microseconds=ts.microsecond)
    s_sym = str(symbol).strip()

    # Try numeric-ish code match
    s_code = _norm_code(s_sym)
    lst = []
    if vi_idx.get("by_code") and s_code in vi_idx["by_code"]:
        lst = vi_idx["by_code"][s_code]
    else:
        # Fall back to name match
        s_name = _norm_name(s_sym)
        if vi_idx.get("by_name") and s_name in vi_idx["by_name"]:
            lst = vi_idx["by_name"][s_name]

    if not lst:
        return False

    # intervals are sorted; linear scan is fine for modest sizes
    for st, en in lst:
        if st <= tod <= en:
            return True
    return False


# -------------------------- Core detector (+ VI) --------------------------

def detect_arbitrage_locf_percent(
    input_path: Path,
    config_path: Optional[Path],
    write_xlsx: bool,
    vi_trigger_path: Optional[Path] = None
) -> Tuple[Path, Path, Optional[Path], int]:
    """Core runner: loads data, applies LOCF + bps logic, VI filtering, writes outputs next to input."""
    df = _read_input_any(input_path)
    cfg = load_yaml_config(config_path)

    # Build per-venue fee dictionary: commission + optional sell-side tax.
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

    # VI filter
    vi_cfg_enabled = bool(cfg.get("vi_filter", {}).get("enabled", True))
    vi_default_end_min = int(cfg.get("vi_filter", {}).get("default_end_minutes", 3))
    vi_index = None
    if vi_cfg_enabled and vi_trigger_path:
        vi_index = _load_vi_intervals_any(vi_trigger_path, default_end_minutes=vi_default_end_min)

    # Output base (same folder as input). Strip one suffix only.
    base_stem = input_path.with_suffix("")

    # LOCF snapshots by symbol
    snapshots: Dict[str, QuoteSnapshot] = {}
    opps: List[dict] = []

    # Iterate in time order; update snapshot for (symbol, venue); evaluate both directions.
    for row in df.itertuples(index=False):
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
        else:  # treat anything else as NXT
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

        # Pre-open cutoff: drop anything before 09:00:30 (09:00:30 exactly is allowed)
        ts_row = getattr(row, "timestamp")
        tod = pd.Timedelta(hours=ts_row.hour, minutes=ts_row.minute, seconds=ts_row.second,
                           microseconds=ts_row.microsecond)
        if tod < pd.Timedelta(hours=9, minutes=0, seconds=30):
            continue

        # VI filter: drop if timestamp within any VI window for this symbol
        if vi_index and _row_in_vi(symbol, ts_row, vi_index):
            continue

        # include original row fields plus signal fields
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

    out_csv, out_parq, out_xlsx = _save_opportunities_all_formats(
        df_raw=df,
        df_opp=opp_df,
        base_stem_path=base_stem,
        write_xlsx=write_xlsx
    )

    return out_csv, out_parq, out_xlsx, len(opp_df)


# ------------------------------- Minimal GUI ----------------------------------

class ArbGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arbitrage Detector (LOCF • % terms)")

        self.input_edit = QLineEdit()
        self.input_browse = QPushButton("Browse…")
        self.input_browse.clicked.connect(self.pick_input)

        self.config_edit = QLineEdit()
        self.config_browse = QPushButton("Config… (optional)")
        self.config_browse.clicked.connect(self.pick_config)

        # NEW: VI Trigger file
        self.vi_edit = QLineEdit()
        self.vi_browse = QPushButton("VI Trigger… (optional)")
        self.vi_browse.clicked.connect(self.pick_vi)

        self.write_xlsx = QCheckBox("Also write XLSX (with raw + opportunities)")
        self.write_xlsx.setChecked(True)

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

        # NEW: VI row
        layout.addWidget(QLabel("VI Trigger file (.xlsx / .csv):"), r, 0)
        layout.addWidget(self.vi_edit, r, 1)
        layout.addWidget(self.vi_browse, r, 2); r += 1

        layout.addWidget(self.write_xlsx, r, 0, 1, 3); r += 1
        layout.addWidget(self.run_btn, r, 0, 1, 3)

        self.resize(780, 190)

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
            out_csv, out_parq, out_xlsx, nrows = detect_arbitrage_locf_percent(
                input_path=input_path,
                config_path=cfg_path,
                write_xlsx=self.write_xlsx.isChecked(),
                vi_trigger_path=vi_path
            )
            msg = (
                "Done!\n\n"
                f"CSV saved:\n{out_csv}\n"
                f"Parquet saved:\n{out_parq}\n"
            )
            if out_xlsx:
                msg += f"\nXLSX saved:\n{out_xlsx}\n"
            msg += f"\nOpportunities found (after VI filter): {nrows:,}"
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