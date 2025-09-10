#!/usr/bin/env python3
# -*- coding: utf-8 -*-.

"""
Minimal GUI wrapper (PySide6) for the LOCF arbitrage detector (percent/bps logic).

Features:
- Browse for input Excel (.xlsx) containing columns:
    timestamp, symbol, venue, fid_41 (ask), fid_51 (bid), fid_61 (ask size), fid_71 (bid size)
- Optional: browse for a YAML config (fees & thresholds); otherwise sensible defaults are used.
- Outputs are written to the SAME folder as the input:
    <input_stem>_arbitrage_windows_bps.csv
    <input_stem>_arbitrage_windows_bps.xlsx  (if "Write XLSX too" is checked)

Dependencies:
    pip install PySide6 pandas numpy pyyaml openpyxl
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
    """Load YAML config if provided; otherwise return defaults."""
    default_cfg = {
        "fees": {
            "krx": {"broker_bps": 8},           # 0.08%
            "nxt": {"broker_bps": 8, "regulatory_bps": 0},  # adjust as needed
        },
        "spread_engine": {
            "edge_rule": {
                "min_net_ticks_after_fees": 1,      # require ≥ 1 tick after fees
                "also_require_min_visible_qty": 1,  # both venues have at least this visible
            }
        }
    }
    if path and yaml:
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            # shallow-merge user into defaults
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
    fees_bps: Dict[str, float],
) -> Optional[dict]:
    """
    Fees in bps (%*100), applied to each side:
        buy_fees  = buy_price  * fees_bps[buy_venue]  / 10000
        sell_fees = sell_price * fees_bps[sell_venue] / 10000
    net_edge_krw = (sell_price - buy_price) - (buy_fees + sell_fees)
    edge_bps     = net_edge_krw / buy_price * 10000
    """
    if sell_price <= buy_price:
        return None
    gross_edge = sell_price - buy_price
    buy_fees = buy_price * fees_bps.get(buy_venue, 0.0) / 10000.0
    sell_fees = sell_price * fees_bps.get(sell_venue, 0.0) / 10000.0
    total_fees = buy_fees + sell_fees
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
    # Reference mid = average of venue mids (if both present) else whichever exists.
    krx_mid = (q.krx_ask + q.krx_bid) / 2 if (q.krx_ask and q.krx_bid) else 0
    nxt_mid = (q.nxt_ask + q.nxt_bid) / 2 if (q.nxt_ask and q.nxt_bid) else 0
    ref_mid = (krx_mid + nxt_mid) / 2 if (krx_mid and nxt_mid) else (krx_mid or nxt_mid)
    tick = get_tick_size(ref_mid) if ref_mid else 10
    return signal["net_edge_krw"] >= tick * min_ticks_after_fees


def load_input_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["timestamp", "symbol", "venue", "fid_41", "fid_51", "fid_61", "fid_71"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    # timestamp may have a leading single quote
    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype(str).str.replace("'", ""), errors="coerce", infer_datetime_format=True
    )
    df = df.dropna(subset=["timestamp", "symbol", "venue"]).copy()
    for c in ["fid_41", "fid_51", "fid_61", "fid_71"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["fid_41", "fid_51", "fid_61", "fid_71"]).copy()
    # Normalize & sort
    df["venue"] = df["venue"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    return df


def detect_arbitrage_locf_percent(
    input_path: Path,
    config_path: Optional[Path],
    write_xlsx: bool
) -> Tuple[Path, Optional[Path], int]:
    """Core runner: loads data, applies LOCF + bps logic, writes outputs next to input."""
    df = load_input_excel(input_path)
    cfg = load_yaml_config(config_path)

    fees_bps = {
        "KRX": float(cfg["fees"]["krx"]["broker_bps"]),
        "NXT": float(cfg["fees"]["nxt"]["broker_bps"]) + float(cfg["fees"]["nxt"].get("regulatory_bps", 0)),
    }
    min_ticks = int(cfg["spread_engine"]["edge_rule"]["min_net_ticks_after_fees"])
    min_visible = int(cfg["spread_engine"]["edge_rule"]["also_require_min_visible_qty"])

    # Output paths (same folder as input)
    base = input_path.with_suffix("")  # strip .xlsx
    out_csv = Path(f"{base}_arbitrage_windows_bps.csv")
    out_xlsx = Path(f"{base}_arbitrage_windows_bps.xlsx") if write_xlsx else None

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
        if meets_tick_threshold(best, snap, min_ticks):
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

    # Save CSV (and optional XLSX)
    opp_df = pd.DataFrame(opps, columns=list(df.columns) + [
        "buy_venue", "sell_venue", "buy_price", "sell_price",
        "edge_krw", "total_fees_krw", "net_edge_krw", "edge_bps", "max_qty"
    ])
    opp_df.to_csv(out_csv, index=False)
    if out_xlsx:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="data")
            opp_df.to_excel(w, index=False, sheet_name="opportunities")

    return out_csv, out_xlsx, len(opp_df)


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

        self.write_xlsx = QCheckBox("Also write XLSX (with raw + opportunities)")
        self.write_xlsx.setChecked(True)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_detector)

        layout = QGridLayout(self)
        r = 0
        layout.addWidget(QLabel("Input Excel (.xlsx):"), r, 0)
        layout.addWidget(self.input_edit, r, 1)
        layout.addWidget(self.input_browse, r, 2); r += 1

        layout.addWidget(QLabel("Config YAML (fees/thresholds):"), r, 0)
        layout.addWidget(self.config_edit, r, 1)
        layout.addWidget(self.config_browse, r, 2); r += 1

        layout.addWidget(self.write_xlsx, r, 0, 1, 3); r += 1
        layout.addWidget(self.run_btn, r, 0, 1, 3)

        self.resize(700, 150)

    def pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select input Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def pick_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML config", "", "YAML Files (*.yml *.yaml);;All Files (*)"
        )
        if path:
            self.config_edit.setText(path)

    def run_detector(self):
        in_path_str = self.input_edit.text().strip()
        if not in_path_str:
            QMessageBox.warning(self, "Missing input", "Please choose an input .xlsx file.")
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

        # UI feedback
        self.setCursor(Qt.WaitCursor)
        self.run_btn.setEnabled(False)
        try:
            out_csv, out_xlsx, nrows = detect_arbitrage_locf_percent(
                input_path=input_path,
                config_path=cfg_path,
                write_xlsx=self.write_xlsx.isChecked()
            )
            msg = f"Done!\n\nCSV saved:\n{out_csv}\n"
            if out_xlsx:
                msg += f"\nXLSX saved:\n{out_xlsx}\n"
            msg += f"\nOpportunities found: {nrows:,}"
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
