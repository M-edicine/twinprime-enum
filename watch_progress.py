#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旁观正在运行的 twin_enum.py 的进度（只读 state 文件，不影响主进程）。

适用场景：主程序跑在你另一个窗口 / 被重定向 / 放进后台，看不到那条进度条时，
再开一个窗口跑本脚本就能看到计数、速度与 ETA。

用法：
    python watch_progress.py                 # 自动挑最新的 state 文件
    python watch_progress.py --state ..\\state\\TwinPrimes_1e+11.varint.state.json
    python watch_progress.py --interval 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
STATE = os.path.join(ROOT, "state")

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def latest_state() -> str | None:
    files = glob.glob(os.path.join(STATE, "*.state.json"))
    return max(files, key=os.path.getmtime) if files else None


def fmt_dur(sec: float) -> str:
    if sec != sec or sec < 0 or sec == float("inf"):
        return "--:--"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="旁观 twin_enum.py 的进度")
    ap.add_argument("--state", default=None, help="state 文件路径，默认自动挑最新")
    ap.add_argument("--interval", type=float, default=2.0, help="刷新间隔秒，默认 2")
    ap.add_argument("--once", action="store_true", help="只打印一次就退出")
    args = ap.parse_args()

    path = args.state or latest_state()
    if not path or not os.path.isfile(path):
        print(f"找不到 state 文件（{STATE} 下没有 *.state.json）——主程序还没开始？")
        return 1
    print(f"旁观 {path}（Ctrl+C 退出）", flush=True)

    last_count = last_t = None
    while True:
        try:
            st = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"读 state 失败（可能正被原子替换，稍后重试）：{e}", flush=True)
            time.sleep(args.interval)
            continue
        lo, hi = st["lo"], st["hi"]
        seg_k, bs = st["seg_k"], st["block_segs"]
        kmax = (hi - 1) // 6
        nseg = kmax // seg_k + 1
        done_seg = st["next_block"]
        frac = min(1.0, done_seg / nseg)
        count = st["count"]
        now = time.time()
        rate = eta = None
        if last_count is not None and now > last_t:
            rate = (count - last_count) / (now - last_t)
            if rate > 0:
                est_total = count / max(frac, 1e-9)
                eta = (est_total - count) / rate
        last_count, last_t = count, now
        bar = "#" * int(28 * frac) + "-" * (28 - int(28 * frac))
        scanned = min(done_seg * seg_k * 6, hi)
        out = (f"[{bar}] {frac*100:6.2f}%  段 {done_seg:,}/{nseg:,}  "
               f"数到 {scanned/1e9:.3f}G  "
               f"已出 {count:,} 对")
        if rate:
            out += f"  {rate/1e6:.2f} M对/秒  剩余 {fmt_dur(eta)}"
        out += "  " + str(st.get("updated", ""))
        if st.get("finished"):
            out = "** 已完成 ** " + out
        print(out, flush=True)
        for p in (st.get("out") or "", (st.get("text") or "")):
            if p and os.path.isfile(p):
                print(f"      {os.path.basename(p)}  {os.path.getsize(p)/1e6:.2f} MB", flush=True)
        if args.once or st.get("finished"):
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
