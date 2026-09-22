#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""进度显示工具：进度条 + 计时器 + ETA。

约定：
  * stdout 是终端时画单行动态进度条（`\\r` 原地刷新）；
  * 被重定向 / 后台捕获时**自动退化**为周期性文本行（每 min_interval 秒一行）；
  * 进度按**成本加权**调用方传进来的 done/total（不是按行数）；
  * ETA 用实测速度外推。
"""

from __future__ import annotations

import sys
import time


def fmt_dur(sec: float) -> str:
    """秒 -> H:MM:SS 或 MM:SS。"""
    if sec is None or sec != sec or sec < 0 or sec == float("inf"):
        return "--:--"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_num(x: float) -> str:
    if x >= 1e9:
        return f"{x/1e9:.3f}G"
    if x >= 1e6:
        return f"{x/1e6:.3f}M"
    if x >= 1e3:
        return f"{x/1e3:.2f}k"
    return f"{x:.0f}"


class Progress:
    """单行动态进度条，非 TTY 自动退化为周期文本行。"""

    def __init__(self, total: float, label: str = "", stream=None,
                 min_interval: float = 3.0, width: int = 28, refresh: float = 0.2):
        self.total = float(total) if total else 1.0
        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        try:
            self.tty = bool(self.stream.isatty())
        except Exception:
            self.tty = False
        self.min_interval = min_interval
        self.refresh = refresh
        self.width = width
        self.t0 = time.perf_counter()
        self.t_last = 0.0
        self.done = 0.0
        self.extra = ""
        self._printed_line = self.tty

    # -- 内部 --------------------------------------------------------------
    def _rate(self) -> float:
        el = time.perf_counter() - self.t0
        return self.done / el if el > 0 else 0.0

    def _eta(self) -> float:
        r = self._rate()
        if r <= 0:
            return float("inf")
        return max(0.0, (self.total - self.done) / r)

    def _line(self) -> str:
        frac = min(1.0, self.done / self.total)
        el = time.perf_counter() - self.t0
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        speed = self._rate()
        unit = "/s"
        return (f"{self.label} [{bar}] {frac*100:6.2f}% "
                f"{fmt_num(self.done)}/{fmt_num(self.total)} "
                f"{fmt_num(speed)}{unit} 已用 {fmt_dur(el)} 剩余 {fmt_dur(self._eta())}"
                + (f" | {self.extra}" if self.extra else ""))

    # -- 公开 --------------------------------------------------------------
    def update(self, done: float = None, extra: str = None, force: bool = False) -> None:
        if done is not None:
            self.done = float(done)
        if extra is not None:
            self.extra = extra
        now = time.perf_counter()
        gap = self.refresh if self.tty else self.min_interval
        if not force and (now - self.t_last) < gap:
            return
        self.t_last = now
        line = self._line()
        if self.tty:
            self.stream.write("\r" + line + " " * 4)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def log(self, msg: str) -> None:
        """附加文本行（不破坏进度条）。"""
        if self.tty:
            self.stream.write("\r" + " " * 100 + "\r")
        self.stream.write(msg + "\n")
        self.stream.flush()
        if self.tty:
            self.stream.write(self._line())

    def close(self, extra: str = None) -> float:
        if extra is not None:
            self.extra = extra
        self.update(self.total, force=True)
        el = time.perf_counter() - self.t0
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
        return el
