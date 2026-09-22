#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式程序：连续枚举 [lo, hi] 内全部孪生素数并落盘（二进制 + 文本）。

算法（由 bench_contest.py 的比武数据裁定）
  枚举用自研 **孪生格分段筛**（6k±1；k 未被标记 <=> (6k-1, 6k+1) 是孪生素数）。
  实测 22.6 M对/秒，是 primesieve 经 Python 绑定配对（4.0 M对/秒）的 5.6 倍，
  且输出天然对齐、不需要"取全体素数再差分"的中间量。

产出（两个文件，覆盖范围写在文件名里）
  <out>.varint   主产物：定长 JSON 头 + 增量 varint 数据体，约 1.4 字节/对
  <out>.txt      人读文本：每行 "p p+2"

进度
  stdout 是终端时画单行动态进度条（原地刷新）；被重定向/后台捕获时自动退化为
  周期性文本行。同时 state 文件持续更新，可用 watch_progress.py 旁观。

可续跑
  中途 Ctrl+C / 断电后重跑同一条命令即从断点继续（state 文件 + .tmp 截断对齐）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                      # 单目录布局：模块与 data/ state/ logs/ 同在仓库根

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import twin_lattice                                    # noqa: E402
import twinio                                          # noqa: E402
from progress import Progress, fmt_dur, fmt_num         # noqa: E402
from twin_lattice import KNOWN                          # noqa: E402

DATA = os.path.join(ROOT, "data")
STATE = os.path.join(ROOT, "state")
LOGS = os.path.join(ROOT, "logs")
PROG = "twin_enum.py v2"


class Logger:
    """同时写 stdout 与日志文件（逐行 flush，保证后台也能实时看）。"""

    def __init__(self, path: str):
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def enumerate_range(lo, hi, seg_k, block_segs, threads, log, tmp_path, state_path,
                    text_path, resume):
    """分段流式枚举，结果增量写入 tmp_path / text_path。返回统计字典。"""
    if threads:
        from numba import set_num_threads
        set_num_threads(threads)
    kmax = (hi - 1) // 6                      # 6k+1 <= hi
    primes = twin_lattice.small_primes(math.isqrt(hi) + 1)
    nseg = kmax // seg_k + 1
    C2 = 0.6601618158468696                   # Hardy-Littlewood 孪生素数常数
    est = 2 * C2 * hi / math.log(hi) ** 2 if hi > 3 else 0
    log(f"  区间 [{lo:,}, {hi:,}]  seg_k={seg_k:,}  block_segs={block_segs}  "
        f"线程={'全部' if not threads else threads}")
    log(f"  k 上界={kmax:,}  段数={nseg:,}  小素数表={primes.size:,} 个")
    log(f"  预期对数（渐近 2*C2*hi/ln^2 hi）≈ {est:,.0f}"
        + (f"；A007508 已知值 {KNOWN[hi]:,}" if hi in KNOWN else ""))

    start_block, count, prev, first_abs, body_bytes, text_bytes = 0, 0, 0, True, 0, 0
    resumed = False
    if resume and os.path.isfile(state_path):
        try:
            st = json.load(open(state_path, encoding="utf-8"))
        except Exception:
            st = {}
        same = (st.get("lo") == lo and st.get("hi") == hi
                and st.get("seg_k") == seg_k and st.get("block_segs") == block_segs)
        if same and not st.get("finished"):
            start_block = st["next_block"]
            count = st["count"]
            prev = st["last_p"]
            first_abs = st["first_absolute"]
            body_bytes = st["body_bytes"]
            text_bytes = st.get("text_bytes", 0)
            resumed = True
            log(f"  ** 断点续算：从第 {start_block:,}/{nseg:,} 段继续，"
                f"已有 {count:,} 对，数据体 {body_bytes:,} 字节")

    prog = Progress(total=float(kmax + 1),
                    label=f"孪生格枚举 {fmt_num(lo)}-{fmt_num(hi)} [进度按 k，1k=6数]")
    t0 = time.perf_counter()
    fh = open(tmp_path, "r+b" if (resumed and os.path.isfile(tmp_path)) else "w+b")
    txt = None
    if text_path:
        if resumed and os.path.isfile(text_path):
            txt = open(text_path, "r+b")
            txt.truncate(text_bytes)                  # 丢弃上次半截写
            txt.seek(0, os.SEEK_END)
        else:
            txt = open(text_path, "wb")
    try:
        if resumed:
            fh.seek(0, os.SEEK_END)
            if fh.tell() != twinio.HEADER_LEN + body_bytes:
                fh.truncate(twinio.HEADER_LEN + body_bytes)
            fh.seek(0, os.SEEK_END)
            prog.update(min(start_block * seg_k, kmax + 1), force=True)
        else:
            fh.write(b"\0" * twinio.HEADER_LEN)
            fh.flush()

        # (3,5)：3 不在 6k±1 格子里，区间含它就先写
        if first_abs and lo <= 3 and hi >= 5:
            head = np.array([3], dtype=np.int64)
            buf = np.empty(64, dtype=np.uint8)
            n, prev, first_abs = twinio.encode_deltas(head, 2, prev, first_abs, buf)
            fh.write(buf[:n].tobytes())
            if txt is not None:
                tb = np.empty(64, dtype=np.uint8)
                tn = int(twinio.encode_pairs_text(head, tb))
                txt.write(tb[:tn].tobytes())
                text_bytes += tn
            count += 1
            body_bytes = fh.tell() - twinio.HEADER_LEN

        for b0 in range(start_block, nseg, block_segs):
            b1 = min(b0 + block_segs, nseg)
            counts = twin_lattice._count_chunk(kmax, primes, seg_k, b0, b1)
            # 块内相对偏移：必须与 _enum_chunk 的约定严格一致（全程偏移 + 块内缓冲
            # 会越界写堆，曾导致进程被 0xC0000005 打死）
            bl = np.zeros(b1 - b0 + 1, dtype=np.int64)
            np.cumsum(counts, out=bl[1:])
            kbuf = np.empty(int(bl[-1]), dtype=np.int64)
            twin_lattice._enum_chunk(kmax, primes, seg_k, b0, b1, bl, kbuf)
            # 后置检查：枚举出的 k 必 >= 1，出现 0 或负数说明有槽位没被写到
            # （偏移量约定不匹配），立刻中止以免把坏数据写进产物
            if kbuf.size and int(kbuf.min()) <= 0:
                raise RuntimeError(
                    f"第 {b0}-{b1} 段枚举输出出现未写入槽位（min={int(kbuf.min())}），"
                    "偏移量约定不匹配，已中止")
            p = 6 * kbuf - 1
            if lo > 3:
                p = p[p >= lo]
            if p.size:
                buf = np.empty(p.size * 10 + 16, dtype=np.uint8)
                nb, prev, first_abs = twinio.encode_deltas(p, 2, prev, first_abs, buf)
                fh.write(buf[:nb].tobytes())
                count += int(p.size)
                fh.flush()
                body_bytes = fh.tell() - twinio.HEADER_LEN
                if txt is not None:
                    tb = np.empty(p.size * 42 + 32, dtype=np.uint8)
                    tn = int(twinio.encode_pairs_text(p, tb))
                    txt.write(tb[:tn].tobytes())
                    txt.flush()
                    text_bytes += tn
            atomic_json(state_path, dict(
                lo=lo, hi=hi, seg_k=seg_k, block_segs=block_segs, next_block=b1,
                count=count, last_p=int(prev), first_absolute=bool(first_abs),
                body_bytes=body_bytes, text_bytes=text_bytes, finished=False,
                program=PROG, updated=time.strftime("%Y-%m-%d %H:%M:%S")))
            prog.update(min(b1 * seg_k, kmax + 1),
                        extra=f"段 {b1:,}/{nseg:,}  数到 {fmt_num(min(b1*seg_k*6, hi))}"
                              f"  已出 {count:,} 对")
    finally:
        fh.close()
        if txt is not None:
            txt.close()
    el = time.perf_counter() - t0
    prog.close(extra=f"{count:,} 对")
    return dict(count=count, body_bytes=body_bytes, text_bytes=text_bytes, seg_k=seg_k,
                block_segs=block_segs, seconds=el, kmax=kmax, nseg=nseg,
                pairs_per_sec=count / el if el > 0 else 0.0)


def verify_text(path: str, p: np.ndarray, log) -> bool:
    """文本产物校验：行数、首行、末行与二进制产物一致。

    行数用 bytes.count(b"\\n") 按 16 MB 块统计——1e11 的文本有 5.8 GB /
    2.24 亿行，用 Python 逐行迭代要一两分钟，按块数只要几秒。
    """
    lines = 0
    first = None
    tail = bytearray()
    with open(path, "rb") as fh:
        while True:
            blk = fh.read(1 << 24)
            if not blk:
                break
            lines += blk.count(b"\n")
            if first is None:
                i = blk.find(b"\n")
                if i >= 0:
                    first = blk[:i]
            tail += blk
            if len(tail) > 4096:
                del tail[:-4096]
    ok = lines == p.size
    log(f"  文本行数 {lines:,} vs 二进制 {p.size:,} " + ("OK" if ok else "!! 不一致"))
    if first is not None and p.size:
        parts = bytes(tail).split(b"\n")
        last = parts[-2] if parts[-1] == b"" else parts[-1]
        exp_first = f"3 5" if p[0] == 3 else f"{p[0]} {p[0]+2}"
        exp_last = f"{p[-1]} {p[-1]+2}"
        f1 = first.decode(errors="replace")
        l1 = last.decode(errors="replace")
        good = (f1 == exp_first and l1 == exp_last)
        log(f"  文本首行 {f1!r}（应 {exp_first!r}）；末行 {l1!r}（应 {exp_last!r}）"
            + (" OK" if good else " !! 不一致"))
        ok &= good
    return ok


def verify(path: str, lo: int, hi: int, log, sample: int = 200, text_path=None) -> bool:
    """收尾校验：头/计数/边界/单调/SHA256/A007508/primesieve/sympy/文本。全过才 True。"""
    log("\n  === 收尾校验 ===")
    ok = True
    meta, p = twinio.load_pairs(path)
    log(f"  头部：范围 [{meta['lo']:,}, {meta['hi']:,}]，count={meta['count']:,}，"
        f"数据体 {meta['body_bytes']:,} 字节 = {meta['body_bytes']/max(1,meta['count']):.3f} 字节/对")
    if p.size != meta["count"]:
        log(f"  !! 解码出 {p.size:,} 对与头部 count 不符")
        ok = False
    # 不用 np.diff（1e11 会多分配 1.8 GB 临时数组），用切片视图比较
    if p.size > 1 and not bool(np.all(p[1:] > p[:-1])):
        log("  !! 序列非严格递增")
        ok = False
    if p.size and (p[0] < lo or p[-1] + 2 > hi):
        log(f"  !! 越界：首 {p[0]}，末 {p[-1]}")
        ok = False
    sha = twinio.sha256_body(path)
    log(f"  数据体 SHA256 {sha}"
        + ("（与头部一致 OK）" if sha == meta.get("sha256_body") else " !! 与头部不一致"))
    ok &= (sha == meta.get("sha256_body"))

    if lo <= 3 and hi in KNOWN:
        good = p.size == KNOWN[hi]
        log(f"  OEIS A007508  pi2({hi:.0e}) 期望 {KNOWN[hi]:,}，实际 {p.size:,} "
            + ("OK" if good else "!! 不一致"))
        ok &= good

    try:
        import primesieve
        t0 = time.perf_counter()
        cnt = int(primesieve.count_twins(lo, hi))
        el = time.perf_counter() - t0
        good = cnt == p.size
        log(f"  primesieve 独立计数 count_twins({lo:,}, {hi:,}) = {cnt:,}，"
            f"用时 {el:.3f}s " + ("OK 一致" if good else "!! 不一致"))
        ok &= good
        if lo > 3:
            c_hi = int(primesieve.count_twins(1, hi))
            c_lo = int(primesieve.count_twins(1, lo))
            log(f"  累计口径参考：count_twins(1,{hi:.0e})={c_hi:,}，"
                f"count_twins(1,{lo:.0e})={c_lo:,}，两者差 {c_hi-c_lo:,}"
                f"（与本次 {p.size:,} 的差额来自跨下界边界的那几对）")
    except ImportError:
        log("  primesieve 未安装，跳过独立计数（是否已安装？）")

    if p.size and sample > 0:
        try:
            import sympy
        except ImportError:
            sympy = None
            log("  sympy 未安装，跳过抽样复核（pip install sympy 可启用）")
        if sympy is not None:
            random.seed(20260921)
            idx = random.sample(range(p.size), min(sample, p.size))
            bad = [int(p[i]) for i in idx
                   if not (sympy.isprime(int(p[i])) and sympy.isprime(int(p[i]) + 2))]
            log(f"  sympy 抽样 {len(idx)} 对："
                + ("全部为素数 OK" if not bad else f"!! 异常 {bad[:5]}"))
            ok &= not bad

    if text_path and os.path.isfile(text_path):
        ok &= verify_text(text_path, p, log)

    log(f"  校验结论：{'全部通过 OK' if ok else '存在失败项 !!'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="连续枚举区间内全部孪生素数并落盘（二进制 varint + 文本）")
    ap.add_argument("--lo", type=float, default=3, help="下界（含），默认 3")
    ap.add_argument("--hi", type=float, default=1e10, help="上界（含），默认 1e10")
    ap.add_argument("--out", default=None, help="产物路径，默认按范围自动命名")
    ap.add_argument("--seg-k", type=float, default=5e5,
                    help="每段 k 个数（默认 5e5，实测最优；调大掉缓存会慢 10 倍）")
    ap.add_argument("--block-segs", type=int, default=256,
                    help="每块含多少段（决定内存占用与 state 更新粒度）")
    ap.add_argument("--threads", type=int, default=0, help="线程数，0=全部核心")
    ap.add_argument("--text", action=argparse.BooleanOptionalAction, default=True,
                    help="产出人读文本版（默认开；--no-text 关闭）")
    ap.add_argument("--no-resume", action="store_true", help="忽略断点，从头重算")
    ap.add_argument("--sample", type=int, default=200, help="sympy 抽样对数，0=不抽样")
    args = ap.parse_args()

    lo, hi = int(args.lo), int(args.hi)
    seg_k, block_segs = int(args.seg_k), args.block_segs
    if hi <= lo:
        print("!! --hi 必须大于 --lo")
        return 2
    for d in (DATA, STATE, LOGS):
        os.makedirs(d, exist_ok=True)

    tag = f"{lo:.0e}_{hi:.0e}" if lo > 3 else f"{hi:.0e}"
    out = args.out or os.path.join(DATA, f"TwinPrimes_{tag}.varint")
    text_path = out[:-len(".varint")] + ".txt" if args.text else None
    tmp_path = out + ".tmp"
    base = os.path.basename(out)
    state_path = os.path.join(STATE, base + ".state.json")
    log_path = os.path.join(LOGS, base + ".log")

    log = Logger(log_path)
    log(f"# 孪生素数连续枚举  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  程序 {PROG}")
    log(f"  二进制产物 -> {out}")
    log(f"  文本产物   -> {text_path if text_path else '（已关闭）'}")
    log(f"  断点状态   -> {state_path}")
    log(f"  本次日志   -> {log_path}")

    st = enumerate_range(lo, hi, seg_k, block_segs, args.threads, log, tmp_path,
                         state_path, text_path, resume=not args.no_resume)
    log(f"  枚举完成：{st['count']:,} 对，用时 {st['seconds']:.3f}s，"
        f"{st['pairs_per_sec']/1e6:.2f} M对/秒")

    sha_body = twinio.sha256_body(tmp_path)
    with open(tmp_path, "r+b") as fh:
        twinio.write_header(fh, dict(
            magic=twinio.MAGIC, program=PROG, lo=lo, hi=hi, seg_k=seg_k,
            block_segs=block_segs, count=st["count"], body_bytes=st["body_bytes"],
            sha256_body=sha_body, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            definition="pairs (p, p+2) with p >= lo and p+2 <= hi",
            encoding="varint LEB128; first value absolute p; then (p_i - p_{i-1})//2"))
    os.replace(tmp_path, out)
    size = os.path.getsize(out)
    log(f"  二进制已落盘：{out}")
    log(f"    {size:,} 字节 = {size/1e6:.2f} MB = {size/max(1,st['count']):.3f} 字节/对")
    log(f"  文件 SHA256 {twinio.sha256_file(out)}")
    if text_path:
        ts = os.path.getsize(text_path)
        log(f"  文本已落盘：{text_path}")
        log(f"    {ts:,} 字节 = {ts/1e6:.2f} MB = {ts/max(1,st['count']):.2f} 字节/对")
    atomic_json(state_path, dict(lo=lo, hi=hi, seg_k=seg_k, block_segs=block_segs,
                                next_block=st["nseg"], count=st["count"], last_p=0,
                                first_absolute=False, body_bytes=st["body_bytes"],
                                text_bytes=st["text_bytes"], finished=True,
                                program=PROG, out=out, text=text_path,
                                updated=time.strftime("%Y-%m-%d %H:%M:%S")))

    ok = verify(out, lo, hi, log, sample=args.sample, text_path=text_path)
    log(f"\n程序结论：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
