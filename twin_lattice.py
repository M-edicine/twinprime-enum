#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孪生格分段筛：用 6k±1 表示候选对 (6k-1, 6k+1)。

核心思想
--------
每个 k>=1 唯一对应一对候选 (6k-1, 6k+1)。k 未被筛掉 **等价于** 两者都是素数，
即 (6k-1, 6k+1) 是孪生素数。于是相对"奇数筛"：

  * 候选数组只有 1/3 大（N/6 个 k，而不是 N/2 个奇数）→ 缓存局部性更好；
  * 判定与输出天然对齐，**不存在跨段衔接问题**（奇数筛需要段尾多带一个奇数）；
  * 例外只有一对 (3,5)（3 不在格子里），计数时手工 +1。

标记规则（素数 p >= 5）：
    6k-1 ≡ 0 (mod p)  <=>  k ≡   inv6  (mod p)
    6k+1 ≡ 0 (mod p)  <=>  k ≡ -inv6  (mod p)
  inv6 = 6 模 p 的逆元。两条等差链都从 k >= (p*p-1)//6 起步；该阈值保证
  链上首个值 >= p*p，因此不会把 p 自身误标为合数（这是奇数筛踩过的坑）。

并行归约注意：每个段写自己的 out[s]，最后求和，
不在 prange 里对同一标量做归约。
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from numba import njit, prange, set_num_threads

# GBK 控制台编不出的字符自动降级，避免 UnicodeEncodeError 打断长跑
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# OEIS A007508：孪生素数对数量 pi2(10^n)
KNOWN = {
    10**6: 8169, 10**7: 58980, 10**8: 440312, 10**9: 3424506,
    10**10: 27412679, 10**11: 224376048, 10**12: 1870585220,
    10**13: 15834664872, 10**14: 135780321665,
}


def small_primes(limit: int) -> np.ndarray:
    """sqrt(上界) 以内的小素数表（一次性，之后复用）。"""
    if limit < 2:
        return np.zeros(0, dtype=np.int64)
    sieve = np.ones(limit + 1, dtype=np.bool_)
    sieve[:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0].astype(np.int64)


def kmax_for(limit: int) -> int:
    """使 6k+1 <= limit 的最大 k（即最大的候选对下标）。"""
    return (limit - 1) // 6


@njit(cache=True)
def _mark_segment(comp, k0, m, primes):
    """把段内 [k0, k0+m) 的被 p 整除的 k 标成 1（p >= 5）。"""
    for j in range(primes.size):
        p = primes[j]
        if p < 5:
            continue
        # 段内最大候选值
        if p * p > 6 * (k0 + m - 1) + 1:
            break
        kmin = (p * p - 1) // 6
        lo = k0 if k0 > kmin else kmin
        if lo >= k0 + m:
            continue
        inv6 = (p + 1) // 6 if p % 6 == 5 else (5 * p + 1) // 6
        r1 = inv6 % p
        r2 = (p - r1) % p
        for t in range(2):
            r = r1 if t == 0 else r2
            k = lo + ((r - lo % p) % p)
            if k < k0 + m:
                comp[k - k0::p] = 1


@njit(cache=True, parallel=True)
def _count_chunk(kmax, primes, seg_k, s0, s1):
    """统计第 [s0, s1) 段的孪生素数对（每段写自己的格子）。"""
    n = s1 - s0
    out = np.zeros(n, dtype=np.int64)
    for t in prange(n):
        s = s0 + t
        k0 = s * seg_k
        if k0 > kmax:
            continue
        m = min(seg_k, kmax - k0 + 1)
        if m <= 0:
            continue
        comp = np.zeros(m, dtype=np.uint8)
        _mark_segment(comp, k0, m, primes)
        c = 0
        i = 0
        if k0 == 0:
            i = 1                     # 跳过 k=0（对应 -1 和 1，不是素数对）
        for q in range(i, m):
            if comp[q] == 0:
                c += 1
        out[t] = c
    return out


@njit(cache=True, parallel=True)
def _enum_chunk(kmax, primes, seg_k, s0, s1, offs, out):
    """把第 [s0, s1) 段的孪生 k 写入 out 的对应切片。

    **offs 必须是段内相对偏移**：长度 (s1-s0+1)、offs[0]=0、offs[t]=前 t 段的累计
    对数。out 也从下标 0 开始写。
    （曾因这里用全程累计偏移、调用方却按块内大小分配 out，导致越界写堆而进程
    被 0xC0000005 打死；两个约定必须严格一致。）
    """
    n = s1 - s0
    for t in prange(n):
        s = s0 + t
        k0 = s * seg_k
        if k0 > kmax:
            continue
        m = min(seg_k, kmax - k0 + 1)
        if m <= 0:
            continue
        comp = np.zeros(m, dtype=np.uint8)
        _mark_segment(comp, k0, m, primes)
        w = offs[t]
        i = 0
        if k0 == 0:
            i = 1
        for q in range(i, m):
            if comp[q] == 0:
                out[w] = k0 + q
                w += 1


def _seg_bounds(kmax: int, seg_k: int):
    return kmax // seg_k + 1


def count(limit: int, seg_k: int = 3 * 10**5, primes=None, progress=None,
          chunk_segs: int = 48, threads: int = 0) -> int:
    """统计 <= limit 的孪生素数对数 pi2(limit)。"""
    if threads:
        set_num_threads(threads)
    kmax = kmax_for(limit)
    if primes is None:
        primes = small_primes(math.isqrt(limit) + 1)
    nseg = _seg_bounds(kmax, seg_k)
    total = 1                      # (3,5) 是唯一不在 6k±1 格子里的孪生素数对
    t0 = time.perf_counter()
    if progress is not None:
        progress.total = float(kmax + 1)
    for c0 in range(0, nseg, chunk_segs):
        c1 = min(c0 + chunk_segs, nseg)
        total += int(_count_chunk(kmax, primes, seg_k, c0, c1).sum())
        if progress is not None:
            # 成本加权：以 k 空间的覆盖量作为权重
            covered = min((c1) * seg_k, kmax + 1)
            progress.update(covered, extra=f"段 {c1}/{nseg}")
    if progress is not None:
        progress.close(extra=f"pi2={total:,}")
    return total


def enumerate_k(limit: int, seg_k: int = 3 * 10**5, primes=None, progress=None,
                chunk_segs: int = 48, threads: int = 0):
    """枚举 <= limit 的所有孪生素数对，返回排序好的 k 数组（对为 (6k-1, 6k+1)）。"""
    if threads:
        set_num_threads(threads)
    kmax = kmax_for(limit)
    if primes is None:
        primes = small_primes(math.isqrt(limit) + 1)
    nseg = _seg_bounds(kmax, seg_k)
    counts = np.zeros(nseg, dtype=np.int64)
    if progress is not None:
        progress.total = float(kmax + 1)
    for c0 in range(0, nseg, chunk_segs):
        c1 = min(c0 + chunk_segs, nseg)
        counts[c0:c1] = _count_chunk(kmax, primes, seg_k, c0, c1)
        if progress is not None:
            progress.update(min(c1 * seg_k, kmax + 1), extra=f"计数段 {c1}/{nseg}")
    offs = np.zeros(nseg + 1, dtype=np.int64)
    np.cumsum(counts, out=offs[1:])
    out = np.empty(int(offs[-1]), dtype=np.int64)
    base = 0
    for c0 in range(0, nseg, chunk_segs):
        c1 = min(c0 + chunk_segs, nseg)
        bl = np.zeros(c1 - c0 + 1, dtype=np.int64)      # 段内相对偏移
        np.cumsum(counts[c0:c1], out=bl[1:])
        _enum_chunk(kmax, primes, seg_k, c0, c1, bl, out[base:base + int(bl[-1])])
        base += int(bl[-1])
        if progress is not None:
            progress.update(min(c1 * seg_k, kmax + 1), extra=f"枚举段 {c1}/{nseg}")
    if progress is not None:
        progress.close(extra=f"{out.size:,} 对")
    return out


def enumerate_starts(limit: int, seg_k: int = 3 * 10**5, primes=None, progress=None,
                     chunk_segs: int = 48, threads: int = 0) -> np.ndarray:
    """返回 <= limit 的全部孪生素数对的较小者 p（对为 (p, p+2)），升序。

    补齐 (3,5)：3 不在 6k±1 格子里，必须单独加，否则总数比 pi2 少 1。
    用返回值 size 做计数校验时，应与 A007508 完全相等。
    """
    ks = enumerate_k(limit, seg_k, primes, progress, chunk_segs, threads)
    p = 6 * ks - 1
    if limit >= 5:
        p = np.concatenate((np.array([3], dtype=np.int64), p))
    return p


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="孪生格（6k±1）分段筛：统计/枚举孪生素数")
    ap.add_argument("limits", nargs="*", default=["1e8"], help="上界，如 1e8")
    ap.add_argument("--seg-k", type=float, default=1e6, help="每段 k 个数，默认 1e6")
    ap.add_argument("--threads", type=int, default=0, help="线程数，0=全部核心")
    ap.add_argument("--repeat", type=int, default=1, help="重复次数，取最快")
    ap.add_argument("--enumerate", action="store_true", help="同时测枚举（写 int64 数组）")
    ap.add_argument("--no-verify", action="store_true", help="不与 A007508 校验")
    args = ap.parse_args()

    seg_k = int(args.seg_k)
    print(f"孪生格筛：段长 {seg_k:.0e} 个 k（约 {6*seg_k:.0e} 个数），"
          f"线程 {'全部' if not args.threads else args.threads}", flush=True)
    for item in args.limits:
        limit = int(float(item))
        primes = small_primes(math.isqrt(limit) + 1)
        best, got = float("inf"), 0
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            got = count(limit, seg_k, primes, None, 48, args.threads)
            best = min(best, time.perf_counter() - t0)
        line = (f"  ≤{limit:.0e}  用时 {best:7.3f}s  吞吐 {limit/best/1e9:7.3f} G数/秒  "
                f"pi2={got:,}")
        if not args.no_verify and limit in KNOWN:
            line += "  校验 " + ("OK" if got == KNOWN[limit] else f"MISMATCH 应为 {KNOWN[limit]:,}")
        print(line, flush=True)
        if args.enumerate:
            t0 = time.perf_counter()
            ks = enumerate_k(limit, seg_k, primes, None, 48, args.threads)
            el = time.perf_counter() - t0
            print(f"      枚举 {ks.size:,} 对，用时 {el:.3f}s，"
                  f"{ks.size/el/1e6:.1f} M对/秒，数组 {ks.nbytes/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
