#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""孪生素数产物的落盘格式：增量 varint 编码 / 解码 / 头部 / 校验和。

文件格式
--------
  [0, 1024)     固定长度头部：UTF-8 JSON，用空格补齐到 1024 字节（可在收尾时原地重写）
  [1024, EOF)   数据体：varint(LEB128) 流
                首值 = 绝对对起点 p（含 (3,5) 的 3）
                其后 = (p_i - p_{i-1}) // 2   （孪生素数相邻起点之差必为偶数，除 2 省 1 字节）

实测压缩率 1.106（1e6）~ 1.415（1e9）字节/对，优于定长 8 字节约 5.7 倍。
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from numba import njit

MAGIC = "TWINPRIMES1"
HEADER_LEN = 1024


@njit(cache=True)
def encode_deltas(vals, div, prev, first_absolute, out):
    """把有序 int64 序列编码进 out；返回 (写出字节数, 新的 prev, 新的 first_absolute)。

    first_absolute=True 时第一个值按绝对值写，之后按 (v-prev)//div 写。
    """
    n = 0
    p = prev
    fa = first_absolute
    for i in range(vals.size):
        v = vals[i]
        if fa:
            d = v
            fa = False
        else:
            d = (v - p) // div
        p = v
        while d >= 0x80:
            out[n] = (d & 0x7F) | 0x80
            n += 1
            d >>= 7
        out[n] = d
        n += 1
    return n, p, fa


@njit(cache=True)
def decode_deltas(buf, div, out):
    """把 varint 数据体解码成有序 int64 序列，返回元素个数。"""
    n = 0
    i = 0
    m = buf.size
    prev = 0
    first = True
    while i < m:
        shift = 0
        d = 0
        while True:
            b = buf[i]
            i += 1
            d |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        if first:
            prev = d
            first = False
        else:
            prev = prev + d * div
        out[n] = prev
        n += 1
    return n


@njit(cache=True)
def _put_uint(v, buf, n):
    """把非负整数按十进制写入 buf[n:]，返回新的 n。"""
    if v == 0:
        buf[n] = 48
        return n + 1
    start = n
    while v > 0:
        buf[n] = 48 + v % 10
        v //= 10
        n += 1
    i, j = start, n - 1
    while i < j:
        t = buf[i]
        buf[i] = buf[j]
        buf[j] = t
        i += 1
        j -= 1
    return n


@njit(cache=True)
def encode_pairs_text(p, buf):
    """把 p 数组写成文本行 "p p+2\\n" 到 buf，返回字节数。

    纯 numba 写字节，避免 Python 逐行格式化——1e11 有 2.24 亿行，
    用 f-string 逐行写要 3~4 分钟，这里只需数秒。
    """
    n = 0
    for i in range(p.size):
        n = _put_uint(p[i], buf, n)
        buf[n] = 32
        n += 1
        n = _put_uint(p[i] + 2, buf, n)
        buf[n] = 10
        n += 1
    return n


def write_header(fh, meta: dict) -> None:
    """在文件开头原地写入定长 JSON 头部（收尾时重写，用于补 count/sha256）。"""
    raw = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(raw) > HEADER_LEN:
        raise ValueError(f"头部超长 {len(raw)} > {HEADER_LEN}")
    fh.seek(0)
    fh.write(raw + b" " * (HEADER_LEN - len(raw)))
    fh.flush()


def read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read(HEADER_LEN)
    meta = json.loads(raw.decode("utf-8").rstrip())
    if meta.get("magic") != MAGIC:
        raise ValueError(f"{path} 不是本程序的产物（magic={meta.get('magic')!r}）")
    return meta


def sha256_body(path: str) -> str:
    """对 [HEADER_LEN, EOF) 的数据体做 SHA256。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(HEADER_LEN)
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_pairs(path: str) -> tuple[dict, np.ndarray]:
    """读回产物：返回 (头部, 全部对起点 p 的数组)。"""
    meta = read_header(path)
    size = os.path.getsize(path) - HEADER_LEN
    buf = np.empty(size, dtype=np.uint8)
    with open(path, "rb") as fh:
        fh.seek(HEADER_LEN)
        buf[:] = np.frombuffer(fh.read(size), dtype=np.uint8)
    out = np.empty(meta["count"], dtype=np.int64)
    n = int(decode_deltas(buf, 2, out))
    return meta, out[:n]
