#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试：不需要 primesieve、不需要大内存，约 10 秒跑完。

覆盖四件事：
  1. 孪生格筛的计数与 OEIS A007508 精确比对
  2. 枚举边界（含 (3,5) —— 唯一不落在 6k±1 格子里的孪生素数对）
  3. varint 编码 / 解码往返无损
  4. 端到端跑一次 twin_enum.py，并且**强制多块**路径
     （多块路径曾因偏移量约定不一致而越界写堆、进程被系统打死；这条是回归测试）

用法：
    python tests/test_smoke.py

临时产物写在 tests/_tmp/ 下，跑完自动清理，不碰系统临时目录。
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                                       # noqa: E402

import twin_lattice                                      # noqa: E402
import twinio                                            # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


def run_e2e(workdir, out_name="t.varint", seg_k="1e5", block_segs="4"):
    """跑一次 twin_enum.py，返回 (returncode, stderr, 产物路径)。"""
    outp = os.path.join(workdir, out_name)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "twin_enum.py"),
         "--lo", "3", "--hi", "1e7", "--no-text", "--seg-k", seg_k,
         "--block-segs", block_segs, "--sample", "0", "--no-resume",
         "--out", outp],
        capture_output=True, text=True, cwd=workdir)
    return r.returncode, (r.stderr or ""), outp


print("1) 计数与 OEIS A007508 比对")
check("pi2(1e6) == 8169", twin_lattice.count(10 ** 6) == 8169)
check("pi2(1e7) == 58980", twin_lattice.count(10 ** 7) == 58980)

print("\n2) 枚举边界")
p = twin_lattice.enumerate_starts(10 ** 6)
check("对数 == 8169", p.size == 8169, f"实际 {p.size}")
check("前三对起点为 3, 5, 11", p[:3].tolist() == [3, 5, 11], str(p[:3].tolist()))

print("\n3) varint 编码 / 解码往返")
buf = np.empty(p.size * 10 + 16, dtype=np.uint8)
n, _prev, _fa = twinio.encode_deltas(p, 2, 0, True, buf)
back = np.empty(p.size, dtype=np.int64)
m = int(twinio.decode_deltas(buf[:n], 2, back))
check("无损", m == p.size and bool(np.array_equal(back[:m], p)),
      f"{n} 字节 = {n/p.size:.3f} 字节/对")

print("\n4) 端到端 + 多块回归（--seg-k 1e5 --block-segs 4 → 4 块）")
td = os.path.join(HERE, "_tmp")
shutil.rmtree(td, ignore_errors=True)
os.makedirs(td, exist_ok=True)
try:
    code, err, outp = run_e2e(td)
    check("exit code == 0", code == 0, err.strip().splitlines()[-1] if err.strip() else "")
    if os.path.isfile(outp):
        meta, q = twinio.load_pairs(outp)
        check("头部 count == 58980", meta["count"] == 58980, str(meta["count"]))
        check("读回对数一致", q.size == 58980, str(q.size))
        check("首对为 3", int(q[0]) == 3, str(int(q[0])))
        check("末对为 9999971", int(q[-1]) == 9999971, str(int(q[-1])))
    else:
        check("产物已生成", False, outp)
finally:
    shutil.rmtree(td, ignore_errors=True)
    # 清掉本次测试在仓库根留下的 state / log
    for d, ext in (("state", ".state.json"), ("logs", ".log")):
        f = os.path.join(ROOT, d, "t.varint" + ext)
        if os.path.isfile(f):
            os.remove(f)

print()
if fails:
    print(f"失败 {len(fails)} 项：{fails}")
    sys.exit(1)
print("全部通过 OK")
