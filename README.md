# twinprime-enum

连续枚举给定区间内的**全部孪生素数**并落盘：紧凑二进制（约 1.5 字节/对）+ 人读文本，带断点续跑与三重交叉校验。

使用 Python + numba 写的自包含实现。

## 使用方法
依赖安装：

```bash
pip install -r requirements.txt        # numpy + numba
```

使用示例：

```bash
python twin_enum.py --lo 3 --hi 1e8            # 枚举到 10^8，约 1 秒
python twin_enum.py --lo 3 --hi 1e11           # 枚举到 10^11，约 30 秒
python twin_enum.py --lo 1e9 --hi 1e10         # 只算区间 (10^9, 10^10]
python twin_enum.py --lo 3 --hi 1e8 --no-text  # 不产出文本版
```

程序自检：

```bash
python tests/test_smoke.py
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `--lo` / `--hi` | 区间边界，默认 `3` / `1e10` |
| `--seg-k` | 每段 k 个数，默认 `5e5`，**实测最优区间是 $ 3\times 10^5 \sim 5\times 10^5 $,调到 $ 3\times 10^6 $ 会掉进缓存外、慢约 10 倍** |
| `--block-segs` | 每块含多少段，决定内存占用与 state 更新粒度，默认 256 |
| `--threads` | 线程数，默认用满所有核心 |
| `--text` / `--no-text` | 是否产出人读文本版（默认开） |
| `--no-resume` | 忽略断点，从头重算 |
| `--sample N` | 收尾对 N 个命中项做 sympy 素性抽样，默认 `200` |

## 输出位置

```
data/TwinPrimes_1e+11.varint    主产物
data/TwinPrimes_1e+11.txt       人读文本，每行 "p p+2"
state/<产物名>.state.json        断点状态
logs/<产物名>.log                运行日志（逐行flush）
```

进度：`stdout` 是终端时画单行动态进度条（含速度、已用时间、ETA），被重定向或放后台时自动退化为周期性文本行。另开窗口运行 `python watch_progress.py` 可旁观进度。

进度条的分母是 $ k $ 而非数字：孪生格把候选压缩成 $ k $，每个 $ k $ 管6个数（6k−1 与 6k+1）。

## 实测性能（本机 16 逻辑核 / Windows / numba 0.62）

| 项目 | 本工具 | primesieve 12.15 |
|---|---|---|
| 计数 $ \pi_{2}(10^{10}) $，16 线程 | 14.33 G数/秒 | **46.04 G数/秒** |
| 计数 $ \pi_{2}(10^{10}) $，单线程 | 2.06 G数/秒 | **5.81 G数/秒** |
| 枚举配对 | **22.6 M对/秒** | 4.0 M对/秒 |
| 端到端 $ 10^{11} $ | 27.4 s / 224,376,048 对 | —— |

$ 10^{11} $ 端到端包含：写 351.8 MB 二进制 + 5.32 GB 文本，以及收尾校验。

**压缩率：**

| 上界 | 字节/对 | 对比int64 |
|---|---|---|
| $ 10^{6} $| 1.106 |$ 7.2\times $|
| $ 10^{9} $| 1.415 |$ 5.7\times $|
| $ 10^{11} $| 1.568 |$ 5.1\times $|

## 算法概览

### 孪生格：

除 2、3 外的素数 $ p\equiv \pm1 \pmod 6 $，于是每个 $ k \ge 1 $ 对应唯一一对候选 $ (6k−1, 6k+1) $。

标记规则（素数 $ p \ge 5$，`inv6` 为 6 模 $ p $ 的逆元）：

```
6k−1 ≡ 0 (mod p)   ⟺   k ≡  inv6  (mod p)
6k+1 ≡ 0 (mod p)   ⟺   k ≡ −inv6  (mod p)
```

两条公差为 $ p $ 的等差数列，起点统一取 $ k \ge \left\lfloor \frac{p^2 - 1}{6} \right\rfloor $。这个阈值同时覆盖两条链，
确保不将 $ p $ 自身误标为合数（$ p^2 $ 是 $ p $ 在该链上的第一个合数）。

例外：`(3,5)`不在 $ 6k\pm1 $ 格子里，单独补在最前。

### 并行

段与段完全独立，用 numba `prange` 并行，每段写自己的输出槽位再汇总。

## 输出格式

```
[0, 1024)     定长 JSON 头（十进制 ASCII 补齐）
              magic / lo / hi / seg_k / count / body_bytes / sha256_body /
              definition / encoding / created
[1024, EOF)   varint(LEB128) 数据体
                首值 = 绝对起点 p
                其后 = (p_i − p_{i−1}) // 2
```


读回来（`twinio` 是仓库内的模块）：

```python
import twinio
meta, p = twinio.load_pairs("data/TwinPrimes_1e+11.varint")
print(meta["count"], p[:5])      # 224376048 [3 5 11 17 29]
```

## 正确性验证

1. **OEIS A007508 比对：** 整段（`--lo 3`）且上界是 10 的幂时，总数必须完全相等。
2. **primesieve 独立计数：** 用另一个完全独立的实现算同一区间做对照（可选依赖）。
3. **sympy 抽样复核：** 对命中项用大整数重算、并确认 $ p $ 与 $ p+2 $ 都是素数。


## 局限

- 只在给定区间内工作，不做增量索引，要查任意位置的孪生素数需要重新扫。
- 校验阶段会把整个产物读回内存解码：$ 10^{11} $ 时约 1.8 GB（2.24 亿个 int64）。
- 分段筛的复杂度随上界线性增长，本项目面向 $ 10^9 \sim 10^{12} $ 量级。

## 依赖

运行必需：`numpy`、`numba`。可选：`sympy`（抽样复核）、`primesieve`（独立计数）。
可选项缺失时对应校验自动跳过并在日志里说明，不会中断枚举。


## 许可

见 `LICENSE`。若你在本项目中链接或再分发 primesieve，
需一并遵守其 [BSD-2-Clause](https://github.com/kimwalisch/primesieve/blob/master/COPYING) 许可。
