# -*- coding: utf-8 -*-
import os
import re
import math
import struct
import logging
import time
import tracemalloc
from contextlib import contextmanager
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# --- plotting ---
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator, NullFormatter, FixedLocator

# Try to make Chinese text display correctly on Windows.
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def format_height_um(v: float) -> str:
    """高度显示规则（μm，不含单位）：

    - |v| > 1000：保留到个位（无小数）
    - |v| <= 1000：保留 1 位小数
    """
    try:
        if v is None:return "NA"
        v = float(v)
        if not math.isfinite(v):return "NA"
        if abs(v) > 1000.0:return f"{v:.0f}"
        return f"{v:.1f}"
    except Exception:return "NA"


# =========================
# 解密核心
# 说明：不做后缀检测；只要“能解密”就解密。
# 核心逻辑 UTF-16LE 头部 + float32(<5f>) 数据表解析，输出为可读的 StartOfData + 5列纯数字。
# =========================

def _tajm_find_header_end(buf: bytes, log=None) -> int:
    """
    头部是 UTF-16LE 文本（常见 ASCII+0 的形式），通常以 CRLF 结束，然后进入二进制区。
    从前 64KB 内寻找最后一个仍然像“文本头部”的 CRLF 位置作为头部结束。
    """
    crlf = b"\r\x00\n\x00"
    scan_limit = min(len(buf), 65536)

    last_good = None
    pos = 0
    hits = 0

    while True:
        p = buf.find(crlf, pos, scan_limit)
        if p == -1:break
        hits += 1
        candidate = p + len(crlf)
        window = buf[candidate:candidate + 512]
        if len(window) < 64:break
        odd = window[1::2]  # UTF-16LE 高字节
        zero_ratio = odd.count(0) / max(1, len(odd))
        if log:log(f"[tajm][header-scan] CRLF@{p} -> candidate={candidate}, odd_zero_ratio={zero_ratio:.3f}")
        if zero_ratio >= 0.55:
            last_good = candidate
            pos = candidate
        else:break
    if log:log(f"[tajm][header-scan] scanned_bytes={scan_limit}, crlf_hits={hits}, header_end={last_good}")
    if last_good is None:raise RuntimeError("TAJM: 未能定位 UTF-16LE 头部结束位置（找不到可靠的 CRLF 终点）。")
    return last_good


def _tajm_find_data_start_by_first_time(buf: bytes, search_from: int, log=None) -> int:
    """
    数据区通常为 5 个 little-endian float32 组成的记录（20 字节/行）。
    优先寻找 time=0.004 的 float32；找不到则用启发式扫描。
    """
    needle = struct.pack("<f", 0.004)
    p = buf.find(needle, search_from)

    if log:log(f"[tajm][data-find] try needle float32(0.004) from offset {search_from} -> pos={p}")

    if p != -1:return p

    scan_to = min(len(buf) - 40, search_from + 4096)
    if log:log(f"[tajm][data-find] fallback scan range [{search_from}, {scan_to}]")

    for start in range(search_from, scan_to):
        try:t, temp, dim, force, flow = struct.unpack_from("<5f", buf, start)
        except struct.error:break
        if not all(map(math.isfinite, [t, temp, dim, force, flow])):continue

        # 粗略范围约束（参考 TAJM.py）
        if not (0 <= t <= 1e6 and -500 <= temp <= 2000 and -1e7 <= dim <= 1e7 and
                -1e6 <= force <= 1e6 and -1e6 <= flow <= 1e6):
            continue

        # 再看下一行 time 是否非递减
        try:t2 = struct.unpack_from("<f", buf, start + 20)[0]
        except struct.error:continue

        if math.isfinite(t2) and t2 >= t:
            if log:log(f"[tajm][data-find] fallback hit at offset {start} -> first_row=(t={t}, temp={temp}, ..), next_t={t2}")
            return start
    raise RuntimeError("TAJM: 未能定位数据区起点（float32 表）。")


def _tajm_parse_rows(buf: bytes, start: int, log=None, max_rows: int = 200000):
    """
    从 start 开始按 <5f> 读取记录，直到遇到不合理数据停止。
    """
    rows = []
    pos = start
    prev_t = -1e18
    stops = 0

    while pos + 20 <= len(buf) and len(rows) < max_rows:
        t, temp, dim, force, flow = struct.unpack_from("<5f", buf, pos)

        if not all(map(math.isfinite, [t, temp, dim, force, flow])):
            stops += 1
            if log:
                log(f"[tajm][parse] stop(non-finite) at pos={pos}, row_index={len(rows)}")
            break

        if not (0 <= t <= 1e6 and -500 <= temp <= 2000 and -1e7 <= dim <= 1e7 and
                -1e6 <= force <= 1e6 and -1e6 <= flow <= 1e6):
            stops += 1
            if log:
                log(f"[tajm][parse] stop(out-of-range) at pos={pos}, row_index={len(rows)} "
                    f"values=(t={t}, temp={temp}, dim={dim}, force={force}, flow={flow})")
            break

        if t < prev_t - 1e-3:
            stops += 1
            if log:
                log(f"[tajm][parse] stop(time-not-monotonic) at pos={pos}, row_index={len(rows)} t={t} prev_t={prev_t}")
            break

        prev_t = t
        rows.append((t, temp, dim, force, flow))
        pos += 20

        if log and len(rows) in (1, 2, 3, 10, 100, 500, 1000, 2000, 5000, 10000):
            log(f"[tajm][parse] progress rows={len(rows)} latest=(t={t}, temp={temp}, dim={dim}, force={force}, flow={flow}) pos={pos}")

    if log:
        log(f"[tajm][parse] done rows={len(rows)}, start={start}, end_pos={pos}, stops={stops}")

    return rows


def try_tajm_decrypt_to_text(in_path: str, log=None) -> dict:
    """
    尝试用 TAJM 解密思路把“二进制/不可读”文件转成可读取的文本。
    - 不做后缀检测（按你的要求）
    - 只判断能否成功解密：成功则返回 ok=True + text + rows 等信息；失败返回 ok=False
    """
    t0 = datetime.datetime.now()
    r = {"ok": False, "input": in_path}

    try:
        if log:
            log(f"[tajm] try decrypt: {in_path}")

        if not os.path.isfile(in_path):
            raise FileNotFoundError(f"找不到文件：{in_path}")

        with open(in_path, "rb") as f:buf = f.read()

        if len(buf) < 64:raise RuntimeError("文件太小，疑似不是 TAJM 可解密格式。")

        if log:
            log(f"[tajm][io] file_size={len(buf)} bytes; leading_bytes={buf[:8].hex(' ')}")

        header_end = _tajm_find_header_end(buf, log=log)
        header_text = buf[:header_end].decode("utf-16le", errors="ignore")

        data_start = _tajm_find_data_start_by_first_time(buf, header_end, log=log)
        rows = _tajm_parse_rows(buf, data_start, log=log)

        if not rows:
            raise RuntimeError("TAJM: 未解析到任何数据记录。")

        if not header_text.endswith("\r\n"):
            header_text += "\r\n"

        out_lines = [header_text.rstrip("\r\n"), "StartOfData"]
        for t, temp, dim, force, flow in rows:
            out_lines.append(f"{t:.6f}\t{temp:.5f}\t{dim:.6f}\t{force:.8f}\t{flow:.5f}")
        out_text = "\r\n".join(out_lines) + "\r\n"

        dt = (datetime.datetime.now() - t0).total_seconds()
        r.update({
            "ok": True,
            "text": out_text,
            "rows": len(rows),
            "header_end": header_end,
            "data_start": data_start,
            "elapsed": dt,
        })
        if log:
            log(f"[tajm] ✅ decrypt ok rows={len(rows)} header_end={header_end} data_start={data_start} elapsed={dt:.3f}s")
        return r

    except Exception as e:
        if log:
            log(f"[tajm] ❌ decrypt failed: {e}")
        r["err"] = str(e)
        return r


def load_text_prefer_tajm(in_path: str, logger: logging.Logger) -> Tuple[str, dict]:
    """
    统一入口：优先尝试 TAJM 解密（能解密就解密），否则走原来的自动编码读取。
    ⚠️ 但如果文件本身就是“可读文本且包含 StartOfData”，则直接按纯文本读取，
       避免把 UTF-16LE 文本误当作 TAJM 二进制去解密（会导致数据全是 e-39 这类垃圾值）。
    返回：(text, meta)
      meta: {"used": "tajm"/"plain", ...}
    """
    # 0) 先快速判断：文件是否已经是 UTF-16LE 文本且含 StartOfData
    try:
        with open(in_path, "rb") as f:
            head = f.read(256 * 1024)  # 读前 256KB 足够判断
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            try:
                probe = head.decode("utf-16", errors="ignore")
                if "StartOfData" in probe:
                    text = read_text_auto(in_path, logger)
                    return text, {"used": "plain", "reason": "already_has_StartOfData"}
            except Exception:
                pass
    except Exception:
        # 判断失败不影响后续流程
        pass

    # 1) 再尝试解密（不做后缀判断）
    r = try_tajm_decrypt_to_text(in_path, log=logger.debug)
    if r.get("ok"):
        return r["text"], {"used": "tajm", **r}

    # 2) 回退到原有文本读取
    text = read_text_auto(in_path, logger)
    return text, {"used": "plain"}


# =========================
# 可调参数
# =========================
START_TEMP_FOR_T1 = 60.0   # 你说“数据开始(60℃附近)”，默认从>=60开始找T1
MONOTONIC_FIX_T1_T2 = True  # 修复：强制在[T1,T2]区间高度单调不下降，避免出现“裂开点”
MONOTONIC_METHOD = "bridge"  # bridge(线性桥接掉点) / isotonic(平滑) / cummax(硬钳制)
# 如果文件没有 StartOfData 标记（极少数情况），用温度范围过滤掉头部乱入的数字。
FALLBACK_TEMP_MIN = 40.0
FALLBACK_TEMP_MAX = 400.0
NUM_EXT_PATTERNS = tuple(f"*.{i:03d}" for i in range(1, 1000))  # 兼容 Windows 文件对话框过滤
FLOAT_FIND_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class ParsedLine:
    raw: str
    nums: List[float]               # 本行所有“数字token”
    spans: List[Tuple[int, int]]    # 每个数字token在原行中的 (start,end)


def read_utf16le_no_bom(path: str) -> str:
    """强制按 UTF-16LE 读取，并去掉文件开头的 BOM（若存在）。"""
    with open(path, "rb") as f:
        data = f.read()
    # UTF-16LE BOM: FF FE
    if data[:2] == bytes((255, 254)):
        data = data[2:]
    text = data.decode("utf-16le", errors="strict")
    # 保险：极少数情况下解码后仍残留 BOM 字符
    bom = chr(65279)
    if text.startswith(bom):
        text = text[len(bom):]
    return text


def write_utf16le_with_bom(path: str, text: str) -> None:
    """以 UTF-16LE 写入，并写入 BOM(FF FE)。

    说明：
    - 你之前要求“去除 BOM”时，我们输出的是“UTF-16LE 无 BOM”。
      但不少 Windows 软件/编辑器会因为没有 BOM 而把文件显示成 ANSI/ASCII。
    - 如果你希望它被稳定识别为“UTF-16 LE”，就需要保留 BOM。
    """
    bom_char = chr(65279)
    if text.startswith(bom_char):
        text = text[len(bom_char):]

    data = b"\xff\xfe" + text.encode("utf-16le")
    with open(path, "wb") as f:
        f.write(data)




def read_text_auto(path: str, logger: Optional[logging.Logger] = None) -> str:
    """尽量自动识别编码读取文本（参考 ces.py 的“多编码尝试 + 记录日志”思路）。

    尝试顺序：
    1) UTF-16LE（二进制解码，可带/不带 BOM）
    2) UTF-16（由 Python 依据 BOM 自动选择端序）
    3) UTF-8
    4) GBK

    返回：
    - 解码后的文本（如文本首字符为 BOM 字符，会被移除）
    """
    # 1) utf-16le：允许有/无 BOM
    try:
        with open(path, "rb") as f:
            data = f.read()
        had_bom = data[:2] == b"\xff\xfe"
        if had_bom:
            data2 = data[2:]
        else:
            data2 = data
        text = data2.decode("utf-16le", errors="strict")
        bom_char = chr(65279)
        if text.startswith(bom_char):
            text = text[len(bom_char):]
        if logger:
            logger.debug(f"读取文件成功：编码=utf-16le，BOM={'有' if had_bom else '无'}，字节数={len(data)}")
        return text
    except Exception as e:
        if logger:
            logger.debug(f"尝试 utf-16le 读取失败：{e}，继续尝试 utf-16/utf-8/gbk")

    # 2~4) 其他编码
    for enc in ["utf-16", "utf-8", "gbk"]:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                text = f.read()
            bom_char = chr(65279)
            if text.startswith(bom_char):
                text = text[len(bom_char):]
            if logger:
                logger.debug(f"读取文件成功：编码={enc}，字符数={len(text)}")
            return text
        except Exception as e:
            if logger:
                logger.debug(f"尝试 {enc} 读取失败：{e}")

    raise UnicodeDecodeError("unknown", b"", 0, 1, "无法识别文件编码（utf-16le/utf-16/utf-8/gbk 均失败）")


def write_utf16le_no_bom(path: str, text: str) -> None:
    """以 UTF-16LE 写入（不写 BOM），与 ces.py 的输出方式一致。"""
    bom_char = chr(65279)
    if text.startswith(bom_char):
        text = text[len(bom_char):]
    with open(path, "w", encoding="utf-16le", newline="") as f:
        f.write(text)

def parse_lines(text: str) -> List[ParsedLine]:
    out = []
    for line in text.splitlines(keepends=True):
        spans = []
        nums = []
        for m in FLOAT_FIND_RE.finditer(line):
            spans.append((m.start(), m.end()))
            try:
                nums.append(float(m.group()))
            except ValueError:
                nums.append(math.nan)
        out.append(ParsedLine(raw=line, nums=nums, spans=spans))
    return out


def extract_xy_from_file(parsed: List[ParsedLine], logger: logging.Logger):
    """从 ParsedLine 中提取 (温度x, 高度y)。

    关键：TA Instruments/TA Trios 这类导出的 txt 前面通常有大量 header 行，里面也会包含很多数字。
    我们必须避免把 header 行当成数据行，否则 T2 会被 header 里的数字“污染”。

    ✅ 这里按更严格的“数据块”规则读取（解决 StartOfData 后仍混入非数据行、导致第三列最大值异常的问题）：
    - 先定位包含 'StartOfData' 的行（忽略大小写/前后空白）。
    - 从其下一行开始，只读取“看起来像真实数据行”的连续区段：
        * 至少 5 个数字（TMA 常见为 5 列：Time, Temp, Dim, Force, Flow）
        * 第2个=温度，第3个=高度（与 tma_tem_right_v1.py 一致）
      一旦在已开始读取数据后遇到不满足条件的行，就认为数据块结束并停止（避免尾部/二段 header 继续污染）。

    - 若文件没有 StartOfData（极少数）：回退到温度范围过滤（x 在 [FALLBACK_TEMP_MIN, FALLBACK_TEMP_MAX]），
      同时同样要求至少 5 个数字。
    """
    xs: List[float] = []
    ys: List[float] = []
    line_idxs: List[int] = []

    marker_idx: Optional[int] = None
    for i, pl in enumerate(parsed):
        if 'startofdata' in (pl.raw or '').lower():
            marker_idx = i
            break

    def is_data_line(pl: ParsedLine) -> bool:
        # 真实数据行通常至少 5 列数字：t, temp, dim, force, flow
        return (
            len(pl.nums) >= 5
            and math.isfinite(pl.nums[1])
            and math.isfinite(pl.nums[2])
        )

    if marker_idx is not None:
        started = False
        for idx in range(marker_idx + 1, len(parsed)):
            pl = parsed[idx]
            if is_data_line(pl):
                started = True
                xs.append(pl.nums[1])
                ys.append(pl.nums[2])
                line_idxs.append(idx)
            else:
                if started:
                    # ✅ 数据块结束（遇到非数据行就停止）
                    break

        logger.debug(f"检测到 StartOfData：数据起始行={marker_idx+1}，数据行数={len(xs)}")
        return xs, ys, line_idxs

    # ❗极少数文件没有 StartOfData：回退过滤（按温度范围）
    for idx, pl in enumerate(parsed):
        if is_data_line(pl):
            x = pl.nums[1]
            if FALLBACK_TEMP_MIN <= x <= FALLBACK_TEMP_MAX:
                xs.append(x)
                ys.append(pl.nums[2])
                line_idxs.append(idx)

    logger.debug(
        f"未检测到 StartOfData：使用温度范围过滤 [{FALLBACK_TEMP_MIN},{FALLBACK_TEMP_MAX}]，数据行数={len(xs)}"
    )
    return xs, ys, line_idxs

def detect_T2_T1(xs: List[float], ys: List[float], logger: logging.Logger) -> Tuple[float, float]:
    """按用户定义自动识别 T1/T2（StartOfData 后每行 5 个数字：第2个=温度，第3个=高度）。

    逻辑与 tma_tem_right_v1.py 保持一致：
    - T2：第三列(高度)的最大值所对应的第二列(温度)。
    - T1：在 x>=START_TEMP_FOR_T1 且 x<=T2 的范围内，第三列(高度)的最小值所对应的第二列(温度)。
      若该范围内没有点，则退化为 x<=T2 的范围内寻找最小值。
    """
    if not xs:raise ValueError("文件中没有可用数据点（无法解析温度/高度）。")

    # T2：高度最大值对应温度
    max_i = max(range(len(ys)), key=lambda i: ys[i])
    T2 = xs[max_i]
    y_max = ys[max_i]

    # T1：从 x>=START_TEMP_FOR_T1 到 x<=T2 范围内的高度最小值对应温度
    candidates = [i for i in range(len(xs)) if xs[i] >= START_TEMP_FOR_T1 and xs[i] <= T2]
    if not candidates:
        # 如果没有>=60的点（比如数据从50开始且很短），退化为从最小温度到T2
        candidates = [i for i in range(len(xs)) if xs[i] <= T2]
    if not candidates:
        # 理论兜底：全量最小
        candidates = list(range(len(xs)))

    min_i = min(candidates, key=lambda i: ys[i])
    T1 = xs[min_i]
    y_min = ys[min_i]

    logger.debug(f"T2: max(height) 索引={max_i}, 温度={T2}, 高度={y_max}")
    logger.debug(
        f"T1: min(height) 索引={min_i}, 温度={T1}, 高度={y_min} "
        f"(搜索区间: x>= {START_TEMP_FOR_T1} 且 x<=T2, 候选点数={len(candidates)})"
    )
    return T1, T2


def piecewise_map(x: float, xmin: float, xmax: float, T1o: float, T2o: float, T1n: float, T2n: float) -> float:
    # 分三段线性映射： [xmin,T1o]->[xmin,T1n], [T1o,T2o]->[T1n,T2n], [T2o,xmax]->[T2n,xmax]
    if x <= T1o:
        if abs(T1o - xmin) < 1e-12:return x + (T1n - T1o)
        return xmin + (x - xmin) * (T1n - xmin) / (T1o - xmin)
    elif x <= T2o:
        if abs(T2o - T1o) < 1e-12:return T1n
        return T1n + (x - T1o) * (T2n - T1n) / (T2o - T1o)
    else:
        if abs(xmax - T2o) < 1e-12:
            return x + (T2n - T2o)
        return T2n + (x - T2o) * (xmax - T2n) / (xmax - T2o)


def linear_interp(x: float, xp: List[float], fp: List[float]) -> float:
    # xp 递增；越界夹到边界
    if x <= xp[0]:return fp[0]
    if x >= xp[-1]:return fp[-1]

    lo, hi = 0, len(xp) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xp[mid] <= x:lo = mid
        else:hi = mid

    x0, x1 = xp[lo], xp[lo + 1]
    y0, y1 = fp[lo], fp[lo + 1]
    if abs(x1 - x0) < 1e-12:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def aggregate_duplicates(xp: List[float], fp: List[float], ndigits: int = 8) -> Tuple[List[float], List[float]]:
    # 合并近似相同x（避免插值xp非严格递增）
    bucket = {}
    for x, y in zip(xp, fp):
        k = round(x, ndigits)
        bucket.setdefault(k, []).append(y)
    xs = sorted(bucket.keys())
    ys = [sum(bucket[k]) / len(bucket[k]) for k in xs]
    return xs, ys


def isotonic_regression(y, eps: float = 1e-12):
    """PAV isotonic regression (non-decreasing). Returns a new list."""
    n = len(y)
    if n <= 1:
        return list(y)

    # blocks: [start, end, weight, avg]
    blocks = []
    for i, yi in enumerate(y):
        blocks.append([i, i, 1.0, float(yi)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3] + eps:
            b2 = blocks.pop()
            b1 = blocks.pop()
            w = b1[2] + b2[2]
            avg = (b1[3] * b1[2] + b2[3] * b2[2]) / w
            blocks.append([b1[0], b2[1], w, avg])

    out = [0.0] * n
    for s, e, w, avg in blocks:
        for k in range(s, e + 1):
            out[k] = avg
    return out

def bridge_dropouts(x_seg, y_seg, tol):
    """    在已按温度升序排列的序列上修复“掉点”：
    - 找到低于历史最高值的连续区段
    - 用区段左右两端的正常点做线性插值抬高填补

    只抬高，不降低；最后再做一次非递减保证。
    """
    n = len(y_seg)
    if n == 0:
        return []

    y = list(y_seg)
    running_max = y[0]
    bad_start = None

    for j in range(1, n):
        if y[j] < running_max - tol:
            if bad_start is None:
                bad_start = j
        else:
            if bad_start is not None:
                L = bad_start - 1
                R = j
                xL, yL = x_seg[L], y[L]
                xR, yR = x_seg[R], y[R]
                if yR < yL:
                    yR = yL
                denom = (xR - xL)
                for k in range(bad_start, R):
                    if abs(denom) < 1e-12:
                        y_target = yL
                    else:
                        t = (x_seg[k] - xL) / denom
                        y_target = yL + t * (yR - yL)
                    if y[k] < y_target:
                        y[k] = y_target
                bad_start = None
            running_max = max(running_max, y[j])

    # 掉点到末尾：没有右锚点，只能抬到 running_max
    if bad_start is not None:
        for k in range(bad_start, n):
            if y[k] < running_max:
                y[k] = running_max

    # 最终保证非递减
    m = -float('inf')
    for j in range(n):
        if y[j] > m:m = y[j]
        else:y[j] = m
    return y


def enforce_monotonic_between(xs: List[float], ys: List[float], x1: float, x2: float, logger: logging.Logger):
    """
    修复“裂开点”：在温度区间 [x1, x2] 内强制高度 ys 随温度 xs 单调不下降。
    - xs/ys 为同长度列表（按文件行顺序）。
    - 只对区间内的点做修正；区间外保持不变。
    """
    if x1 >= x2:return ys

    idxs = [i for i, x in enumerate(xs) if x1 <= x <= x2]
    if len(idxs) < 2:return ys

    # 按温度升序做单调修正（避免温度不严格递增导致误判）
    idxs_sorted = sorted(idxs, key=lambda i: xs[i])
    seg = [ys[i] for i in idxs_sorted]

    tol = max(1e-9, (max(seg) - min(seg)) * 1e-6)
    method = (MONOTONIC_METHOD or "isotonic").lower()
    if method == "cummax":
        fixed = []
        m = -float('inf')
        for v in seg:
            if v > m:m = v
            fixed.append(m)
    elif method in ("bridge", "interp", "linear"):  # 线性桥接掉点
        xseg = [xs[i] for i in idxs_sorted]
        fixed = bridge_dropouts(xseg, seg, tol)
    else:
        fixed = isotonic_regression(seg)

    ys2 = list(ys)
    changed = 0
    max_abs_delta = 0.0
    for i, v in zip(idxs_sorted, fixed):
        d = v - ys2[i]
        if abs(d) > 1e-10:
            changed += 1
            if abs(d) > max_abs_delta:
                max_abs_delta = abs(d)
        ys2[i] = v

    if changed:logger.debug(f"单调修复已应用：区间[{x1}, {x2}] 内修改点数={changed}, 最大改变量={max_abs_delta}")
    else:logger.debug(f"单调修复检查：区间[{x1}, {x2}] 内无需修改")
    return ys2


def format_like(original_num_str: str, value: float) -> str:
    # 尽量保留原数字的小数位风格
    if "e" in original_num_str.lower():
        return f"{value:.8e}"
    if "." in original_num_str:
        decimals = len(original_num_str.split(".")[-1])
        return f"{value:.{decimals}f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def apply_warp(parsed: List[ParsedLine], T1n: float, T2n: float, mg: float, logger: logging.Logger) -> Tuple[str, float, float]:
    """
    ✅ 规则核验：
    - 一行有 5 个数字：取第2/第3个数字（index 1/2）
    - 一行有 4 个数字：取第2/第3个数字（index 1/2）
    - 一行有 3 个数字：取第2/第3个数字（index 1/2）
    - 少于 3 个数字：不处理该行
    """
    # ✅ 只从 StartOfData 后提取真实数据段，避免 header 数字干扰（这是 T2 识别错误的根因）
    xs, ys, data_line_indices = extract_xy_from_file(parsed, logger)

    if not xs:raise ValueError("没有找到有效数据行：每行至少要有3个数字，才能取第2列温度和第3列高度。")

    xmin, xmax = min(xs), max(xs)
    T1o, T2o = detect_T2_T1(xs, ys, logger)

    logger.debug(f"数据点统计：点数={len(xs)}，温度范围=[{xmin}, {xmax}]")
    logger.debug(f"原始高度范围：[{min(ys)}, {max(ys)}]")

    # 校验新T1/T2
    if not (xmin <= T1n <= xmax and xmin <= T2n <= xmax):
        raise ValueError(f"新T1/T2必须在温度范围内 [{xmin}, {xmax}]，当前 T1={T1n}, T2={T2n}")
    if T1n >= T2n:
        raise ValueError(f"新T1必须小于新T2，当前 T1={T1n}, T2={T2n}")

    # 1) 原x -> 新x'
    x_prime = [piecewise_map(x, xmin, xmax, T1o, T2o, T1n, T2n) for x in xs]

    # 2) (x', y) -> 在原x上插值得到新高度
    pairs = sorted(zip(x_prime, ys), key=lambda p: p[0])
    xp = [p[0] for p in pairs]
    fp = [p[1] for p in pairs]
    xp, fp = aggregate_duplicates(xp, fp, ndigits=8)

    y_new = [linear_interp(x, xp, fp) for x in xs]

    # ✅ 修复：确保从T1到T2单调上升（非下降），避免出现单点下凹导致曲线‘裂开’
    if MONOTONIC_FIX_T1_T2:y_new = enforce_monotonic_between(xs, y_new, T1n, T2n, logger)

    # 2.5) ✅ 按 mg 做高度缩放（参考 ces.py：new_h = h * 0.5 / mg）
    if mg <= 0:raise ValueError(f"mg 必须大于0,当前 mg={mg}")
    scale = 0.5 / mg
    y_new = [v * scale for v in y_new]
    logger.debug(f"已应用 mg 缩放:mg={mg}，缩放系数=0.5/mg={scale}")

    # 3) ✅ 写回：只替换“第3个数字token”（高度）
    new_lines = [pl.raw for pl in parsed]
    for local_i, line_idx in enumerate(data_line_indices):
        pl = parsed[line_idx]
        span = pl.spans[2]  # ✅ 第3个数字的原始位置
        old_str = pl.raw[span[0]:span[1]]
        new_str = format_like(old_str, y_new[local_i])
        new_lines[line_idx] = pl.raw[:span[0]] + new_str + pl.raw[span[1]:]

    logger.debug(f"修改后高度范围：[{min(y_new)}, {max(y_new)}]")

    return "".join(new_lines), T1o, T2o


# =========================
# GUI 日志 Handler
# =========================

class TextHandler(logging.Handler):
    """Write logs to a Tk ScrolledText with colored segments.

    - 默认只显示 INFO / ERROR（以及 CRITICAL）
    - 当 is_debug_mode() 为 True 时，额外显示 DEBUG / WARNING
    - 选择新文件时，可先写入黄色分隔符 "=========="（无时间/函数名前缀）
    - 函数名：蓝色；文件路径：红色；LG 列表：绿色
    """

    def __init__(self, text_widget: ScrolledText, is_debug_mode=None):
        super().__init__()
        self.text_widget = text_widget
        self.is_debug_mode = is_debug_mode
        self._init_tags()

    def _init_tags(self):
        try:
            tw = self.text_widget
            tw.tag_configure("SEP", foreground="#D4A000")        # 黄
            tw.tag_configure("FUNC", foreground="#1E5AA8")       # 蓝
            tw.tag_configure("PATH", foreground="#C00000")       # 红
            tw.tag_configure("LG", foreground="#0B7D0B")         # 绿

            tw.tag_configure("INFO", foreground="#000000")
            tw.tag_configure("DEBUG", foreground="#666666")
            tw.tag_configure("WARNING", foreground="#CC7A00")
            tw.tag_configure("ERROR", foreground="#C00000")
            tw.tag_configure("CRITICAL", foreground="#FFFFFF", background="#C00000")
        except Exception:
            pass

    def _level_tag(self, levelno: int) -> str:
        if levelno >= logging.CRITICAL:return "CRITICAL"
        if levelno >= logging.ERROR:return "ERROR"
        if levelno >= logging.WARNING:return "WARNING"
        if levelno >= logging.INFO:return "INFO"
        return "DEBUG"

    def _insert_colored_message_line(self, tw: ScrolledText, line: str, base_tag: str):
        # 文件路径红色
        m = re.match(r"^(选择文件:\s*)(.+)$", line)
        if m:
            tw.insert(tk.END, m.group(1), base_tag)
            tw.insert(tk.END, m.group(2), "PATH")
            return

        m = re.match(r"^(默认输出文件:\s*)(.+)$", line)
        if m:
            tw.insert(tk.END, m.group(1), base_tag)
            tw.insert(tk.END, m.group(2), "PATH")
            return

        m = re.match(r"^(输入文件:\s*)(.+)$", line)
        if m:
            tw.insert(tk.END, m.group(1), base_tag)
            tw.insert(tk.END, m.group(2), "PATH")
            return

        m = re.match(r"^(输出文件:\s*)(.+)$", line)
        if m:
            tw.insert(tk.END, m.group(1), base_tag)
            tw.insert(tk.END, m.group(2), "PATH")
            return

        m = re.match(r"^(选择输出路径:\s*)(.+)$", line)
        if m:
            tw.insert(tk.END, m.group(1), base_tag)
            tw.insert(tk.END, m.group(2), "PATH")
            return

        # LG 列表绿色（整段）
        lg_pos = line.find("LG=")
        if lg_pos != -1:
            tw.insert(tk.END, line[:lg_pos], base_tag)
            tw.insert(tk.END, line[lg_pos:], "LG")
            return

        tw.insert(tk.END, line, base_tag)

    def emit(self, record):
        # 是否启用 debug 输出
        debug_on = False
        try:
            debug_on = bool(self.is_debug_mode()) if callable(self.is_debug_mode) else False
        except Exception:
            debug_on = False

        if debug_on:
            allowed = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
        else:
            allowed = (logging.INFO, logging.ERROR, logging.CRITICAL)

        if record.levelno not in allowed:
            return

        # 分隔符：仅显示黄色 "=========="
        try:
            if isinstance(record.msg, str) and record.msg.strip() == "==========":
                def append_sep():
                    try:
                        self.text_widget.configure(state="normal")
                        self.text_widget.insert(tk.END, "==========\n", "SEP")
                        self.text_widget.see(tk.END)
                        self.text_widget.configure(state="disabled")
                    except Exception:
                        pass

                try:
                    self.text_widget.after(0, append_sep)
                except Exception:
                    append_sep()
                return
        except Exception:
            pass

        # 生成头部与正文（函数名单独上色）
        try:
            ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        except Exception:
            ts = ""

        base_tag = self._level_tag(record.levelno)
        header_prefix = f"[{ts}] {record.levelname} - "
        func_name = getattr(record, "funcName", "") or ""

        msg = record.getMessage()

        # 异常堆栈（logger.exception）
        if record.exc_info:
            try:
                if self.formatter:
                    exc_text = self.formatter.formatException(record.exc_info)
                else:
                    exc_text = logging.Formatter().formatException(record.exc_info)
            except Exception:
                exc_text = ""
            if exc_text:
                msg = msg + "\n" + exc_text

        def append():
            try:
                self.text_widget.configure(state="normal")

                # 头：prefix + funcName(蓝) + 换行
                self.text_widget.insert(tk.END, header_prefix, base_tag)
                self.text_widget.insert(tk.END, func_name, "FUNC")
                self.text_widget.insert(tk.END, "\n", base_tag)

                # 正文：逐行插入，匹配路径/LG
                for line in msg.splitlines() or [""]:
                    self._insert_colored_message_line(self.text_widget, line, base_tag)
                    self.text_widget.insert(tk.END, "\n", base_tag)

                self.text_widget.see(tk.END)
                self.text_widget.configure(state="disabled")
            except Exception:
                pass

        try:
            self.text_widget.after(0, append)
        except Exception:
            append()


@contextmanager
def _scoped_ui_debug(app, enabled: bool):
    """Temporarily enable verbose log display in UI for one action."""
    old = getattr(app, "_ui_debug_active", False)
    try:
        app._ui_debug_active = bool(enabled)
        yield
    finally:
        app._ui_debug_active = old


def compute_lg_list(mg_values: List[float], H2: float) -> str:
    """Compute LG list string like LG=[{0.45,xxx},{0.46,xxxx}] with LG rounded to 3 decimals."""
    if H2 is None or H2 == 0:
        raise ValueError("H2 is 0, cannot compute LG.")
    out_items = []
    for mg in mg_values:
        lg = 1000.0 * mg / (math.pi * 3.35 * 3.35 * H2 / 1000.0)
        out_items.append(f"{{{mg:.2f},{lg:.3f}}}")
    return "LG=[" + ",".join(out_items) + "]"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TMA数据分析处理器")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.t1_var = tk.StringVar()
        self.t2_var = tk.StringVar()
        self.mg_var = tk.StringVar(value="0.50")

        # UI debug 显示开关：仅对单次动作生效（执行/选择文件时临时置 True）
        self._ui_debug_active = False
        self.debug_var = tk.BooleanVar(value=False)

        # 绘图导出路径（绘图窗口复用）
        self.export_use_newpath_var = tk.BooleanVar(value=False)
        self.export_newpath_var = tk.StringVar(value=r"\\WIN-EK6CSI67PON\TMAresult\LS")

        self._build_ui()
        self.logger = self._setup_logger()

        self._cached_parsed: Optional[List[ParsedLine]] = None


        # 最近一次“执行修改”产生的文本/解析结果（用于绘图）
        self._last_output_text: Optional[str] = None
        self._last_output_parsed: Optional[List[ParsedLine]] = None
        self._last_run_mg: Optional[float] = None

        # 绘图窗口缓存
        self._plot_win: Optional[tk.Toplevel] = None
        self._plot_canvas: Optional[FigureCanvasTkAgg] = None
        self._plot_fig: Optional[Figure] = None
        self._plot_toolbar: Optional[ttk.Frame] = None
    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        # 第1行
        ttk.Button(frm, text="文件路径", command=self.on_choose_file).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", pady=4)

        # 第2行
        ttk.Button(frm, text="输出路径", command=self.on_choose_output).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", pady=4)

        # 第3行
        row3 = ttk.Frame(frm)
        row3.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(row3, text="T1=").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(row3, width=12, textvariable=self.t1_var).grid(row=0, column=1, padx=(0, 16))
        ttk.Label(row3, text="T2=").grid(row=0, column=2, padx=(0, 6))
        ttk.Entry(row3, width=12, textvariable=self.t2_var).grid(row=0, column=3, padx=(0, 16))
        ttk.Label(row3, text="Mg=").grid(row=0, column=4, padx=(0, 6))
        ttk.Entry(row3, width=8, textvariable=self.mg_var).grid(row=0, column=5)

        # 第4行
        row4 = ttk.Frame(frm)
        row4.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(row4, text="执行调整", command=self.on_execute).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(row4, text="使用绘图", command=self.on_plot).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(row4, text="退出软件", command=self.root.destroy).grid(row=0, column=2)

        # 第5行 log
        ttk.Label(frm, text="Soft Log").grid(row=4, column=0, sticky="w")
        tk.Checkbutton(frm, text="DeBug Mode", variable=self.debug_var, fg="orange").grid(row=4, column=1, sticky="w", padx=8)
        self.log_box = ScrolledText(frm, height=14)
        self.log_box.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.log_box.configure(state="disabled")
        frm.rowconfigure(5, weight=1)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("curve_warp")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False
    
        handler = TextHandler(self.log_box, is_debug_mode=lambda: self._ui_debug_active)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(funcName)s\n%(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def on_choose_file(self):
        debug_enabled = bool(self.debug_var.get())
        with _scoped_ui_debug(self, debug_enabled):
            self.logger.debug("进入 on_choose_file")
            path = filedialog.askopenfilename(
                title="选择输入文件",
                filetypes=[("全部文件", "*.*"), ("001 文件", "*.001"), ("TXT 文件", "*.txt"), ("001-999 文件", NUM_EXT_PATTERNS)]
            )
            if not path:
                return

            self.input_path.set(path)
            # 每次打开新文件，刷新 mg 默认值
            self.mg_var.set("0.50")

            # 切换文件后，清空上一次修改/绘图缓存
            self._last_output_text = None
            self._last_output_parsed = None
            self._last_run_mg = None

            base, ext = os.path.splitext(path)
            out_default = base + "_new" + (ext if ext else ".txt")
            self.output_path.set(out_default)

            self.logger.info("==========")

            self.logger.info(f"选择文件: {path}")
            self.logger.info(f"默认输出文件: {out_default}")

            try:
                text, meta = load_text_prefer_tajm(path, self.logger)
                self.logger.debug(f"读取来源: {meta.get('used')}" + (f" rows={meta.get('rows')}" if meta.get('used')=='tajm' else ""))
                parsed = parse_lines(text)
                self._cached_parsed = parsed

                # ✅ 只从 StartOfData 后提取真实数据段，避免 header 数字干扰 T2
                xs, ys, _line_idxs = extract_xy_from_file(parsed, self.logger)

                T1o, T2o = detect_T2_T1(xs, ys, self.logger)

                self.t1_var.set(f"{T1o:.6g}")
                self.t2_var.set(f"{T2o:.6g}")


                # INFO模式下，额外输出T2对应高度(H2)以及LG列表
                try:
                    debug_enabled = bool(self.debug_var.get()) if hasattr(self, 'debug_var') else False
                except Exception:
                    debug_enabled = False
                if not debug_enabled:
                    try:
                        # H2：T2对应的高度（最大高度）
                        H2 = float(max(ys)) if ys else float("nan")
                        self.logger.info(f"T2对应高度(H2)={format_height_um(H2)}")
                        mg_values = [round(0.45 + 0.01*i, 2) for i in range(11)]
                        self.logger.info(compute_lg_list(mg_values, H2))
                    except Exception as _e:
                        self.logger.error(f"LG计算失败：{_e}")
                    self.logger.info(f"自动识别并填入: T1={T1o}, T2={T2o}")

            except Exception as e:
                self.logger.exception("读取/识别 T1/T2 失败")
                messagebox.showerror("错误", f"读取/识别T1/T2失败：\n{e}")

    def on_choose_output(self):
        debug_enabled = bool(self.debug_var.get()) if hasattr(self, 'debug_var') else False
        with _scoped_ui_debug(self, debug_enabled):
            self.logger.debug("进入 on_choose_output")
            path = filedialog.asksaveasfilename(
                title="选择输出保存路径",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not path:
                return
            self.output_path.set(path)
            self.logger.info(f"选择输出路径: {path}")

    def on_execute(self):
        in_path = self.input_path.get().strip()
        out_path = self.output_path.get().strip()
    
        if not in_path:
            messagebox.showwarning("提示", "请先选择输入文件。")
            return
        if not out_path:
            messagebox.showwarning("提示", "请选择输出文件路径。")
            return
    
        # 仅对本次“执行修改”流程生效：执行前读取一次复选框状态
        debug_enabled = bool(self.debug_var.get())
        if debug_enabled:
            # 这条是 INFO（UI 可见）
            self.logger.info("⚠️ DEBUG信息模式已开启：显示详细日志")
        t0 = time.perf_counter()
        tracemalloc_started = False
        if debug_enabled:
            try:
                tracemalloc.start()
                tracemalloc_started = True
            except Exception:
                tracemalloc_started = False
    
        try:
            with _scoped_ui_debug(self, debug_enabled):
                try:
                    T1n = float(self.t1_var.get().strip())
                    T2n = float(self.t2_var.get().strip())
                    mg = float(self.mg_var.get().strip())
                except ValueError:
                    messagebox.showerror("错误", "T1/T2/mg 输入必须是数字。")
                    return
    
                # UI 里只显示关键 INFO
                self.logger.info(f"开始执行修改：T1={T1n}  T2={T2n}  mg={mg}")
    
                # 详细信息（仅写入 debug_log）
                self.logger.debug("========== 开始执行修改 ==========")
                self.logger.debug(f"输入文件: {in_path}")
                self.logger.debug(f"输出文件: {out_path}")
    
                if self._cached_parsed is None:
                    text, meta = load_text_prefer_tajm(in_path, self.logger)
                    self.logger.debug(
                        f"执行时读取来源: {meta.get('used')}"
                        + (f" rows={meta.get('rows')}" if meta.get('used') == 'tajm' else "")
                    )
                    parsed = parse_lines(text)
                    self._cached_parsed = parsed
                else:
                    parsed = self._cached_parsed
    
                new_text, T1o, T2o = apply_warp(parsed, T1n=T1n, T2n=T2n, mg=mg, logger=self.logger)
                write_utf16le_with_bom(out_path, new_text)

                # 缓存本次修改结果（供“开始绘图”直接使用）
                self._last_output_text = new_text
                try:self._last_output_parsed = parse_lines(new_text)
                except Exception:self._last_output_parsed = None
                self._last_run_mg = mg
    
                self.logger.info(f"✅ 修改完成：{out_path}")
        except Exception as e:
            self.logger.exception("执行修改失败")
            messagebox.showerror("错误", f"执行修改失败：\n{e}")
        finally:
            if debug_enabled:
                elapsed = time.perf_counter() - t0
                mem_info = ""
                if tracemalloc_started:
                    try:
                        cur, peak = tracemalloc.get_traced_memory()
                        mem_info = f" | 内存(Tracemalloc): current={cur/1024:.1f}KB, peak={peak/1024:.1f}KB"
                    except Exception:
                        pass
                    try:
                        tracemalloc.stop()
                    except Exception:
                        pass
                # DEBUG 模式下显示“执行耗时 / 内存占用”（INFO 输出，UI 可见）
                self.logger.info(f"⚠️ DEBUG指标: 执行耗时={elapsed:.3f}s{mem_info}")

    def on_plot(self):
        in_path = self.input_path.get().strip()
        if not in_path:
            messagebox.showwarning("提示", "请先选择输入文件。")
            return

        # 仅对本次“开始绘图”流程生效：执行前读取一次复选框状态
        debug_enabled = bool(self.debug_var.get())
        if debug_enabled:
            self.logger.info("⚠️ DEBUG信息模式已开启：显示详细日志")

        with _scoped_ui_debug(self, debug_enabled):
            try:
                # 选择数据源：优先使用最近一次“执行修改”的结果；否则使用原始输入文件
                if self._last_output_parsed is not None:
                    parsed = self._last_output_parsed
                    src_tag = "修改后"
                    mg_for_title = self._last_run_mg
                else:
                    if self._cached_parsed is None:
                        text, meta = load_text_prefer_tajm(in_path, self.logger)
                        self.logger.debug(
                            f"绘图时读取来源: {meta.get('used')}"
                            + (f" rows={meta.get('rows')}" if meta.get('used') == 'tajm' else "")
                        )
                        parsed = parse_lines(text)
                        self._cached_parsed = parsed
                    else:
                        parsed = self._cached_parsed
                    src_tag = "原始"
                    mg_for_title = None

                xs, ys, _line_idxs = extract_xy_from_file(parsed, self.logger)
                if not xs or not ys:
                    raise ValueError("未提取到绘图数据（请确认 StartOfData 后存在 5 列数据，且第2/3列为温度/高度）。")

                # 读取当前 UI 的 T1/T2（用于标注）
                try:
                    t1 = float(self.t1_var.get().strip())
                    t2 = float(self.t2_var.get().strip())
                except Exception:
                    t1, t2 = None, None

                # subtitle 质量：优先使用最近一次执行修改的 mg；否则用当前输入框 mg
                mg_val = None
                if mg_for_title is not None:
                    mg_val = mg_for_title
                else:
                    try:
                        mg_val = float(self.mg_var.get().strip())
                    except Exception:
                        mg_val = None

                # ---- 计算 y 轴范围（新规则）----
                y_min_data = float(min(ys))
                y_max_data = float(max(ys))

                # 基础下限：若最小值 > -200 μm，则下限固定为 -200；否则向下取整到 500 的整数倍
                if y_min_data > -200.0:
                    y0_base = -200.0
                else:
                    y0_base = math.floor(y_min_data / 500.0) * 500.0

                # 基础上限：向上取整到 500 的整数倍
                y1_base = math.ceil(y_max_data / 500.0) * 500.0
                if y1_base < 0:
                    y1_base = 0.0
                if y1_base <= y0_base:
                    y1_base = y0_base + 500.0

                # 实际绘图范围（会因标注超框而动态调整）
                y0_plot = float(y0_base)
                y1_plot = float(y1_base)

                # 找到 T1/T2 对应的点（用于判断标注是否会超框）
                idx1 = int(min(range(len(xs)), key=lambda i: abs(xs[i] - t1)))
                idx2 = int(min(range(len(xs)), key=lambda i: abs(xs[i] - t2)))
                x1_pt, y1_pt = float(xs[idx1]), float(ys[idx1])
                x2_pt, y2_pt = float(xs[idx2]), float(ys[idx2])

                def _ann_margin(cur_y0: float, cur_y1: float) -> float:
                    # 用于估计“文字标注”在 y 方向可能占用的空间（经验值）
                    yr = max(1.0, cur_y1 - cur_y0)
                    return max(80.0, yr * 0.04)

                ann_margin = _ann_margin(y0_plot, y1_plot)

                # 若 T2 温度标注会超出框，则上限抬高 200（标注刻度仍以 y1_base 为准）
                if y2_pt + ann_margin > y1_plot:
                    y1_plot += 200.0
                    ann_margin = _ann_margin(y0_plot, y1_plot)

                # 若 T1 温度标注（点下方）会超出框：y 轴下限每次额外降低 100，直到不超出
                while y1_pt - ann_margin < y0_plot:
                    y0_plot -= 100.0
                    ann_margin = _ann_margin(y0_plot, y1_plot)

                if y1_plot <= y0_plot:
                    y1_plot = y0_plot + 500.0

                # y 轴刻度：
                # - 若下限 > -500：标注“当前下限、0、500、1000...”
                # - 若下限 <= -500：在上述基础上额外增加 -500
                y_ticks_major = [float(y0_plot), 0.0]
                if y0_plot < -500.0 - 1e-9:
                    y_ticks_major.append(-500.0)
                y_ticks_major.extend([float(v) for v in range(500, int(y1_base) + 1, 500)])
                # 过滤并去重排序
                y_ticks_major = sorted({v for v in y_ticks_major if y0_plot - 1e-9 <= v <= y1_base + 1e-9})

                # 小刻度：每 100 一个（覆盖绘图范围，便于虚线背景/小刻度显示）
                y_minor_start = int(math.floor(y0_plot / 100.0) * 100)
                y_minor_end = int(math.ceil(y1_plot / 100.0) * 100)
                y_ticks_minor = list(range(y_minor_start, y_minor_end + 1, 100))
# ---- 创建/复用绘图窗口 ----
                if self._plot_win is not None and self._plot_win.winfo_exists():
                    win = self._plot_win
                    # 清理旧 canvas
                    if self._plot_canvas is not None:
                        try:self._plot_canvas.get_tk_widget().destroy()
                        except Exception:pass
                        self._plot_canvas = None
                else:
                    win = tk.Toplevel(self.root)
                    win.title("TMA - 绘图")
                    win.geometry("1000x650")
                    self._plot_win = win

                # 工具栏（导出 TIFF）
                if self._plot_toolbar is not None:
                    try:
                        self._plot_toolbar.destroy()
                    except Exception:
                        pass
                    self._plot_toolbar = None
                toolbar = ttk.Frame(win)
                toolbar.pack(fill='x', padx=6, pady=6)
                self._plot_toolbar = toolbar

                def _export_tiff():
                    try:
                        if self._plot_fig is None:
                            raise RuntimeError('未找到可导出的图像。')
                        # 输出路径：原文件名.tif，路径不变
                        base = os.path.splitext(os.path.basename(in_path))[0]
                        out_dir = os.path.dirname(in_path)

                        try:

                            if getattr(self, 'export_use_newpath_var', None) is not None and bool(self.export_use_newpath_var.get()):

                                cand = (self.export_newpath_var.get() or '').strip()

                                if cand and os.path.isdir(cand):

                                    out_dir = cand

                        except Exception:

                            pass

                        tif_path = os.path.join(out_dir, base + '.tif')
                        fig = self._plot_fig
                        # 以 16:9 输出 3840x2160，ppi=300（6.4x3.6 inch @ 300dpi）
                        from io import BytesIO
                        from PIL import Image, TiffImagePlugin
                        old_size = tuple(fig.get_size_inches())

                        try:
                            fig.set_size_inches(12.8, 7.2, forward=True)
                            buf = BytesIO()
                            fig.savefig(buf, format='tiff', dpi = 300)
                            buf.seek(0)
                        finally:
                            # 还原画布尺寸，避免影响窗口显示
                            try:
                                fig.set_size_inches(old_size[0], old_size[1], forward=True)
                                if self._plot_canvas is not None:
                                    try:
                                        self._plot_canvas.draw_idle()
                                    except Exception:pass
                            except Exception:pass
                        img = Image.open(buf)
                        # TIFF 元信息：作者/拍摄日期/ppi
                        tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
                        tiffinfo[315] = "TMAmax"  # Artist/作者
                        # 本地时间（YYYY:MM:DD HH:MM:SS）
                        dt_str = datetime.datetime.now().astimezone().strftime("%Y:%m:%d %H:%M:%S")
                        tiffinfo[306] = dt_str  # DateTime
                        # 300 ppi
                        tiffinfo[282] = (300, 1)  # XResolution
                        tiffinfo[283] = (300, 1)  # YResolution
                        tiffinfo[296] = 2         # ResolutionUnit: inch
                        # 无损压缩保存（体积更小）
                        img.save(
                            tif_path,
                            compression='tiff_adobe_deflate',
                            tiffinfo=tiffinfo,
                            dpi=(300, 300),)
                        self.logger.info(f'已生成TIFF图: {tif_path}')
                        messagebox.showinfo('完成', f'已生成TIFF图：\n{tif_path}')
                    except Exception as ee:
                        self.logger.exception('生成TIFF失败')
                        messagebox.showerror('错误', f'生成TIFF失败：\n{ee}')
                def _export_tiff_universal():
                    try:
                        if self._plot_fig is None:
                            raise RuntimeError('未找到可导出的图像。')
                        # 输出路径：Universal - 原文件名.tif，路径不变
                        base = os.path.splitext(os.path.basename(in_path))[0]
                        out_dir = os.path.dirname(in_path)

                        try:

                            if getattr(self, 'export_use_newpath_var', None) is not None and bool(self.export_use_newpath_var.get()):

                                cand = (self.export_newpath_var.get() or '').strip()

                                if cand and os.path.isdir(cand):

                                    out_dir = cand

                        except Exception:

                            pass

                        tif_path = os.path.join(out_dir, 'Universal - ' + base + '.tif')
                        fig = self._plot_fig
                        from io import BytesIO
                        from PIL import Image, TiffImagePlugin
                        old_size = tuple(fig.get_size_inches())
                        try:
                            fig.set_size_inches(12.8, 7.2, forward=True)
                            buf = BytesIO()
                            fig.savefig(buf, format='tiff', dpi=300)
                            buf.seek(0)
                        finally:
                            try:
                                fig.set_size_inches(old_size[0], old_size[1], forward=True)
                                if self._plot_canvas is not None:
                                    try:
                                        self._plot_canvas.draw_idle()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        img = Image.open(buf)
                        # TIFF 元信息：作者/拍摄日期/ppi
                        tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
                        tiffinfo[315] = "TMAmax"  # Artist/作者
                        dt_str = datetime.datetime.now().astimezone().strftime("%Y:%m:%d %H:%M:%S")
                        tiffinfo[306] = dt_str  # DateTime
                        tiffinfo[282] = (300, 1)  # XResolution
                        tiffinfo[283] = (300, 1)  # YResolution
                        tiffinfo[296] = 2         # ResolutionUnit: inch
                        img.save(
                            tif_path,
                            compression='tiff_adobe_deflate',
                            tiffinfo=tiffinfo,
                            dpi=(300, 300),
                        )
                        self.logger.info(f'已生成仿生TIFF图: {tif_path}')
                        messagebox.showinfo('完成', f'已生成仿生TIFF图：\n{tif_path}')
                    except Exception as ee:
                        self.logger.exception('生成仿生TIFF失败')
                        messagebox.showerror('错误', f'生成仿生TIFF失败：\n{ee}')

                ttk.Button(toolbar, text='生成TIFF图(.tif)', command=_export_tiff).pack(side='left')
                ttk.Button(toolbar, text='仿生输出', command=_export_tiff_universal).pack(side='left', padx=(8, 0))

                # 第二行：出图路径设置
                pathbar = ttk.Frame(win)
                pathbar.pack(fill='x', padx=6, pady=(0, 6))
                ttk.Checkbutton(pathbar, text='启用新路径', variable=self.export_use_newpath_var).pack(side='left')
                ttk.Entry(pathbar, textvariable=self.export_newpath_var, width=60).pack(side='left', padx=8, fill='x', expand=True)

                def _choose_export_dir():
                    try:
                        d = filedialog.askdirectory(title='选择出图文件夹')
                    except Exception:
                        d = ''
                    if d:
                        # 清空再写入（视觉上等价于 set）
                        self.export_newpath_var.set('')
                        self.export_newpath_var.set(d)
                        self.logger.info(f'出图路径已调整为: {d}')

                ttk.Button(pathbar, text='调整出图路径', command=_choose_export_dir).pack(side='left', padx=(8, 0))

                # ---- 绘图 ----
                fig = Figure(figsize=(12.8, 7.2), dpi=100)
                ax = fig.add_subplot(111)
                self._plot_fig = fig

                # 曲线（线宽适中）- matplotlib linewidth 单位为 points
                ax.plot(xs, ys, color="black", linewidth=1.5)

                # x 轴范围 0~300（带标尺：大刻度50，小刻度10）
                x_left = 0.0
                x_right = 300.0
                ax.set_xlim(x_left, x_right)
                ax.xaxis.set_major_locator(MultipleLocator(50))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel("温度（℃）", fontsize=12)
                ax.tick_params(axis='x', which='major', length=6)
                ax.tick_params(axis='x', which='minor', length=3)

                # y 轴：500 的倍数（大刻度500，小刻度100）
                ax.set_ylim(y0_plot, y1_plot)

                # 背景网格：x 每隔 10、y 每隔 100 画细虚线（作为背景）
                for gx in range(0, 301, 10):
                    ax.axvline(gx, linestyle="--", linewidth=0.4, color="#999999", alpha=0.35, zorder=0)
                y_grid_start = int(math.floor(y0_plot / 100.0)) * 100
                y_grid_end = int(math.ceil(y1_plot / 100.0)) * 100
                for gy in range(y_grid_start, y_grid_end + 1, 100):
                    ax.axhline(gy, linestyle="--", linewidth=0.4, color="#999999", alpha=0.35, zorder=0)
                ax.yaxis.set_major_locator(FixedLocator(y_ticks_major))

                ax.yaxis.set_minor_locator(FixedLocator(y_ticks_minor))

                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_ylabel("尺寸变化(μm)", fontsize=12)
                ax.tick_params(axis='y', which='major', length=6)
                ax.tick_params(axis='y', which='minor', length=3)


                # 背景：500×500 正方形（(0,0)~(500,500)），最细虚线

                try:

                    ax.plot([0, 500, 500, 0, 0], [0, 0, 500, 500, 0], linestyle=(0, (1, 3)), linewidth=0.5, color='0.8', zorder=0)

                except Exception:

                    pass


                # 标题/副标题
                fig.suptitle("TMA", fontsize=16, fontweight="bold")
                mg_text = f"{mg_val:g}" if mg_val is not None else "x"
                method_note = f"方法日志：30°C平衡，施加 0.06 N；\n                 20°C/min加热至 250°C；\n称取质量：{mg_text}mg。"
                # 副标题隐藏，方法日志作为绘图备注显示

                # 标注信息（INFO）
                self.logger.info("开始绘图")
                self.logger.info(f"{src_tag}数据: 点数={len(xs)}")

                # T1/T2 标注：用“黑色十字短线”与曲线相交表示点
                y_range = (y1_plot - y0_plot)
                tick_half_y = max(10.0, y_range * 0.02)   # 竖向半长（数据坐标）
                tick_half_x = 2.0                         # 横向半长（℃）
                tick_lw = 1.2

                def _draw_cross(xc: float, yc: float):
                    x_left = max(0.0, xc - tick_half_x)
                    x_right = min(300.0, xc + tick_half_x)
                    ax.vlines(xc, yc - tick_half_y, yc + tick_half_y, colors="black", linewidth=tick_lw)
                    ax.hlines(yc, x_left, x_right, colors="black", linewidth=tick_lw)

                if x1_pt is not None and y1_pt is not None:
                    _draw_cross(x1_pt, y1_pt)
                    ax.annotate(
                        f"{x1_pt:.2f}℃", (x1_pt, y1_pt),
                        xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=12, color="black"
                    )

                if x2_pt is not None and y2_pt is not None:
                    _draw_cross(x2_pt, y2_pt)
                    ax.annotate(
                        f"{x2_pt:.2f}℃", (x2_pt, y2_pt),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color="black"
                    )

                # 高度差：T1 点向右延出平行线（到 T2 的 x 位置），箭头指向 T1 延长线与 T2 点
                if (x1_pt is not None and y1_pt is not None and x2_pt is not None and y2_pt is not None):
                    # 仅延出 T1 的水平线：从 T1 到 T2 的 x
                    ax.plot([x1_pt, x2_pt], [y1_pt, y1_pt], color="black", lw=0.8)

                    # 双向箭头：在 T2 的 x 位置，从 T1 水平线(y1) 到 T2 点(y2)
                    ax.annotate(
                        "",
                        xy=(x2_pt, y2_pt),
                        xytext=(x2_pt, y1_pt),
                        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0, shrinkA=0, shrinkB=0),
                    )
                    # 只标注“多少 μm”（避免与曲线重叠：取 T2 右侧一点，并把文字的左下角放在曲线之上）
                    dh = abs(y2_pt - y1_pt)

                    def _interp_y(xq: float) -> float:
                        pts = sorted(zip(xs, ys), key=lambda t: t[0])
                        xs_s = [p[0] for p in pts]
                        ys_s = [p[1] for p in pts]
                        if xq <= xs_s[0]:return float(ys_s[0])
                        if xq >= xs_s[-1]:return float(ys_s[-1])
                        # 线性插值
                        import bisect
                        j = bisect.bisect_left(xs_s, xq)
                        xL, yL = xs_s[j - 1], ys_s[j - 1]
                        xR, yR = xs_s[j], ys_s[j]
                        if xR == xL:return float(yL)
                        t = (xq - xL) / (xR - xL)
                        return float(yL + (yR - yL) * t)

                    x_text = min(300.0, x2_pt + 15.0)
                    y_curve = _interp_y(x_text)
                    margin = max(15.0, (y1_plot - y0_plot) * 0.03)
                    y_text = y_curve + margin
                    y_text = min(y_text, y1_plot - margin * 0.2) # 防止顶到上边界

                    ax.text(
                        x_text, y_text,
                        f"{format_height_um(dh)} μm",
                        fontsize=12, ha="left", va="bottom", color="black"
                    )
                canvas = FigureCanvasTkAgg(fig, master=win)
                # 绘图完成后添加“方法日志”备注：放置在左上角，且尽量避免与曲线重叠
                x_left_final = x_left
                method_artist = None
                for _ in range(20):
                    ax.set_xlim(x_left_final, x_right)
                    if method_artist is not None:
                        try:
                            method_artist.remove()
                        except Exception:
                            pass
                    method_artist = ax.text(
                        x_left_final + 10,
                        y1_plot - 100,
                        method_note,
                        ha='left',
                        va='top',
                        fontsize=12,
                    )
                    canvas.draw()
                    overlap = False
                    try:
                        renderer = canvas.get_renderer()
                        bbox = method_artist.get_window_extent(renderer=renderer)
                        pts = ax.transData.transform(list(zip(xs, ys)))
                        overlap = any((bbox.x0 <= px <= bbox.x1 and bbox.y0 <= py <= bbox.y1) for px, py in pts)
                    except Exception:
                        overlap = False
                    if not overlap:
                        break
                    x_left_final -= 20.0

                # 若向左扩展了 x 轴，补齐负区间的背景虚线（每 10°C 一条）
                if x_left_final < 0:
                    gx_start = int(math.floor(x_left_final / 10.0)) * 10
                    for gx in range(gx_start, 0, 10):
                        ax.axvline(gx, linestyle=(0, (2, 2)), linewidth=0.5, color='0.8', zorder=0)

                # 最终再 draw 一次，确保布局刷新
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self._plot_canvas = canvas
            except Exception as e:
                self.logger.exception("绘图失败")
                messagebox.showerror("错误", f"绘图失败：\n{e}")

def main():
    root = tk.Tk()
    root.geometry("820x520")
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
