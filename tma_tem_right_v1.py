import os
import re
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


# =========================
# 可调参数
# =========================
START_TEMP_FOR_T1 = 60.0   # 你说“数据开始(60℃附近)”，默认从>=60开始找T1
MONOTONIC_FIX_T1_T2 = True  # 修复：强制在[T1,T2]区间高度单调不下降，避免出现“裂开点”
MONOTONIC_METHOD = "bridge"  # bridge(线性桥接掉点) / isotonic(平滑) / cummax(硬钳制)
# 如果文件没有 StartOfData 标记（极少数情况），用温度范围过滤掉头部乱入的数字。
FALLBACK_TEMP_MIN = 40.0
FALLBACK_TEMP_MAX = 400.0
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

    规则：
    - 若存在 'StartOfData' 行：只取其后的数据行。
    - 若不存在：回退到温度范围过滤（x 在 [FALLBACK_TEMP_MIN, FALLBACK_TEMP_MAX]），以避免诸如 2026、2601057 这类头部数字。
    - 无论一行有 3/4/5 个数字，都取第2个数字=温度、第3个数字=高度。
    """
    xs, ys, line_idxs = [], [], []

    marker_idx = None
    for i, pl in enumerate(parsed):
        if 'StartOfData' in pl.raw:
            marker_idx = i
            break

    if marker_idx is not None:
        # ✅ 只取 StartOfData 之后的真实数据段，避免 header 的数字干扰 T2
        for idx in range(marker_idx + 1, len(parsed)):
            pl = parsed[idx]
            if len(pl.nums) >= 3 and math.isfinite(pl.nums[1]) and math.isfinite(pl.nums[2]):
                xs.append(pl.nums[1])
                ys.append(pl.nums[2])
                line_idxs.append(idx)
        logger.debug(f"检测到 StartOfData：数据起始行={marker_idx+1}，数据行数={len(xs)}")
        return xs, ys, line_idxs

    # ❗极少数文件没有 StartOfData：回退过滤（按温度范围）
    for idx, pl in enumerate(parsed):
        if len(pl.nums) >= 3 and math.isfinite(pl.nums[1]) and math.isfinite(pl.nums[2]):
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
    if not xs:
        raise ValueError("文件中没有可用数据点（无法解析温度/高度）。")

    # T2：高度最大值对应温度
    max_i = max(range(len(ys)), key=lambda i: ys[i])
    T2 = xs[max_i]

    # T1：从 x>=START_TEMP_FOR_T1 到 x<=T2 范围内的高度最小值对应温度
    candidates = [i for i in range(len(xs)) if xs[i] >= START_TEMP_FOR_T1 and xs[i] <= T2]
    if not candidates:
        # 如果没有>=60的点（比如数据从50开始且很短），退化为从最小温度到T2
        candidates = [i for i in range(len(xs)) if xs[i] <= T2]
    min_i = min(candidates, key=lambda i: ys[i])
    T1 = xs[min_i]

    logger.debug(f"T2索引={max_i}, T2温度={T2}, T2高度={ys[max_i]}")
    logger.debug(f"T1索引={min_i}, T1温度={T1}, T1高度={ys[min_i]} (搜索区间: x>= {START_TEMP_FOR_T1} 且 x<=T2)")
    return T1, T2


def piecewise_map(x: float, xmin: float, xmax: float, T1o: float, T2o: float, T1n: float, T2n: float) -> float:
    # 分三段线性映射： [xmin,T1o]->[xmin,T1n], [T1o,T2o]->[T1n,T2n], [T2o,xmax]->[T2n,xmax]
    if x <= T1o:
        if abs(T1o - xmin) < 1e-12:
            return x + (T1n - T1o)
        return xmin + (x - xmin) * (T1n - xmin) / (T1o - xmin)
    elif x <= T2o:
        if abs(T2o - T1o) < 1e-12:
            return T1n
        return T1n + (x - T1o) * (T2n - T1n) / (T2o - T1o)
    else:
        if abs(xmax - T2o) < 1e-12:
            return x + (T2n - T2o)
        return T2n + (x - T2o) * (xmax - T2n) / (xmax - T2o)


def linear_interp(x: float, xp: List[float], fp: List[float]) -> float:
    # xp 递增；越界夹到边界
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]

    lo, hi = 0, len(xp) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid

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
        if y[j] > m:
            m = y[j]
        else:
            y[j] = m

    return y


def enforce_monotonic_between(xs: List[float], ys: List[float], x1: float, x2: float, logger: logging.Logger):
    """
    修复“裂开点”：在温度区间 [x1, x2] 内强制高度 ys 随温度 xs 单调不下降。
    - xs/ys 为同长度列表（按文件行顺序）。
    - 只对区间内的点做修正；区间外保持不变。
    """
    if x1 >= x2:
        return ys

    idxs = [i for i, x in enumerate(xs) if x1 <= x <= x2]
    if len(idxs) < 2:
        return ys

    # 按温度升序做单调修正（避免温度不严格递增导致误判）
    idxs_sorted = sorted(idxs, key=lambda i: xs[i])
    seg = [ys[i] for i in idxs_sorted]


    tol = max(1e-9, (max(seg) - min(seg)) * 1e-6)
    method = (MONOTONIC_METHOD or "isotonic").lower()
    if method == "cummax":
        fixed = []
        m = -float('inf')
        for v in seg:
            if v > m:
                m = v
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

    if changed:
        logger.debug(f"单调修复已应用：区间[{x1}, {x2}] 内修改点数={changed}, 最大改变量={max_abs_delta}")
    else:
        logger.debug(f"单调修复检查：区间[{x1}, {x2}] 内无需修改")

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


def apply_warp(parsed: List[ParsedLine], T1n: float, T2n: float, logger: logging.Logger) -> Tuple[str, float, float]:
    """
    ✅ 规则核验：
    - 一行有 5 个数字：取第2/第3个数字（index 1/2）
    - 一行有 4 个数字：取第2/第3个数字（index 1/2）
    - 一行有 3 个数字：取第2/第3个数字（index 1/2）
    - 少于 3 个数字：不处理该行
    """
    # ✅ 只从 StartOfData 后提取真实数据段，避免 header 数字干扰（这是 T2 识别错误的根因）
    xs, ys, data_line_indices = extract_xy_from_file(parsed, logger)

    if not xs:
        raise ValueError("没有找到有效数据行：每行至少要有3个数字，才能取第2列温度和第3列高度。")

    xmin, xmax = min(xs), max(xs)
    T1o, T2o = detect_T2_T1(xs, ys, logger)

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
    if MONOTONIC_FIX_T1_T2:
        y_new = enforce_monotonic_between(xs, y_new, T1n, T2n, logger)

    # 3) ✅ 写回：只替换“第3个数字token”（高度）
    new_lines = [pl.raw for pl in parsed]
    for local_i, line_idx in enumerate(data_line_indices):
        pl = parsed[line_idx]
        span = pl.spans[2]  # ✅ 第3个数字的原始位置
        old_str = pl.raw[span[0]:span[1]]
        new_str = format_like(old_str, y_new[local_i])
        new_lines[line_idx] = pl.raw[:span[0]] + new_str + pl.raw[span[1]:]

    return "".join(new_lines), T1o, T2o


# =========================
# GUI 日志 Handler
# =========================
class TextHandler(logging.Handler):
    def __init__(self, text_widget: ScrolledText):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")

        self.text_widget.after(0, append)


# =========================
# GUI 主程序
# =========================
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("T1/T2 曲线拉伸修改工具")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.t1_var = tk.StringVar()
        self.t2_var = tk.StringVar()

        self._build_ui()
        self.logger = self._setup_logger()

        self._cached_parsed: Optional[List[ParsedLine]] = None

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        # 第1行
        ttk.Button(frm, text="选择文件", command=self.on_choose_file).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", pady=4)

        # 第2行
        ttk.Button(frm, text="输出文件", command=self.on_choose_output).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", pady=4)

        # 第3行
        row3 = ttk.Frame(frm)
        row3.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(row3, text="T1修改").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(row3, width=12, textvariable=self.t1_var).grid(row=0, column=1, padx=(0, 16))
        ttk.Label(row3, text="T2修改").grid(row=0, column=2, padx=(0, 6))
        ttk.Entry(row3, width=12, textvariable=self.t2_var).grid(row=0, column=3)

        # 第4行
        row4 = ttk.Frame(frm)
        row4.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(row4, text="执行修改", command=self.on_execute).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(row4, text="退出软件", command=self.root.destroy).grid(row=0, column=1)

        # 第5行 log
        ttk.Label(frm, text="log日志（DEBUG）").grid(row=4, column=0, columnspan=2, sticky="w")
        self.log_box = ScrolledText(frm, height=14)
        self.log_box.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.log_box.configure(state="disabled")
        frm.rowconfigure(5, weight=1)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("curve_warp")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        handler = TextHandler(self.log_box)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def on_choose_file(self):
        path = filedialog.askopenfilename(
            title="选择输入txt文件",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.input_path.set(path)

        base, ext = os.path.splitext(path)
        out_default = base + "_new" + (ext if ext else ".txt")
        self.output_path.set(out_default)

        self.logger.debug(f"选择文件: {path}")
        self.logger.debug(f"默认输出文件: {out_default}")

        try:
            text = read_utf16le_no_bom(path)
            parsed = parse_lines(text)
            self._cached_parsed = parsed

            # ✅ 只从 StartOfData 后提取真实数据段，避免 header 数字干扰 T2
            xs, ys, _line_idxs = extract_xy_from_file(parsed, self.logger)

            T1o, T2o = detect_T2_T1(xs, ys, self.logger)

            self.t1_var.set(f"{T1o:.6g}")
            self.t2_var.set(f"{T2o:.6g}")

            self.logger.debug(f"自动识别并填入: T1={T1o}, T2={T2o}")

        except Exception as e:
            self.logger.exception("读取/识别 T1/T2 失败")
            messagebox.showerror("错误", f"读取/识别T1/T2失败：\n{e}")

    def on_choose_output(self):
        path = filedialog.asksaveasfilename(
            title="选择输出保存路径",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not path:
            return
        self.output_path.set(path)
        self.logger.debug(f"选择输出路径: {path}")

    def on_execute(self):
        in_path = self.input_path.get().strip()
        out_path = self.output_path.get().strip()

        if not in_path:
            messagebox.showwarning("提示", "请先选择输入文件。")
            return
        if not out_path:
            messagebox.showwarning("提示", "请选择输出文件路径。")
            return

        try:
            T1n = float(self.t1_var.get().strip())
            T2n = float(self.t2_var.get().strip())
        except ValueError:
            messagebox.showerror("错误", "T1/T2 输入必须是数字。")
            return

        self.logger.debug("========== 开始执行修改 ==========")
        self.logger.debug(f"输入文件: {in_path}")
        self.logger.debug(f"输出文件: {out_path}")

        try:
            if self._cached_parsed is None:
                text = read_utf16le_no_bom(in_path)
                parsed = parse_lines(text)
            else:
                parsed = self._cached_parsed

            new_text, T1o, T2o = apply_warp(parsed, T1n, T2n, self.logger)
            write_utf16le_with_bom(out_path, new_text)
            self.logger.debug(f"完成：原始T1={T1o}, 原始T2={T2o} -> 新T1={T1n}, 新T2={T2n}")
            self.logger.debug("保存完成。")
            self.logger.debug("========== 修改结束 ==========")
            messagebox.showinfo("完成", f"修改完成，已保存：\n{out_path}")

        except Exception as e:
            self.logger.exception("执行修改失败")
            messagebox.showerror("错误", f"执行修改失败：\n{e}")


def main():
    root = tk.Tk()
    root.geometry("820x520")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
