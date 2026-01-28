import os
import re
import math
import struct
import traceback
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import filedialog


# =========================
# Core parsing / exporting
# pyinstaller -F -w -i tajm.ico TAJM.py
# =========================

def is_valid_ext_001_099(path: str) -> bool:
    _, ext = os.path.splitext(path)
    if not re.fullmatch(r"\.\d{3}", ext or ""):
        return False
    n = int(ext[1:])
    return 1 <= n <= 99


def find_header_end(buf: bytes, log=None) -> int:
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
        if p == -1:
            break
        hits += 1
        candidate = p + len(crlf)

        window = buf[candidate:candidate + 512]
        if len(window) < 64:
            break

        odd = window[1::2]  # UTF-16LE 高字节
        zero_ratio = odd.count(0) / max(1, len(odd))

        if log:
            log(f"[header-scan] CRLF@{p} -> candidate={candidate}, odd_zero_ratio={zero_ratio:.3f}")

        if zero_ratio >= 0.55:
            last_good = candidate
            pos = candidate
        else:
            break

    if log:
        log(f"[header-scan] scanned_bytes=0..{scan_limit}, crlf_hits={hits}, header_end={last_good}")

    if last_good is None:
        raise RuntimeError("未能定位 UTF-16LE 头部结束位置（找不到可靠的 CRLF 终点）。")

    return last_good


def find_data_start_by_first_time(buf: bytes, search_from: int, log=None) -> int:
    """
    数据区通常为 5 个 little-endian float32 组成的记录（20 字节/行）。
    优先寻找 time=0.004 的 float32；找不到则用启发式扫描。
    """
    needle = struct.pack("<f", 0.004)
    p = buf.find(needle, search_from)

    if log:
        log(f"[data-find] try needle float32(0.004) from offset {search_from} -> pos={p}")

    if p != -1:
        return p

    scan_to = min(len(buf) - 40, search_from + 4096)
    if log:
        log(f"[data-find] fallback scan range [{search_from}, {scan_to}]")

    for start in range(search_from, scan_to):
        try:
            t, temp, dim, force, flow = struct.unpack_from("<5f", buf, start)
        except struct.error:
            break

        if not all(map(math.isfinite, [t, temp, dim, force, flow])):
            continue

        if 0 <= t <= 10 and -300 <= temp <= 500:
            try:
                t2 = struct.unpack_from("<f", buf, start + 20)[0]
            except struct.error:
                continue

            if math.isfinite(t2) and t2 >= t:
                if log:
                    log(f"[data-find] fallback hit at offset {start} -> first_row=(t={t}, temp={temp}, ...), next_t={t2}")
                return start

    raise RuntimeError("未能定位数据区起点（float32 表）。")


def parse_rows(buf: bytes, start: int, log=None, max_rows: int = 200000):
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
                log(f"[parse] stop(non-finite) at pos={pos}, row_index={len(rows)}")
            break

        if not (0 <= t <= 1e6 and -500 <= temp <= 2000 and -1e7 <= dim <= 1e7 and
                -1e6 <= force <= 1e6 and -1e6 <= flow <= 1e6):
            stops += 1
            if log:
                log(f"[parse] stop(out-of-range) at pos={pos}, row_index={len(rows)} "
                    f"values=(t={t}, temp={temp}, dim={dim}, force={force}, flow={flow})")
            break

        if t < prev_t - 1e-3:
            stops += 1
            if log:
                log(f"[parse] stop(time-not-monotonic) at pos={pos}, row_index={len(rows)} t={t} prev_t={prev_t}")
            break

        prev_t = t
        rows.append((t, temp, dim, force, flow))
        pos += 20

        if log and len(rows) in (1, 2, 3, 10, 100, 500, 1000, 2000, 5000, 10000):
            log(f"[parse] progress rows={len(rows)} latest=(t={t}, temp={temp}, dim={dim}, force={force}, flow={flow}) pos={pos}")

    if log:
        log(f"[parse] done rows={len(rows)}, start={start}, end_pos={pos}, stops={stops}")

    return rows


def write_utf16le_with_bom(path: str, text: str, log=None):
    if text.startswith("\ufeff"):
        text = text[1:]
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")  # BOM
        f.write(text.encode("utf-16le"))
    if log:
        log(f"[write] UTF-16LE+BOM wrote: {path} (chars={len(text)})")


def compute_output_path(in_path: str, suffix: str, name_append: str, log=None) -> str:
    """
    suffix 输入框：空 -> .txt；否则输出扩展名变为 suffix（带点或不带点都行）
    name_append 输入框：空 -> 不加；否则在“输出扩展名之前”追加该字符串
      例如： 低温-260128.001 + suffix=txt + append=_A
        -> 低温-260128_A.txt
    """
    base, _ = os.path.splitext(in_path)

    # append
    app = name_append.strip()
    if app:
        base_out = base + app
    else:
        base_out = base

    # suffix/ext
    if not suffix.strip():
        out_ext = ".txt"
    else:
        s = suffix.strip()
        out_ext = s if s.startswith(".") else "." + s

    out_path = base_out + out_ext
    if log:
        log(f"[path] in={in_path} suffix='{suffix}' append='{name_append}' -> out={out_path}")
    return out_path


def decrypt_one_file(in_path: str, suffix: str, name_append: str, log=None) -> dict:
    """
    处理单个文件：读取 -> 找头部 -> 找数据 -> 解 -> 输出。
    输出结构：UTF-16LE 头部文本 + StartOfData + 纯数字数据（5列）。
    """
    t0 = datetime.now()
    result = {"input": in_path, "ok": False}

    if log:
        log(f"==========\n[task] start: {in_path}")

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"找不到文件：{in_path}")

    if not is_valid_ext_001_099(in_path):
        raise ValueError(f"扩展名不在 .001~.099：{in_path}")

    size = os.path.getsize(in_path)
    if log:
        log(f"[io] file_size={size} bytes")

    with open(in_path, "rb") as f:
        buf = f.read()

    if len(buf) < 2:
        raise RuntimeError("文件太小，无法解析。")

    if log:
        log(f"[io] leading_bytes={buf[:2].hex(' ')} (expect FF FE for UTF-16LE header)")

    # 头部
    header_end = find_header_end(buf, log=log)
    header_text = buf[:header_end].decode("utf-16le", errors="ignore")

    if log:
        preview = header_text[:400].replace("\r", "\\r").replace("\n", "\\n")
        log(f"[header] header_end={header_end}, header_chars={len(header_text)} preview='{preview}...'")

    # 数据区
    data_start = find_data_start_by_first_time(buf, header_end, log=log)
    rows = parse_rows(buf, data_start, log=log)

    if not rows:
        raise RuntimeError("未解析到任何数据记录。")

    # 输出路径
    out_path = compute_output_path(in_path, suffix, name_append, log=log)

    # 输出内容：头部 + StartOfData + 纯数字
    if not header_text.endswith("\r\n"):
        header_text += "\r\n"

    out_lines = [header_text.rstrip("\r\n"), "StartOfData"]
    for t, temp, dim, force, flow in rows:
        out_lines.append(f"{t:.6f}\t{temp:.5f}\t{dim:.6f}\t{force:.8f}\t{flow:.5f}")

    out_text = "\r\n".join(out_lines) + "\r\n"
    write_utf16le_with_bom(out_path, out_text, log=log)

    dt = (datetime.now() - t0).total_seconds()
    if log:
        log(f"[task] done: ok rows={len(rows)} header_end={header_end} data_start={data_start} elapsed={dt:.3f}s out={out_path}")

    result.update({
        "ok": True,
        "output": out_path,
        "rows": len(rows),
        "header_end": header_end,
        "data_start": data_start,
        "elapsed": dt,
    })
    return result


# =========================
# GUI + Logging (thread-safe)
# =========================

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("基于ROM架构的tma数据一键解密系统")

        self.log_queue = queue.Queue()
        self.running_lock = threading.Lock()

        # --- Left controls (7 rows) ---
        self.btn_single = tk.Button(root, text="单文件快速执行", command=self.on_single_select)
        self.btn_multi = tk.Button(root, text="多文件快速执行", command=self.on_multi_select)

        # W row 1: output suffix note
        self.lbl_suffix = tk.Label(root, text="备注：输出后缀（空=txt；填 csv → .csv；填 .log → .log）")

        # R row 1: suffix entry
        self.suffix_var = tk.StringVar(value="")
        self.entry_suffix = tk.Entry(root, textvariable=self.suffix_var)

        # W row 2: name append note
        self.lbl_append = tk.Label(root, text="备注：文件名追加内容（空=不加；填 _A → 输出文件名末尾追加 _A）")

        # R row 2: append entry
        self.append_var = tk.StringVar(value="")
        self.entry_append = tk.Entry(root, textvariable=self.append_var)

        # F row
        self.mt_var = tk.BooleanVar(value=False)
        self.chk_mt = tk.Checkbutton(root, text="多线程（勾选后最多 6 线程）", variable=self.mt_var)

        # --- Right log ---
        self.txt_log = tk.Text(root, height=22, width=100, wrap="word")
        self.txt_log.configure(state="disabled")
        self.scroll = tk.Scrollbar(root, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=self.scroll.set)

        # --- Layout: 7 rows, left col + log colspans 3 + scrollbar ---
        # structure:
        # x1 L L L
        # x2 L L L
        # W  L L L
        # R  L L L
        # W  L L L
        # R  L L L
        # F  L L L

        self.btn_single.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.btn_multi.grid(row=1, column=0, sticky="ew", padx=8, pady=6)

        self.lbl_suffix.grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.entry_suffix.grid(row=3, column=0, sticky="ew", padx=8, pady=6)

        self.lbl_append.grid(row=4, column=0, sticky="w", padx=8, pady=6)
        self.entry_append.grid(row=5, column=0, sticky="ew", padx=8, pady=6)

        self.chk_mt.grid(row=6, column=0, sticky="w", padx=8, pady=6)

        self.txt_log.grid(row=0, column=1, rowspan=7, columnspan=3, sticky="nsew", padx=(8, 0), pady=6)
        self.scroll.grid(row=0, column=4, rowspan=7, sticky="ns", padx=(0, 8), pady=6)

        # Expand log area
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_columnconfigure(3, weight=1)
        for r in range(7):
            root.grid_rowconfigure(r, weight=1)

        # poll log queue
        self.root.after(80, self.flush_log_queue)

        # initial logs
        self.log("ROM架构准备就绪")
        self.log(f"任务ID {os.getpid()} ")
        self.log(f"Cpu线程 {os.cpu_count()}")
        self.log("")
        self.log("[执行规则]")
        self.log("1.选择文件，格式为 .001~.099(理论上tor文件也能执行)")
        self.log("2.读取UTF-16LE头部")
        self.log("3.使用ROM架构解析float32数据表")
        self.log("4.仿原生输出数据")
        self.log("")
        self.log("[输出后缀]")
        self.log("默认是空,后缀为txt")
        self.log("如果填写.003 则会变成xxx.003")
        self.log("")
        self.log("[文件名追加]")
        self.log("默认不追加，则 1.001-> 1.txt")
        self.log("否则在扩展名前追加(例如 _A 则 1.001-> 1_A.txt)")
        self.log("")
        self.log("[多线程]")
        self.log("多文件时显著加速，并发上限 6 线程")
        self.log("")
        

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.log_queue.put(f"[{ts}] {msg}\n")

    def flush_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.txt_log.configure(state="normal")
                self.txt_log.insert("end", line)
                self.txt_log.see("end")
                self.txt_log.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(80, self.flush_log_queue)

    def on_single_select(self):
        self.log("按钮1 clicked: 单个选择对话框打开")
        path = filedialog.askopenfilename(
            title="选择 .001~.099 文件（单个）",
            filetypes=[("001-099 Files", "*.0??"), ("All Files", "*.*")],
        )
        if not path:
            self.log("按钮1: 用户取消选择")
            return
        self.log(f"按钮1: 选择完成 -> {path} （立即执行）")
        self.start_processing([path])

    def on_multi_select(self):
        self.log("按钮2 clicked: 多个选择对话框打开")
        paths = filedialog.askopenfilenames(
            title="选择 .001~.099 文件（多个）",
            filetypes=[("001-099 Files", "*.0??"), ("All Files", "*.*")],
        )
        if not paths:
            self.log("按钮2: 用户取消选择")
            return

        paths = list(paths)
        self.log(f"按钮2: 选择完成 -> {len(paths)} 个文件（立即执行）")
        for p in paths[:30]:
            self.log(f"  - {p}")
        if len(paths) > 30:
            self.log(f"  ... 还有 {len(paths) - 30} 个未展开")

        self.start_processing(paths)

    def start_processing(self, paths):
        # Prevent overlapping runs
        if not self.running_lock.acquire(blocking=False):
            self.log("⚠️ 当前已有任务在运行中，忽略本次请求。")
            return

        suffix = self.suffix_var.get()
        append = self.append_var.get()
        use_mt = bool(self.mt_var.get())

        # validate/partition
        valid, invalid = [], []
        for p in paths:
            if is_valid_ext_001_099(p):
                valid.append(p)
            else:
                invalid.append(p)

        self.log("==========")
        self.log(f"[run] requested_files={len(paths)} valid={len(valid)} invalid={len(invalid)}")
        self.log(f"[run] suffix='{suffix}' append='{append}' multi_thread={use_mt}")

        if invalid:
            for p in invalid:
                self.log(f"[run] invalid_ext_skip: {p}")

        if not valid:
            self.log("[run] 没有可处理的有效文件（必须 .001~.099）。结束。")
            self.running_lock.release()
            return

        # run in background thread to keep UI responsive
        t = threading.Thread(target=self._run_worker, args=(valid, suffix, append, use_mt), daemon=True)
        t.start()

    def _run_worker(self, files, suffix, append, use_mt):
        try:
            self.log(f"[run] start batch files={len(files)} thread={threading.current_thread().name}")
            results = []
            errors = []

            if use_mt and len(files) > 1:
                max_workers = min(6, (os.cpu_count() or 1), len(files))
                self.log(f"[mt] ThreadPoolExecutor max_workers={max_workers}")

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    fut_map = {}
                    for f in files:
                        self.log(f"[mt] submit: {f}")
                        fut_map[ex.submit(self._safe_decrypt, f, suffix, append)] = f

                    for fut in as_completed(fut_map):
                        f = fut_map[fut]
                        try:
                            r = fut.result()
                            results.append(r)
                        except Exception as e:
                            tb = traceback.format_exc()
                            errors.append((f, e, tb))
                            self.log(f"[mt] ❌ exception in {f}: {e}")
                            self.log(tb)
            else:
                self.log("[st] single-thread mode")
                for i, f in enumerate(files, 1):
                    self.log(f"[st] ({i}/{len(files)}) processing: {f}")
                    try:
                        r = self._safe_decrypt(f, suffix, append)
                        results.append(r)
                    except Exception as e:
                        tb = traceback.format_exc()
                        errors.append((f, e, tb))
                        self.log(f"[st] ❌ exception in {f}: {e}")
                        self.log(tb)

            ok_count = sum(1 for r in results if r.get("ok"))
            fail_count = len(files) - ok_count

            self.log("==========")
            self.log(f"[run] finished: total={len(files)} ok={ok_count} fail={fail_count}")

            for r in results:
                if r.get("ok"):
                    self.log(f"[result] ✅ {r['input']} -> {r['output']} rows={r['rows']} "
                             f"header_end={r['header_end']} data_start={r['data_start']} elapsed={r['elapsed']:.3f}s")
                else:
                    self.log(f"[result] ❌ {r.get('input')} -> (no output)")

            if errors:
                self.log("==========")
                self.log("[errors] 发生错误的文件如下（含堆栈）：")
                for f, e, tb in errors:
                    self.log(f"[errors] file={f} err={e}")
                    self.log(tb)

            self.log("==========")
            self.log("[run] done.")
        finally:
            try:
                self.running_lock.release()
            except RuntimeError:
                pass

    def _safe_decrypt(self, path, suffix, append):
        return decrypt_one_file(path, suffix, append, log=self.log)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
