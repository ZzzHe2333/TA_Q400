import os
import sys
import threading
import subprocess
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import locale


class PackerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tk 打包器 (PyInstaller)")
        self.geometry("860x620")
        self.minsize(820, 560)

        self.log_queue = queue.Queue()
        self.proc = None
        self.worker_thread = None

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        # ===== 顶部区域：路径选择 =====
        frm_top = ttk.LabelFrame(self, text="路径选择")
        frm_top.pack(fill="x", padx=10, pady=8)

        self.py_path_var = tk.StringVar()
        self.ico_path_var = tk.StringVar()

        # Python脚本路径
        row1 = ttk.Frame(frm_top)
        row1.pack(fill="x", padx=8, pady=8)
        ttk.Label(row1, text="Python 脚本(.py)：", width=16).pack(side="left")
        py_entry = ttk.Entry(row1, textvariable=self.py_path_var)
        py_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="选择...", command=self.pick_py).pack(side="left", padx=4)

        # ICO路径
        row2 = ttk.Frame(frm_top)
        row2.pack(fill="x", padx=8, pady=(0, 10))
        ttk.Label(row2, text="图标(.ico，可选)：", width=16).pack(side="left")
        ico_entry = ttk.Entry(row2, textvariable=self.ico_path_var)
        ico_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="选择...", command=self.pick_ico).pack(side="left", padx=4)
        ttk.Button(row2, text="清空图标", command=lambda: self.ico_path_var.set("")).pack(side="left", padx=4)

        # ===== 中部区域：可选参数 =====
        frm_opts = ttk.LabelFrame(self, text="打包可选参数（复选框）")
        frm_opts.pack(fill="x", padx=10, pady=8)

        self.opt_onefile = tk.BooleanVar(value=True)       # 常用：单文件
        self.opt_noconsole = tk.BooleanVar(value=False)    # GUI 程序常用
        self.opt_clean = tk.BooleanVar(value=True)
        self.opt_noconfirm = tk.BooleanVar(value=True)
        self.opt_noupx = tk.BooleanVar(value=False)
        self.opt_strip = tk.BooleanVar(value=False)        # 有些环境/平台不一定有效

        grid = ttk.Frame(frm_opts)
        grid.pack(fill="x", padx=10, pady=10)

        ttk.Checkbutton(grid, text="--onefile（生成单个 exe）", variable=self.opt_onefile).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(grid, text="--noconsole（无控制台窗口/GUI 程序用）", variable=self.opt_noconsole).grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Checkbutton(grid, text="--clean（清理临时缓存）", variable=self.opt_clean).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(grid, text="--noconfirm（覆盖不询问）", variable=self.opt_noconfirm).grid(row=1, column=1, sticky="w", padx=8, pady=6)

        ttk.Checkbutton(grid, text="--noupx（不使用 UPX 压缩）", variable=self.opt_noupx).grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(grid, text="--strip（尝试剥离符号）", variable=self.opt_strip).grid(row=2, column=1, sticky="w", padx=8, pady=6)

        for c in range(2):
            grid.columnconfigure(c, weight=1)

        # ===== 执行按钮 =====
        frm_run = ttk.Frame(self)
        frm_run.pack(fill="x", padx=10, pady=8)

        self.run_btn = tk.Button(
            frm_run,
            text="开始打包为 EXE",
            font=("Microsoft YaHei UI", 16, "bold"),
            height=2,
            command=self.start_build
        )
        self.run_btn.pack(fill="x")

        # ===== 底部 log =====
        frm_log = ttk.LabelFrame(self, text="日志输出")
        frm_log.pack(fill="both", expand=True, padx=10, pady=8)

        self.log_text = tk.Text(frm_log, height=12, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.log_text.configure(state="disabled")

        sb = ttk.Scrollbar(frm_log, orient="vertical", command=self.log_text.yview)
        sb.pack(side="right", fill="y", padx=8, pady=8)
        self.log_text.configure(yscrollcommand=sb.set)

        # 初始提示
        self.log("提示：请先确保安装 PyInstaller：pip install pyinstaller\n")

    def pick_py(self):
        path = filedialog.askopenfilename(
            title="选择 Python 脚本",
            filetypes=[("Python Script", "*.py"), ("All Files", "*.*")]
        )
        if path:
            self.py_path_var.set(os.path.abspath(path))

    def pick_ico(self):
        path = filedialog.askopenfilename(
            title="选择 ICO 图标（可选）",
            filetypes=[("Icon", "*.ico"), ("All Files", "*.*")]
        )
        if path:
            self.ico_path_var.set(os.path.abspath(path))

    def build_command(self, script_path, ico_path):
        cmd = [sys.executable, "-m", "PyInstaller"]

        if self.opt_onefile.get():
            cmd.append("--onefile")
        if self.opt_noconsole.get():
            cmd.append("--noconsole")
        if self.opt_clean.get():
            cmd.append("--clean")
        if self.opt_noconfirm.get():
            cmd.append("--noconfirm")
        if self.opt_noupx.get():
            cmd.append("--noupx")
        if self.opt_strip.get():
            cmd.append("--strip")

        if ico_path:
            cmd += ["--icon", ico_path]

        cmd.append(script_path)
        return cmd

    def start_build(self):
        script_path = self.py_path_var.get().strip()
        ico_path = self.ico_path_var.get().strip()

        if not script_path:
            messagebox.showwarning("缺少脚本", "请先选择要打包的 Python 脚本(.py)。")
            return
        if not os.path.isfile(script_path):
            messagebox.showerror("路径错误", f"脚本不存在：\n{script_path}")
            return

        if ico_path and (not os.path.isfile(ico_path)):
            messagebox.showerror("路径错误", f"ICO 不存在：\n{ico_path}")
            return

        cmd = self.build_command(script_path, ico_path if ico_path else "")
        workdir = os.path.dirname(script_path)

        self.run_btn.config(state="disabled")
        self.log("\n" + "=" * 70)
        self.log("开始打包...\n")
        self.log(f"工作目录：{workdir}\n")
        self.log("命令：\n" + " ".join(self._quote_if_needed(x) for x in cmd) + "\n\n")

        self.worker_thread = threading.Thread(
            target=self._run_pyinstaller,
            args=(cmd, workdir),
            daemon=True
        )
        self.worker_thread.start()

    def _run_pyinstaller(self, cmd, workdir):
        # 先简单检查 PyInstaller 是否可用
        try:
            check = subprocess.run(
                [sys.executable, "-m", "PyInstaller", "--version"],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=locale.getpreferredencoding(False),
                errors="replace"
            )
            if check.returncode != 0:
                self.log_queue.put(("ERROR", "未检测到可用的 PyInstaller。请先运行：pip install pyinstaller\n"))
                self.log_queue.put(("DONE", None))
                return
        except Exception as e:
            self.log_queue.put(("ERROR", f"检查 PyInstaller 失败：{e}\n"))
            self.log_queue.put(("DONE", None))
            return

        try:
            enc = locale.getpreferredencoding(False)

            self.proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding=enc,
                errors="replace"
            )

            for line in self.proc.stdout:
                self.log_queue.put(("LINE", line))

            rc = self.proc.wait()
            if rc == 0:
                self.log_queue.put(("OK", "\n✅ 打包完成！输出一般在脚本目录下的 dist/ 里。\n"))
            else:
                self.log_queue.put(("ERROR", f"\n❌ 打包失败，返回码：{rc}\n"))

        except Exception as e:
            self.log_queue.put(("ERROR", f"\n运行 PyInstaller 出错：{e}\n"))
        finally:
            self.log_queue.put(("DONE", None))

    def _poll_log_queue(self):
        try:
            while True:
                typ, payload = self.log_queue.get_nowait()
                if typ == "LINE":
                    self.log(payload)
                elif typ == "OK":
                    self.log(payload)
                elif typ == "ERROR":
                    self.log(payload)
                elif typ == "DONE":
                    self.run_btn.config(state="normal")
                    self.proc = None
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _quote_if_needed(s: str) -> str:
        if any(ch.isspace() for ch in s):
            return f'"{s}"'
        return s


if __name__ == "__main__":
    app = PackerGUI()
    app.mainloop()
