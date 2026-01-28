import os
import tkinter as tk
from tkinter import filedialog, messagebox
import datetime
import time
"""
错误码
20 数字类型错误
21 信息不全
777 未知错误
v1 做好了
v2 编码换成utf-16 le 不然有问题
v3 默认选择txt，更方便了
"""
#
last_output_dir = None

# ==========================
#  GUI 日志输出函数
# ==========================


def log(msg: str):
    now = datetime.datetime.now().strftime("[%Y%m%d %H:%M:%S]")
    log_text.insert(tk.END, f"{now} {msg}\n")
    log_text.see(tk.END)

class ShuJuChuLi():
    """后端"""
    def __init__(self, m):
        self.m = m
        log(f"初始化 Shujuchuli 类，传入的重量 m 为: {self.m}")

    def daoru_duqu(self, path1, path2):
        if not path1.endswith('.txt'):
            log(f"文件 {path1} 不是一个txt文件！")
            return

        encodings = ['utf-16le', 'utf-8', 'gbk']
        lines = None

        for encoding in encodings:
            try:
                with open(path1, 'r', encoding=encoding) as file:
                    lines = file.readlines()
                log(f"成功使用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                log(f"使用 {encoding} 编码读取文件失败，尝试下一种编码...")

        if lines is None:
            log("所有编码格式均无法读取文件！")
            return

        start_of_data_index = None

        for idx, line in enumerate(lines):
            if "StartOfData" in line:
                start_of_data_index = idx
                #log(f"搜索关键词 StartOfData")
                log(f"找到StartOfData在第 {idx + 1} 行")
                break

        if start_of_data_index is None:
            log("没有找到 'StartOfData'。")
            return

        part1 = lines[:start_of_data_index + 1]
        part2 = lines[start_of_data_index + 1:]

        with open(path2, 'w', encoding='utf-16-le') as new_file:
            new_file.writelines(part1)

        processed_part2 = self.chuli(''.join(part2))

        with open(path2, 'a', encoding='utf-16-le') as new_file:
            new_file.write(processed_part2)

        log("文件处理完成。")

    def chuli(self, data):
        zhongliang = float(self.m)
        processed_lines = []

        for line in data.splitlines():
            parts = line.split()

            if len(parts) >= 3:
                processed_line = ""
                part2 = float(parts[1])
                part3 = float(parts[2])

                if 0 < part2 < 999:
                    new_part3 = self.chuli_fangsuo(zhongliang, part3)
                    parts[2] = new_part3

                for yuansu in parts:
                    processed_line = processed_line + " " + str(yuansu)
                processed_lines.append(processed_line)
            else:
                return

        return '\n'.join(processed_lines)

    def kong():
        pass

    def chuli_fangsuo(self, m: float, h: float):
        if type(m) == int or type(m) == float:
            if m >= 1 or m <= 0:
                log("[Math Erro] 质量不该大于等于1或者小于等于0")
                return 20
            else:
                New_h = h * 0.5 / m
                return New_h
            return 777
        return 20


# ==========================
#      功能函数
# ==========================
def choose_file(entry1, entry2):
    log(f"打开提示框选择文件")
    file_path = filedialog.askopenfilename(
        title="选择文件",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if file_path:
        
        entry1.delete(0, tk.END)
        entry1.insert(0, file_path)

        renamed_file = rename_file(file_path)
        entry2.delete(0, tk.END)
        entry2.insert(0, renamed_file)

        log(f"选择源文件：{file_path}")
        log(f"输出文件自动重命名：{renamed_file}")

        


def rename_file(file_path):
    new_file_path = file_path.replace(".", "_new.")
    #log(f"输出文件重命名：{new_file_path}")
    return new_file_path


def save_file_as(entry2):
    file_path = filedialog.asksaveasfilename(
        title="保存文件为",
        filetypes=[
            ("txt文件", "*.txt"),
            ("All files", "*.*"),
            ("恭喜你发现小彩蛋", "*.xiaocaidan001"),
            ("福建锦浪新材料", "*.xiaocaidan002"),
            ("研发部特别定制款", "*.xiaocaidan003")
        ]
    )
    if file_path:
        entry2.delete(0, tk.END)
        entry2.insert(0, file_path)
        log(f"选择输出文件：{file_path}")


def execute_action(entry1, entry2, entry3):
    value1 = entry1.get()
    value2 = entry2.get()
    value3 = entry3.get()

    b = ["源文件位置", "新文件输出位置", "样品重量"]
    for i, v in enumerate([value1, value2, value3]):
        if not v:
            messagebox.showinfo("发生错误", f"[缺少{b[i]}数据]\n若使用软件存在问题\n请微信联系研发部主管.")
            log(f"缺少参数：{b[i]}")
            return

    log("开始执行处理流程...")
    log(f"源文件: {value1}")
    log(f"输出文件: {value2}")
    log(f"样品重量: {value3}")

    main_hanshu = ShuJuChuLi(m=value3)
    main_hanshu.daoru_duqu(value1, value2)

    messagebox.showinfo("处理完成", f"源文件路径: {value1}\n新文件路径: {value2}\n原始重量: {value3}")
    log("所有处理已完成。")


def quit_app(root):
    log("程序执行退出。")
    #time.sleep(5)
    root.destroy()


def resize_widgets(event):
    button_width = 10
    entry_shurulujing.config(width=button_width * 5)
    entry_shuchulujing.config(width=button_width * 5)
    entry3_entry.config(width=button_width * 3)


# ==========================
#        GUI 布局
# ==========================

root = tk.Tk()
root.title("锦浪研发部TMA统一数据管理器")
root.geometry("800x400")
root.resizable(True, True)

# 窗口居中
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 400) // 2
y = (screen_height - 300) // 2
root.geometry(f"+{x}+{y}")

# 控件
button1 = tk.Button(root, text="选择源文件", command=lambda: choose_file(entry_shurulujing, entry_shuchulujing))
button1.grid(row=0, column=0, padx=5, pady=5)
"""
在网格布局中放置按钮控件
    Args:
        row (int): 按钮所在的行号
        column (int): 按钮所在的列号
        padx (int): 水平方向的内边距（像素）
        pady (int): 垂直方向的内边距（像素）
"""
#entry_shurulujing是输入路径的位置
entry_shurulujing = tk.Entry(root)
entry_shurulujing.insert(0, "点击左侧按钮可以选择源文件")
#entry_shuchulujing是输出路径的位置
entry_shuchulujing = tk.Entry(root)

entry3_frame = tk.Frame(root,width=10,height=10)
entry3_frame.grid_propagate(False)
"""
创建一个新的Frame容器用于组织entry3控件\n
Args:root (tk.Tk): 父级窗口或容器对象
"""

entry3_title = tk.Label(entry3_frame, text="TMA样品重量", anchor="w")
entry3_entry = tk.Entry(entry3_frame, width=20)
entry3_entry.insert(0, "0.50")

button2 = tk.Button(root, text="保存的位置", command=lambda: save_file_as(entry_shuchulujing))
button3 = tk.Button(root, text="执行统一器", command=lambda: execute_action(entry_shurulujing, entry_shuchulujing, entry3_entry))
button4 = tk.Button(root, text="退出", command=lambda: quit_app(root))

canvas = tk.Canvas(root, width=1, bg="black", height=100)

entry_shurulujing.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
canvas.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="ns")

entry3_frame.grid(row=0, column=3, rowspan=2, padx=5, pady=5, sticky="nsew")
entry3_title.pack(fill="x", padx=5, pady=2)
entry3_entry.pack(fill="x", padx=5, pady=2)

button2.grid(row=1, column=0, padx=5, pady=5)
entry_shuchulujing.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

button3.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
button4.grid(row=2, column=1, padx=5, pady=5, sticky="ew")


# ==========================================================
#               日志输出框（新增）
# ==========================================================
log_frame = tk.Frame(root)
log_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=5, pady=5)

scrollbar = tk.Scrollbar(log_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

log_text = tk.Text(log_frame, height=8, yscrollcommand=scrollbar.set)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar.config(command=log_text.yview)

# 布局伸缩
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(1, weight=1)

root.bind("<Configure>", resize_widgets)

root.mainloop()
