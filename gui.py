"""Audio2Score 图形界面：音频 → 钢琴谱 (MIDI / MusicXML)。

依赖 tkinter（Python 自带），后台线程调用 audio2score 流水线，避免卡界面。
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


class Audio2ScoreApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio2Score - 音频转钢琴谱")
        self.root.geometry("720x640")
        self.root.minsize(640, 560)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 输入/输出变量
        self.file_var = tk.StringVar()
        self.dir_var = tk.StringVar()
        # 转谱选项
        self.time_sig_var = tk.StringVar(value="自动")
        self.key_var = tk.StringVar()
        self.hand_split_var = tk.StringVar(value="60")
        self.musicxml_only_var = tk.BooleanVar()

        self._stop = False
        self.worker_thread = None

        self.create_widgets()

    # ---------- 界面 ----------
    def create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)  # 日志区可伸缩

        # 输入
        in_frame = ttk.LabelFrame(main, text="输入", padding="10")
        in_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        in_frame.columnconfigure(1, weight=1)

        ttk.Label(in_frame, text="音频文件:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(in_frame, textvariable=self.file_var).grid(row=0, column=1, sticky="we")
        ttk.Button(in_frame, text="选择文件", command=self.pick_file).grid(row=0, column=2, padx=(5, 0))

        ttk.Label(in_frame, text="音频文件夹:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        ttk.Entry(in_frame, textvariable=self.dir_var).grid(row=1, column=1, sticky="we", pady=(5, 0))
        ttk.Button(in_frame, text="选择文件夹", command=self.pick_dir).grid(row=1, column=2, padx=(5, 0), pady=(5, 0))

        ttk.Label(in_frame, text="（文件与文件夹二选一；输出固定为输入同目录）",
                  foreground="gray").grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 转谱选项
        opt_frame = ttk.LabelFrame(main, text="转谱选项", padding="10")
        opt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt_frame.columnconfigure(1, weight=1)

        ttk.Label(opt_frame, text="拍号:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Combobox(opt_frame, textvariable=self.time_sig_var,
                     values=["自动", "2/4", "3/4", "4/4", "6/8"]).grid(row=0, column=1, sticky="w")
        ttk.Label(opt_frame, text="（可手动输入其他拍号）", foreground="gray").grid(row=0, column=2, sticky="w", padx=(5, 0))

        ttk.Label(opt_frame, text="调号:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        ttk.Entry(opt_frame, textvariable=self.key_var, width=12).grid(row=1, column=1, sticky="w", pady=(5, 0))
        ttk.Label(opt_frame, text="（留空自动；填升号数如 2 或 -3，或音名如 D / b）",
                  foreground="gray").grid(row=1, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        ttk.Label(opt_frame, text="左右手分界音:").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        ttk.Entry(opt_frame, textvariable=self.hand_split_var, width=12).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Label(opt_frame, text="（MIDI 音高，默认 60=C4）", foreground="gray").grid(row=2, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        ttk.Checkbutton(opt_frame, text="只输出 MusicXML（跳过 MIDI）",
                        variable=self.musicxml_only_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 日志
        log_frame = ttk.LabelFrame(main, text="日志", padding="5")
        log_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.tag_config("error", foreground="red")

        # 按钮
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=3, column=0)
        self.start_btn = ttk.Button(btn_frame, text="开始转谱", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        ttk.Button(btn_frame, text="打开所在目录", command=self.open_output).grid(row=0, column=2)

    # ---------- 文件选择 ----------
    def pick_file(self):
        f = filedialog.askopenfilename(title="选择音频文件", filetypes=[
            ("音频文件", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("所有文件", "*.*")])
        if f:
            self.file_var.set(f)
            self.dir_var.set("")

    def pick_dir(self):
        d = filedialog.askdirectory(title="选择音频文件夹")
        if d:
            self.dir_var.set(d)
            self.file_var.set("")

    # ---------- 启动/停止 ----------
    def start(self):
        in_file = self.file_var.get().strip()
        in_dir = self.dir_var.get().strip()

        if in_file and in_dir:
            messagebox.showerror("错误", "文件与文件夹请二选一")
            return
        if not in_file and not in_dir:
            messagebox.showerror("错误", "请选择输入文件或文件夹")
            return

        # 收集输入文件；输出固定为输入同目录
        if in_file:
            files = [in_file]
            out_dir = os.path.dirname(in_file)
        else:
            files = sorted(os.path.join(in_dir, f) for f in os.listdir(in_dir)
                           if f.lower().endswith(AUDIO_EXTS))
            out_dir = in_dir
        if not files:
            messagebox.showerror("错误", "未找到音频文件")
            return
        os.makedirs(out_dir, exist_ok=True)

        # 选项
        opts = {"musicxml_only": self.musicxml_only_var.get(),
                "hand_split": int(self.hand_split_var.get() or 60)}
        ts = self.time_sig_var.get().strip()
        if ts and ts != "自动":
            opts["time_sig"] = ts
        key_spec = self.key_var.get().strip()
        if key_spec:
            opts["key_spec"] = key_spec

        self._stop = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.worker_thread = threading.Thread(
            target=self._worker, args=(files, out_dir, opts), daemon=True)
        self.worker_thread.start()
        self.log(f"开始转谱，共 {len(files)} 个文件（首次需加载模型，请稍候）")

    def stop(self):
        self._stop = True
        self.log("已请求停止（当前文件处理完后结束）")

    def _worker(self, files, out_dir, opts):
        from audio2score.pipeline import transcribe_to_score  # 延迟导入（torch 等较重）
        for i, f in enumerate(files, 1):
            if self._stop:
                self.log("已停止")
                break
            name = os.path.basename(f)
            self.log(f"[{i}/{len(files)}] 转谱: {name}")
            try:
                res = transcribe_to_score(f, out_dir, **opts)
                out = res.get("musicxml") or res.get("midi")
                self.log(f"    完成 -> {os.path.basename(out)}")
            except Exception as e:  # noqa: broad-except — 单文件失败不应中断批量
                self.log(f"    失败: {e}", error=True)
        self.log("全部处理完成")
        self.root.after(0, self._on_done)

    def _on_done(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    # ---------- 日志（线程安全） ----------
    def log(self, message, error=False):
        self.root.after(0, lambda: self._log_impl(message, error))

    def _log_impl(self, message, error=False):
        self.log_text.config(state=tk.NORMAL)
        tag = "error" if error else None
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def open_output(self):
        in_file = self.file_var.get().strip()
        in_dir = self.dir_var.get().strip()
        d = os.path.dirname(in_file) if in_file else in_dir
        if d and os.path.isdir(d):
            os.startfile(d)  # noqa: 仅在 Windows 使用
        else:
            messagebox.showinfo("提示", "请先选择输入文件或文件夹")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = Audio2ScoreApp()
        app.run()
    except Exception as e:  # noqa: broad-except — pythonw 无控制台，落盘并弹窗
        import traceback
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_error.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Audio2Score 启动失败", f"{e}\n\n详情见 gui_error.log")
        except Exception:
            pass
