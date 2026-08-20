# Audio2Score

音频 → 钢琴双谱表乐谱（MusicXML）。目标：把钢琴音乐（独奏或带伴奏）自动转成规范钢琴谱，最后在 MuseScore 人工微调。

## 架构（路线 B，双入口）

```
带伴奏音频 ──► UVR htdemucs_6s 分轨 ──► 钢琴轨 ─┐
独奏钢琴 ─────────────────────────────────────┴─► ByteDance AMT ──► MIDI
                                                        │
      MusicXML ◄── 量化 / 分左右手 / 调号 / 踏板记号 ◄──────┘
        │
        ▼
MuseScore 人工微调
```

## 环境安装

> ⚠️ 本项目**故意不提供 `requirements.txt`**：torch 的 CUDA 本地版本标识（`torch==2.8.0+cu126`）会导致 PyCharm 的依赖分析卡死。

### 1. conda 环境（Python 3.11）

```bash
conda create -n audio2score python=3.11 -y
conda activate audio2score
```

> 必须 3.11：`basic-pitch` 等老库不支持 3.12+。

### 2. PyTorch（GPU / CUDA 12.6）

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

> 需 NVIDIA 驱动 ≥ 560（支持 CUDA 12.6 运行时）。纯推理**无需**另装 CUDA Toolkit——wheel 自带 runtime。

### 3. 音频 / MIDI / 乐谱库

```bash
pip install librosa                  # 自动带 scipy / numba / soundfile
pip install pretty_midi miditoolkit
pip install music21 partitura
pip install madmom-infer               # 下拍检测（纯 numpy 的 madmom 重写，无编译）
```

### 4. 钢琴转录模型（ByteDance）

```bash
pip install piano_transcription_inference
```

#### 4.1 下载模型权重（~172 MB）

包首次运行会用 `wget` 下载，但 **Windows 下 `wget` 不可用**，会静默失败，因此需手动下载到：

```
%USERPROFILE%\piano_transcription_inference_data\note_F1=0.9677_pedal_F1=0.9186.pth
```

下载地址（Zenodo）：

```
https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1
```

- 完整文件约 **172 MB**（代码要求 >160 MB 才判定完整）。

#### 4.2 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期输出含 `True` 与 `NVIDIA GeForce RTX 4060`。

## 模型信息

| 项 | 值 |
|---|---|
| 模型 | ByteDance High-Resolution Piano Transcription（`Note_pedal` 版） |
| 输入 | 16 kHz 单声道 |
| 输出 | MIDI（含踏板 CC64 事件） |
| 音域 | 88 键（A0=21 ~ C8=108） |
| 帧率 | 100 fps |
| 显存占用 | 峰值 ~357 MB（8GB 显存富余） |

## 使用

### 命令行

```bash
# 音频 → MIDI + MusicXML（双谱表）
python -m audio2score input.wav -o out_dir

# 只要 MIDI
python -m audio2score input.wav -o out_dir --midi-only

# 手动指定拍号 / 调号 / 强制 CPU
python -m audio2score input.wav -o out_dir --time-sig 3/4 --key 2 --cpu

# 左右手初始分界音（默认 60=C4；实际分割点随时间变化，此值只作初始先验）
python -m audio2score input.wav -o out_dir --hand-split 60
```

### GUI

双击 `run_gui_example.bat`（先复制为本机 `run_gui.bat` 并改成本机 conda 路径），或直接：

```bash
python gui.py
```

### Python API

```python
from audio2score.pipeline import transcribe_to_score

res = transcribe_to_score("input.wav", "output_dir")
# {'tempo': 150.0, 'midi': 'output_dir/input.mid', 'musicxml': 'output_dir/input.musicxml'}
```

## 目录结构

```
audio2score/
  audio.py       # 音频加载/重采样
  transcribe.py  # ByteDance AMT：音频 → MIDI
  tempo.py       # 速度/拍号/下拍检测
  notation.py    # MIDI → MusicXML（量化/左右手分割/调号/拼写）
  pipeline.py    # 串联：音频 → MIDI → MusicXML
  cli.py         # 命令行入口（python -m audio2score）
gui.py           # tkinter 图形界面
run_gui_example.bat  # 双击启动 GUI（通用示例）
README.md        # 说明
```
