# Audio2Score

音频 → 钢琴双谱表乐谱（MIDI / MusicXML）。把钢琴音乐（当前以独奏为主）自动转成规范钢琴谱，最后在 MuseScore 人工微调。

## 当前状态（2026-08-20）

- ✅ M0 环境 + AMT 选型（ByteDance `piano_transcription_inference`）
- ✅ M1 流水线骨架
- ✅ M2 记谱：速度 / 调号 / 拍号 / 首音 / 时值 / 末尾 / 左右手分割 / 音符拼写 / 拍号检测
- ✅ downbeat 对齐（madmom-infer）：小节线对齐真实强拍、tempo 倍速纠正、弱起处理
- ✅ 休止符可见（makeRests 补 `timeRangeFromBarDuration`）+ subdiv 自适应量化
- ⚠️ 踏板记号（已禁用，待重写）；左右手交叉 ~10 音（接受手动修）
- ✅ M3 GUI（tkinter）
- ❌ 暂缓：带伴奏分轨（demucs/UVR，torchcodec+FFmpeg 版本冲突）

## 目录结构

```
audio2score/
  audio.py        # 音频加载/重采样 16kHz 单声道
  transcribe.py   # ByteDance AMT：音频→MIDI（补前导静音、自写MIDI、带正确速度）
  tempo.py        # 速度/拍号/相位检测
  notation.py     # MIDI→MusicXML（量化/分左右手/调号/拼写/踏板）
  pipeline.py     # 串联：音频→MIDI→MusicXML
  cli.py          # 命令行入口（python -m audio2score）
gui.py            # tkinter GUI
run_gui.bat       # 双击启动 GUI
README.md         # 安装说明（唯一依据，不写 requirements.txt）
testcases/        # 测试样例（case01 = Paper Lily OST 钢琴翻奏）
```

## 运行

```bash
# 命令行
python -m audio2score input.wav -o out_dir \
  [--time-sig 3/4] [--key 2] [--hand-split 60] [--midi-only] [--cpu]

# GUI
双击 run_gui.bat
```

## 环境（重要）

- **不写 `requirements.txt`**（CUDA torch 会让 PyCharm 卡死），安装说明统一写在 `README.md`。
- conda env `audio2score`，Python 3.11，torch 2.8.0+cu126（RTX 4060 8G，Driver 560.94）。
- AMT 权重：`%USERPROFILE%\piano_transcription_inference_data\note_F1=0.9677_pedal_F1=0.9186.pth`（171.9MB，已就位）。

## 关键技术决策与踩坑

- **AMT**：ByteDance（case01 F1 0.97），带踏板 CC64。
- **music21 双谱表**：`Score → 两个 PartStaff + layout.StaffGroup(brace)`，**不要**把两个 PartStaff 包进 `stream.Part`（否则导出 makeNotation 报 no measures）。
- **时值**：AMT offset 不可靠（系统性偏长），`_clip_offsets` 用「整数网格下标 + 截到下一个音 + 全局典型时值」；曾踩浮点舍入 bug（音符把自身 onset 当"下一个音"→ 截成 0.1 拍），已用整数下标修复。
- **左右手**：`_split_hands` 改为**和弦感知 + 时变分割点**（把同时发声的音分组成和弦→每和弦单调分割→EM 循环估计左右手时变音域并重分割），比旧贪心（质心追踪+固定音域先验 60）好：case01/reference.mid 误分 13→10（92.4%→94.2%）。**关键教训**：固定阈值 60 在交叉区两头都错——开头的 LH 和弦顶到 66（旋律在 71+），中段的 RH 旋律下探到 57（低音在 44-54），分割点实际在 ~55 与 ~68 之间移动。剩余 ~10 误分全是重叠区（57-66）单音的**声部连续**歧义（旋律下探/低音上浮），规则法到顶了，要更高需序列/图模型（BACHprop ~92% 并不比本启发式好；piano_svsep GNN 才是真 SOTA）。`--hand-split` 现在只作初始先验。
- **音符拼写**：`_spell_note` 按调号五度圈拼（Bb→A#、E#→F♮，自然音优先）。
- **拍号**：`detect_meter` 返回 (num, den)，自相关 + 倍频消歧；denominator 固定 4（`_is_compound` 复合拍检测对 tempo 敏感已弃用，复合拍靠 `--time-sig` 手动指定）。
- **downbeat**：madmom-infer（纯 numpy）检测下拍 + tempo + 每小节拍数；madmom 的 tempo/beats_per_bar 比 AMT/librosa 可靠，覆盖 AMT 倍速误判（大切な約束 AMT 157.8 → 实际 79）。`RNNDownBeatProcessor` 硬编码 44.1kHz，须在 pipeline 层用 44.1k 音频。
- **长音截断**：`_clip_offsets` 截到 measure 边界（`measure_grid = numerator*subdiv`），避免 onset 不在下拍时跨小节（超满）。
- **休止符可见**：makeMeasures 后对每个 measure 调 `makeRests(fillGaps=True, timeRangeFromBarDuration=True, hideRests=False, inPlace=True)`。**仅 `fillGaps=True` 不够**——它只填前导+内部空隙；不加 `timeRangeFromBarDuration=True` 末尾空隙（末音→小节线）仍缺，m21ToXml 导出时生成 `print-object="no"` 隐形平衡休止符（大切な約束 12 处→0）。可选升级标准 pickup 小节（`Measure.paddingLeft`）。
- **subdiv 自适应**：`_detect_subdiv` 按 onset 间隔自动选 2/4（<0.375 拍的比例 ≥8% 且 ≥3 处则用 4）。**纠正旧结论**：实测大切な約束 subdiv=2 与 4 输出完全相同（388 对象、443 音全保留），并无"十六分音符被合并"——早前 443→378 实为超满截断 bug（已修），非 16th 合并。现有 testcase 均判 2（无 16th），故是 no-op，但对真正 16th 密集曲目有效（合成 16th 输入返回 4）。
- **踏板（已禁用，待重写）**：AMT 踏板踩点不落音符、抖动噪声大，多次尝试（单 spanner+bounce/gap、每段 spanner+音符锚、按小节对齐）都不满意，已禁用。保留的结论供日后重写：`pedalForm=Line` 与 `case01/reference.musicxml` 一致（干净 start/stop 对）；music21 给每个 spanner 分配 `number`（MuseScore 会画多路踏板线，需 `_strip_pedal_numbers` 剥掉）；`SpannerAnchor` 排序靠后乱序故弃用；相邻 spanner 共享边界音符时 music21 把下一 start 排在上一 stop 前。

## 下一步（2026-08-20 待办）

- 踏板重写（已禁用；AMT CC64 抖动噪声大，需先想清楚降噪/锚定方案）。
- 清理：未用依赖 miditoolkit/partitura（`estimate_phase`、`_is_compound` 死代码已删，只剩这两个未用包）。
- 带伴奏分轨：**走"独立环境装最新 demucs + subprocess 调 CLI"路线**，勿在 audio2score 混装（torchcodec 版本矩阵已踩坑）。
- 复调/多声部分离（复杂钢琴曲质量上限）。**已试 piano_svsep（CPJKU GNN）并回退**：`svsep` 独立 env（torch 2.8.0+cpu + torch-geometric，torch_scatter 用纯 PyTorch 替身）曾搭好跑通；实测 case01 只有 94.8%（仅比规则法 94.2% 高 1 音），对 pop/OST 钢琴不值当。**关键坑**：其导出层（partitura `save_musicxml`）对多声部/跨谱表序列化是坏的（`<staves>1` 却用 staff 2、声部时值溢出、非法时值），MuseScore 拒收/排版乱——接入时不能直接用它的导出，得只取它的谱表标签、自己用 music21 重建。结论：价值在跨谱表/多声部古典（DCML），对当前 pop/OST 需求不接入，环境已清理。留作长期目标。
