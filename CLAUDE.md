# Audio2Score

音频 → 钢琴双谱表乐谱（MIDI / MusicXML）。把钢琴音乐（当前以独奏为主）自动转成规范钢琴谱，最后在 MuseScore 人工微调。

## 当前状态（2026-08-20）

- ✅ M0 环境 + AMT 选型（ByteDance `piano_transcription_inference`）
- ✅ M1 流水线骨架
- ✅ M2 记谱：速度 / 调号 / 拍号 / 首音 / 时值 / 末尾 / 左右手分割 / 音符拼写 / 拍号检测
- ✅ downbeat 对齐（madmom-infer）：小节线对齐真实强拍、tempo 倍速纠正、弱起处理、3/4 vs 6/4 消歧
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
- **左右手**：`_split_hands` 改为**和弦感知 + 时变分割点**（把同时发声的音分组成和弦→每和弦单调分割→EM 循环估计左右手时变音域并重分割），比旧贪心（质心追踪+固定音域先验 60）好：case01/reference.mid 误分 13→10（92.4%→94.2%）。**关键教训**：固定阈值 60 在交叉区两头都错——开头的 LH 和弦顶到 66（旋律在 71+），中段的 RH 旋律下探到 57（低音在 44-54），分割点实际在 ~55 与 ~68 之间移动。剩余 ~10 误分全是重叠区（57-66）单音的**声部连续**歧义（旋律下探/低音上浮），规则法到顶了。**已试谱聚类也不如 DP**：无先验 58%（Fiedler 找到的是时间切分「早/晚」不是「左/右」）、固定音域锚点 86%（固定阈值两头错）——因为两只手时间上交错，无监督聚类发现不了「左右手」，关键在**时变**音域先验。要更高需序列/图模型（BACHprop ~92% 不比本启发式好；piano_svsep GNN 实测 94.8% 也仅高 1 音）。`--hand-split` 现在只作初始先验。
- **音符拼写**：`_spell_note` 按调号五度圈拼（Bb→A#、E#→F♮，自然音优先）。
- **拍号**：`detect_meter` 返回 (num, den)，自相关 + 倍频消歧；denominator 固定 4（`_is_compound` 复合拍检测对 tempo 敏感已弃用，复合拍靠 `--time-sig` 手动指定）。
- **downbeat**：madmom-infer（纯 numpy）检测下拍 + tempo + 每小节拍数；madmom 的 tempo/beats_per_bar 比 AMT/librosa 可靠，覆盖 AMT 倍速误判（大切な約束 AMT 157.8 → 实际 79）。`RNNDownBeatProcessor` 硬编码 44.1kHz，须在 pipeline 层用 44.1k 音频。
- **3/4 vs 6/4 消歧（已修）**：6/4 是「两个 3/4 拍组」，madmom 的 HMM 在 3 和 6 之间分不清，`With All Your Heart` 被误判 6/4（实际 3/4）。修法在 `tempo.py:detect_downbeats`：当 bpb=6 时用 `[3]` 重解码，比较「下拍激活强度 / 非下拍激活强度」的分离度（`_downbeat_separation`），3/4 的解释分离度更高（实测 1.82 vs 1.58）则取 3/4。原理：真 3/4 的下拍激活在 beat1≈beat4（两者都是真下拍），6/4 把 beat4 当非下拍拉低了分离度。真 6/4 仍靠 `--time-sig 6/4` 手动指定（本启发式不会误拆真 6/4，因真 6/4 的 beat4 下拍激活弱）。
- **长音截断**：`_clip_offsets` 截到 measure 边界（`measure_grid = numerator*subdiv`），避免 onset 不在下拍时跨小节（超满）。
- **休止符可见**：makeMeasures 后对每个 measure 调 `makeRests(fillGaps=True, timeRangeFromBarDuration=True, hideRests=False, inPlace=True)`。**仅 `fillGaps=True` 不够**——它只填前导+内部空隙；不加 `timeRangeFromBarDuration=True` 末尾空隙（末音→小节线）仍缺，m21ToXml 导出时生成 `print-object="no"` 隐形平衡休止符（大切な約束 12 处→0）。可选升级标准 pickup 小节（`Measure.paddingLeft`）。
- **subdiv 自适应**：`_detect_subdiv` 按 onset 间隔自动选 2/4（<0.375 拍处 ≥2 个即用 4，**已去掉旧的比例 ≥8% 门槛**——它会漏掉稀疏 16 分，如 `With All Your Heart` 16 分间隔占比仅 0.4% 却真实存在）。**纠正旧结论**：实测大切な約束 subdiv=2 与 4 输出完全相同（388 对象、443 音全保留），16 分网格是八分网格的严格超集、无副作用。修后 `With All Your Heart` 16 分音符 92→152、柱式和弦拆开（分解琶音正确写成 16 分）。**滚奏（和弦+波浪线）另论**：AMT 把滚奏输出成 <5ms 几乎同时 onset，MIDI 里滚奏信息已丢，记谱层无法从 onset 恢复；需回 AMT 帧级 `reg_onset_output` 才能救。用户决定**不做**滚奏标记（同踏板）。若日后要做：`ArpeggioMark` 是 Expression 非 Articulation，须 `chord.expressions.append(ArpeggioMark('normal'))`（放 `.articulations` 会崩 `no attribute 'placement'`），导出 `<arpeggiate/>`。
- **踏板（已禁用，待重写）**：AMT 踏板踩点不落音符、抖动噪声大，多次尝试（单 spanner+bounce/gap、每段 spanner+音符锚、按小节对齐）都不满意，已禁用。保留的结论供日后重写：`pedalForm=Line` 与 `case01/reference.musicxml` 一致（干净 start/stop 对）；music21 给每个 spanner 分配 `number`（MuseScore 会画多路踏板线，需 `_strip_pedal_numbers` 剥掉）；`SpannerAnchor` 排序靠后乱序故弃用；相邻 spanner 共享边界音符时 music21 把下一 start 排在上一 stop 前。

## 下一步（2026-08-20 待办）

- ✅ **#1 量化/时值（已修）**：放宽 `_detect_subdiv`（去掉比例门槛），16 分/分解琶音不再被并成柱式和弦。`With All Your Heart` 16 分 92→152、overfull=0。滚奏标记用户决定**不做**（AMT 帧级可救但价值低、检测不确定，同踏板）。
- ✅ **#2 拍号/下拍（已修）**：`detect_downbeats` 加 3/4 vs 6/4 消歧（下拍激活分离度）。`With All Your Heart` 6/4→3/4（200 downbeats）。case01/大切な約束 无回归（3/4、4/4 不变）。复合拍（6/8）denominator 仍靠 `--time-sig` 手动指定。
- **#8 聚类（已探索完，结论：不如 DP）**：谱聚类无先验 58%（Fiedler 找到时间切分）、固定音域锚点 86%（固定阈值两头错），都不如现有 DP 94.2%。结论：无监督聚类发现不了左右手，关键在**时变**音域先验（DP 已做）；剩余 ~6% 交叉音需声部序列语义。⚠️**用户的核心诉求没满足**：要的是「聚类接入流水线、产出实际的 .musicxml 文件」来肉眼判断，而不是准确率数字——之前一直在算百分比/渲染图片，没产出聚类版 .musicxml。
- 踏板重写：用户决定**不做**。
- 带伴奏分轨（demucs/UVR）：用户决定**不做**（已用商业软件验证过可行性）。
- 依赖清理（miditoolkit/partitura）：**暂不清理**。
- git：已提交（含 .gitignore、run_gui_example.bat、README 完善）。LICENSE 未定（MIT 或 Apache 2.0）。
- 复调/多声部分离：**暂缓**。已试 piano_svsep（CPJKU GNN）并回退：实测 case01 94.8%（仅比规则法 94.2% 高 1 音），导出层（partitura `save_musicxml`）对多声部/跨谱表是坏的（`<staves>1` 却用 staff 2、声部时值溢出、非法时值），环境已清理。价值在跨谱表/多声部古典（DCML）。
