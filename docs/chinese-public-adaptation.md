# 中文公开数据适配 / Chinese public-data adaptation

[中文操作说明](#数据权重和运行环境) · [English guide](#english-guide)

只用 train/dev 选择，再冻结测试。 / Select on train/dev, then freeze before testing.

本流程用于优化**纯视觉、自由中文句子**识别。训练和选参工具不读取 test；摄像头录制不属于本流程。公开说话视频中的嘴部图像可以用于研究，但不能证明刻意无声发音已经可用。本页不新增准确率或产品可用性结论。

## 数据、权重和运行环境

使用 [BAAI/Chinese-LiPS 作者数据](https://huggingface.co/datasets/BAAI/Chinese-LiPS)，固定 revision `db96948538811029011eee44602438a26710ecd9`，许可 CC-BY-NC-SA-4.0。[作者论文](https://arxiv.org/abs/2504.15066)描述参考文本来源。下载器使用嘴部视频和作者文本，不读取音频、幻灯片或 OCR；它不会重新逐字校对所有标签。

基座来自 [CNVSRC2025 官方 VSR](https://github.com/liu12366262626/CNVSRC2025/tree/main/VSR) 和[官方发布权重](https://huggingface.co/ReflectionL/CNVSRC2025Baseline)。先阅读 [VSR/LICENSE](https://github.com/liu12366262626/CNVSRC2025/blob/main/VSR/LICENSE)，仅用于允许的非商业研究比较。适配器不会自动安装到 GUI，也不会解除中文逐次确认。

研究读取器在加载时核对以下来源，随后使用 `weights_only=True` 和完整 state key/形状严格加载：

| 对象 | 固定值 |
| --- | --- |
| 作者源码 revision | `e5c4454016ba4eef9e586e77dd58e8981bb5c3e1` |
| 权重 revision | `b16f238d0df860da7e3b9834f959780b1d388f44` |
| 基座 SHA256 | `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c` |
| 配置 SHA256 | `b0464bcac797a2bafd98102d8ddaf041c6c9957b52566e5019c2a43de6860a04` |
| 字符词表 SHA256 | `635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0` |

以下命令从仓库根目录运行，假定已有包含项目依赖的 `.venv` 和本地作者文件。优先复用已下载权重；变量中的路径需要按本机情况替换。视频、运行缓存和权重放在非 Git 目录，输出目录使用新的名字。

优先使用持久的非 iCloud 路径保存源码、Python 环境、权重、视频和实验记录，例如 macOS 的 `~/Library/Caches/lipflow-research` 或 `~/.cache/lipflow-research`。iCloud 占位文件可能在读取时等待下载；`/tmp` 也可能被系统清理，均不适合唯一的断点恢复副本。开始长实验前确认数据已在本地完整落盘，冻结使用的源码并保留可恢复的备份。

使用 uv 时，可以在安装项目依赖前设置环境位置。下面从完整的本地仓库根目录运行；原有命令里的 `.venv/bin/python` 应相应替换为 `"$LIPFLOW_PYTHON"`。已有验证环境可以直接指定解释器，无需重新安装或改变正在运行的实验：

```bash
LIPFLOW_WORK_ROOT="$HOME/Library/Caches/lipflow-research"
mkdir -p "$LIPFLOW_WORK_ROOT"
export UV_PROJECT_ENVIRONMENT="$LIPFLOW_WORK_ROOT/venv"
uv sync --python 3.11
LIPFLOW_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
```


```bash
LIPFLOW_RESEARCH_CHECKPOINT=/local/model_avg_cncvs_2_3_cnvsrc.pth
LIPFLOW_RESEARCH_SOURCE=/local/CNVSRC2025
LIPFLOW_PRIOR_TEST=/local/previous-chinese-lips/test.json
LIPFLOW_DATA=/local/chinese-lips-round4
LIPFLOW_RUN=/local/chinese-lips-round4-runs
```

## 先保留新的测试说话人

此前已经查看过的 test 不能继续承担后续选参的独立验收。固定下一轮采样种子、条数和说话人数；排除旧 test 的**全部说话人**，不能按旧识别好坏选择排除。

```bash
.venv/bin/python scripts/fetch_chinese_lips.py \
  --accept-noncommercial-license \
  --output-dir "$LIPFLOW_DATA" \
  --seed lipflow-chinese-lips-round4 \
  --train-count 180 --train-speakers 12 \
  --dev-count 30 --dev-speakers 6 \
  --test-count 50 --test-speakers 10 \
  --exclude-test-manifest "$LIPFLOW_PRIOR_TEST" \
  --max-download-mb 500
```

这些条数是可复现实例，不是已执行结果。`--exclude-test-manifest` 要求旧清单来自同一固定数据版本，并核对说话人与原始 ID 对应。它只读取身份信息，记录旧清单哈希和被排除的人；不得传入预测质量排行。按种子哈希选择新成员，train/dev/test 官方说话人集合必须互不重叠。若所需独立说话人不足，工具失败，不会回退复用旧 test。

下载器通过有配额的 HTTP Range 获取指定 ZIP 成员；服务器忽略范围请求时失败，不回退下载整包。核对元数据 SHA256、成员 CRC 和本地视频 SHA256；完整 ZIP 未下载，因此不能声称验证过完整 ZIP 的 SHA256。保持新 `test.json` 封存：训练、解码网格与 epoch 选择都不使用它。查看最终测试结果后再开发，必须另设新的独立测试。

## 只排除无法训练的完整 train 样本

CNVSRC 字符词表固定。训练目标含不支持的字符时，应记录并排除**整个 train 片段**，不能删字、更换姓名、转换金额或改写参考句子使其适配词表。必须显式指定排除参数：

```bash
.venv/bin/python scripts/prepare_chinese_training.py \
  --manifest "$LIPFLOW_DATA/train.json" \
  --vocabulary "$LIPFLOW_RESEARCH_SOURCE/VSR/datamodule/char_units.txt" \
  --exclude-unsupported-training-samples \
  --output "$LIPFLOW_DATA/train-supported.json"
```

准备工具拒绝 dev/test 和已存在的输出文件。新清单保留入选样本原文、被排除的 ID/字符，以及原清单和词表哈希；原清单不改写。**所有 dev/test 参考字符和样本仍参与计分**，包括固定词表无法表示的字符。不能把训练排除规则套用到验证或测试来改善数字。

## 以 epoch 0 为基线，选择 CTC 或联合适配

[train_chinese_joint_adapter.py](../scripts/train_chinese_joint_adapter.py) 只接受 train/dev 清单。通用适配要求说话人、内容、视频身份分离；个性化模式还要求独立的真实场次。正式训练前先对完整 dev 做 epoch 0 预测，将 `baseline.json`、`protocol.json` 和 `report.json` 写出。即使训练被中断，也保留未适配基座的证据。

下面是同样参数的两组预设实验，分别执行并保留各自报告；不能根据 test 选择目标、层数、学习率或 epoch：

```bash
.venv/bin/python scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss ctc \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --output "$LIPFLOW_RUN/ctc"

.venv/bin/python scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss joint \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --output "$LIPFLOW_RUN/joint"
```

仅训练最后指定的视觉编码块及输出归一化。前端、CTC 头、前向/反向解码器和所有 BatchNorm 统计保持冻结；冻结的解码器仍允许损失对视觉编码结果反传。每步只准备当前片段，不缓存整套训练图像。

- `ctc`：完整目标的 CTC 损失按参考 token 数归一化，包含重复 token 所需的完整帧对齐检查；没有 `zero_infinity` 隐藏非法目标。
- `joint`：`0.1 × CTC + 0.9 × (0.7 × forward_attention + 0.3 × reverse_attention)`。两个注意力项使用基座的标签平滑准则，按输出 token 数（包括 EOS）归一化；反向目标只用于有标签的 **train** 教师强制训练。这个归一化是小规模适配协议，不宣称逐项数值等同于作者原训练损失。

按完整 dev 的原始 CER 选择最早的严格最优完整 epoch；还要求相对 epoch 0 的候选覆盖、整句正确率及推理失败条件不退步。CER 相同时保留先前 epoch。超时或中断不选择半个 epoch；选不到改善时保留 epoch 0，不保存可启用适配器。1800 秒是从训练开始计的**软预算**，包含后续完整 dev 评测，排除模型加载和 epoch 0；当前一步或完整 dev 评测可超过预算，报告必须保留实际用时。

合格适配器仅写出被允许参数、基座绑定和训练元数据。研究评估加载器核对完整允许参数集合、来源、形状、dtype 和有限值后才修改模型。相对 dev 改善不会自动激活权重或证明中文可用。

### 从完整 epoch 边界恢复

完成 epoch 0 后写出 `training_resume.pth`，随后每次完整训练和完整 dev 评测结束再原子更新。恢复状态包括该边界的当前可训练参数、AdamW 状态、训练样本顺序、Python/NumPy/PyTorch 随机状态及设备随机状态、已提交的选择报告和累计活动预算。最佳适配器另存为 `encoder_adapter_epochN.pth`，`encoder_adapter.pth` 原子指向已提交的最佳版本；恢复时核对其哈希，不能把崩溃前发布但尚未提交的版本当成新最佳结果。

只在原输出目录存在有效恢复状态时，给**原命令**增加 `--resume`。下面延续上面的联合适配实例，其参数不变：

```bash
.venv/bin/python scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss joint \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --resume --output "$LIPFLOW_RUN/joint"
```

恢复前核对数据清单、基座/作者源码/词表/配置、训练协议、指定设备、列明的本地实现文件和 Python/PyTorch/NumPy/OpenCV 版本；不一致时拒绝继续。参数、优化器张量与最佳适配器也需通过形状、dtype、有限值及来源检查。恢复从上次完整 epoch 的下一轮开始，不重跑 epoch 0。中断的半轮更新全部丢弃；它们的步骤数可以留在诊断报告中，但不能参加模型选择。

`active_budget.json` 在训练步骤边界和正常退出时记录累计活动用时。恢复采用恢复状态与预算台账中的较大值，保留此前已消耗的预算，不会因重启重新获得 1800 秒；程序停机期间及重新加载模型的时间不计入活动预算。强制杀进程或断电可能丢失最后一次台账更新之后的全部活动用时，包括长训练步或正在进行的完整 dev 评测；这段时间不保证很短，不能将台账视为硬退出情况下的完整计时。报告中保留这项限制。预算已耗尽时，恢复也不会再完成新的训练轮。

## 只在 dev 比较解码，再冻结完整配置

[compare_chinese_decoders.py](../scripts/compare_chinese_decoders.py) 拒绝 train、test、heldout 和混入 test 样本的清单。每段只提取一次视觉编码，在内存中复用；图像读取和解码器都得不到参考文字。不同搜索创建新的 beam/CTC scorer，清除前一次注意力投影缓存。

```bash
.venv/bin/python scripts/compare_chinese_decoders.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/dev.json" \
  --adapter "$LIPFLOW_RUN/joint/encoder_adapter.pth" \
  --ctc-weights 0.1 0.3 0.5 0.7 \
  --beam-size 40 --nbest 10 --reverse-weight 0.3 \
  --cache-max-mib 256 --pure-ctc-max-workspace-mib 1024 \
  --output "$LIPFLOW_RUN/joint-dev-decoding.json"
```

只有确实保存了该适配器时才加 `--adapter`；评估 epoch 0 时去掉它。默认预设网格为 `[0.1, 0.3, 0.5, 0.7, 1.0]`，length bonus 固定 0，无外部语言模型。上面的 16 GB 机器实例预先省略 `1.0`：纯 CTC 关闭 prebeam，会在完整词表上展开帧×假设张量，应将这一资源限制记入实验协议，不能事后按 CER 删掉较差配置。

若包含 CTC `1.0`，工具在调用解码器前检查 `6 × T × 2 × beam × vocabulary × dtype_bytes` 的工作区估算；超过显式预算时把样本标为 `resource_limited`，保存估算/预算和空结果，仍保留 CER 分母。256 MiB 是编码缓存预算，1024 MiB 是纯 CTC 工作区估算预算；二者都不是整个进程内存的上界。资源不足时使用预先固定的小开发清单或不同运行协议，并记录差异。

以开发集**第一候选的原始 CER**选基础 CTC 权重，相同时按权重升序。反向重评分只对选中的基础配置执行一次；基础和重评分结果都保留。候选体 token 倒序后教师强制评分，输入从原始 SOS 开始，目标包含原始 EOS；输入只有编码图像和候选 ID。

解码比较脚本保存所有配置，但它的最低 CER 选择本身不检查相对 `CTC=0.5、reverse=0` 的覆盖、整句正确率和失败门槛。冻结工作流必须另外应用这些条件：在同一完整 dev 上 CER 严格降低，候选覆盖和整句正确率不下降，基线与候选均无推理失败。未通过时保留原配置；不能把降低可供用户核对结果的比例作为改进。

默认 beam 最大步数等于编码帧数。ESPnet 在最后一步为剩余候选追加一个没有评分的 EOS；若最后搜索已经选择 EOS，会出现两个结尾 EOS。工具保留原始 token 序列、追加终止证据、结尾 EOS 个数和原始分数，去掉结尾 EOS 后才取得候选正文。**整个 beam 只要有一个强制终止候选，该段的全部反向重评分都跳过**，包括该候选位于保留的 n-best 之外时；原始候选顺序和分数不变。这样不会把缺失 EOS 分数的候选混入教师强制重评分尺度。

每段只计算一次画面质量和 CTC greedy 输出，随编码缓存复用。所有配置都调用与训练验证、冻结验收相同的 `confidence.assess(policy='review', language='zh')`；未知 token、低画面质量和重复输出会被标为 retry。原始识别文字仍保留并参与 CER，retry 不能通过清空或润色文字来改善错误率。中文结果继续要求人工核对。

```text
S_new = S_beam + (1 - ctc_weight) × reverse_weight × (S_reverse - S_forward)
```

`S_forward`、`S_reverse` 都是未加权的序列对数分数。公式替换前向注意力份额，保留原 beam 分数的尺度和 CTC 贡献；不做分数 softmax 概率解释，不按参考长度调整。记录原始 `beam_score` 和重评分后的 `score`。n-best oracle CER 使用参考找最接近候选，**只显示候选空间诊断，不能用来选择参数或输出给用户**。

报告每组配置的完整准备/编码时间加该配置解码时间；重评分另加反向时间。实际比较墙钟时间共享编码，因此不能将其除以配置数冒充独立端到端等待时间。模型加载和 warmup 另记。

完成 dev 选择后，单独保存冻结记录：源码 commit、数据/适配器/基座 SHA256、训练目标和 epoch、CTC 权重、reverse weight、beam、nbest、length bonus、归一化、许可及资源限制。所有这些值在第一次读取新 test 的识别结果前确定。重评分没有改善时冻结 reverse weight 为 0；没有适配改善时冻结 epoch 0。

## 冻结后对同一新 test 做 A/B/C 三臂比较

[evaluate_cnvsrc.py](../scripts/evaluate_cnvsrc.py) 的研究验收只运行传入的**单一配置**，不做网格或依据 test 重排行参。在第一次查看新 test 识别结果前冻结三臂；三臂使用同一完整 test、相同预处理、beam 40、n-best 10、length bonus 0 和无外部语言模型。

| 臂 | 模型 | 解码 | 比较用途 |
| --- | --- | --- | --- |
| A | 原始基座 | CTC 0.5，reverse 0 | 预先固定的原流程基线 |
| B | dev 选定适配器；无合格适配器时为原始基座 | 通过 dev 门槛并冻结的解码配置 | 最终候选流程 |
| C | 原始基座 | 与 B 完全相同 | 在相同解码下隔离模型适配效果 |

A→B 衡量完整流程变化，C→B 衡量同一解码设置下的模型变化。若 C 与 A 或 B 完全相同，复用同一报告，不能重复推理再择优。以下变量应填写已经冻结的值；`0.5/0.0` 只是命令示例，并非本轮选择结论。

```bash
LIPFLOW_FINAL_CTC=0.5
LIPFLOW_FINAL_REVERSE=0.0
LIPFLOW_FINAL_ADAPTER=/local/chosen-run/encoder_adapter.pth

.venv/bin/python scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight 0.5 \
  --reverse-weight 0.0 --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-A-original-test.json"

.venv/bin/python scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --adapter "$LIPFLOW_FINAL_ADAPTER" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight "$LIPFLOW_FINAL_CTC" \
  --reverse-weight "$LIPFLOW_FINAL_REVERSE" --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-B-pipeline-test.json"

.venv/bin/python scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight "$LIPFLOW_FINAL_CTC" \
  --reverse-weight "$LIPFLOW_FINAL_REVERSE" --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-C-same-decoder-base-test.json"
```

B 为 epoch 0 时去掉 `--adapter`，并复用它与 C 的相同报告；不能为了生成适配器结果而启用未通过 dev 门槛的权重。若 B 的解码与 A 相同，C 直接使用 A 报告。研究验收退出码 2 表示没有达到声明门槛，通常不代表模型崩溃。错误、空预测和词表外参考字符全部保留在 CER 分母中。

[compare_chinese_reports.py](../scripts/compare_chinese_reports.py) 只比较已保存的原始预测，不运行模型。必须传入冻结的 `test.json`，核对每段恰好覆盖一次、参考/说话人/视频哈希与场次等身份相同，避免两个报告共同漏掉样本。以下示例假定三臂各有对应文件；若复用了报告，改为实际复用的路径：

```bash
.venv/bin/python scripts/compare_chinese_reports.py \
  "$LIPFLOW_RUN/frozen-A-original-test.json" \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --seed 0 --resamples 5000 \
  --output "$LIPFLOW_RUN/frozen-A-to-B-paired.json"

.venv/bin/python scripts/compare_chinese_reports.py \
  "$LIPFLOW_RUN/frozen-C-same-decoder-base-test.json" \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --seed 0 --resamples 5000 \
  --output "$LIPFLOW_RUN/frozen-C-to-B-paired.json"
```

差值定义为 B 减去基线，负 CER 差值表示原始编辑错误减少。95% 区间以完整说话人为簇，固定 seed 0 重采样 5000 次；每次汇总字符错误数和参考字符数后计算 CER。报告包含逐说话人结果及输入报告/清单哈希。说话人很少、共享录制条件或未知预训练重叠仍限制其解释；区间不证明无声摄像头输入可用，也不能用于测试后重新选参。

可对保存的原始预测做离线错误分解；它不重跑视频，不产生纠正文案或新的模型：

```bash
.venv/bin/python scripts/analyze_chinese_errors.py \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --vocabulary "$LIPFLOW_RESEARCH_SOURCE/VSR/datamodule/char_units.txt" \
  --output "$LIPFLOW_RUN/frozen-B-pipeline-errors.json"
```

## 结果的范围

CER 按 NFKC/casefold 后的 Unicode 字母与数字计算，忽略空白、标点；金额数字仍保留，繁简汉字保持不同，不翻译、不做 LLM 润色。CER 是编辑错误率，beam 分数和候选覆盖也不是正确概率。

新测试说话人与本轮训练/开发、旧已查看测试分离，可以减少本轮选参泄漏，但基座预训练重叠仍未完整审计。`training_overlap_checked` 不能仅因哈希或本轮说话人分离就设成 true。公开视频为普通发声口型，嘴部裁剪缺少真实摄像头环境和刻意无声发音证据；即使 CER 改善，也不能解除实验状态或声称纯无声中文自由输入达到可用标准。实际本轮测量留在独立 JSON/报告中，不用命令示例替代。


## English guide

This workflow adapts an existing model for **visual-only recognition of unrestricted Mandarin sentences**. Training and decoder selection use train/dev only. Every input to the visual model is a mouth video; audio, OCR, translated English predictions, and LLM rewriting are excluded from this experiment. Public videos of normally voiced speech provide useful visual evidence, but they do not establish deliberate silent articulation or webcam dictation accuracy. This guide describes procedures and safeguards, not a new accuracy or usability result.

### Sources, licenses, and a durable local environment

The dataset is [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS), pinned to revision `db96948538811029011eee44602438a26710ecd9`, under CC-BY-NC-SA-4.0. The [author paper](https://arxiv.org/abs/2504.15066) explains how the reference text was obtained. The downloader uses the authors' labels; it does not independently verify every character in every clip.

The base visual model uses [CNVSRC2025 VSR author code](https://github.com/liu12366262626/CNVSRC2025/tree/main/VSR) and the [official checkpoint](https://huggingface.co/ReflectionL/CNVSRC2025Baseline). Read the author's [VSR/LICENSE](https://github.com/liu12366262626/CNVSRC2025/blob/main/VSR/LICENSE) before enabling the research reader. These artifacts and derived weights are restricted to the allowed noncommercial comparative research. The scripts do not install an adapter into the GUI, enable automatic Mandarin pasting, or grant deployment rights.

The research loader verifies the following identities before loading the complete checkpoint with `weights_only=True`, exact state keys, and tensor shape checks:

| Artifact | Pinned identity |
| --- | --- |
| Author source revision | `e5c4454016ba4eef9e586e77dd58e8981bb5c3e1` |
| Checkpoint revision | `b16f238d0df860da7e3b9834f959780b1d388f44` |
| Checkpoint SHA256 | `577cd9558eea111683a406bc25d69c7161cdb79534c2273fc0d0f044c356231c` |
| Configuration SHA256 | `b0464bcac797a2bafd98102d8ddaf041c6c9957b52566e5019c2a43de6860a04` |
| Character vocabulary SHA256 | `635e12ebb5f7dcd60637a4f3c329cd543f1e0e34aa4a6d62ba87185c3666aae0` |

Run commands from a fully available local repository checkout. Prefer durable storage outside iCloud for the checkout, environment, author files, datasets, and run records. An iCloud placeholder may block a read while downloading; `/tmp` can be cleaned by the operating system. Neither should hold the only recovery copy of a long experiment. Keep large weights and videos outside Git and preserve a backup of the frozen source and run artifacts.

With uv, an external environment can be selected before installing dependencies. The macOS example below uses a cache directory; Linux users can use `$HOME/.cache/lipflow-research` instead. Reuse an already validated environment when available rather than changing dependencies during a run.

```bash
LIPFLOW_WORK_ROOT="$HOME/Library/Caches/lipflow-research"
mkdir -p "$LIPFLOW_WORK_ROOT"
export UV_PROJECT_ENVIRONMENT="$LIPFLOW_WORK_ROOT/venv"
uv sync --python 3.11
LIPFLOW_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"

LIPFLOW_RESEARCH_CHECKPOINT=/local/model_avg_cncvs_2_3_cnvsrc.pth
LIPFLOW_RESEARCH_SOURCE=/local/CNVSRC2025
LIPFLOW_PRIOR_TEST=/local/previous-chinese-lips/test.json
LIPFLOW_DATA="$LIPFLOW_WORK_ROOT/chinese-lips-round4"
LIPFLOW_RUN="$LIPFLOW_WORK_ROOT/chinese-lips-round4-runs"
```

Replace `/local/...` with existing author files and the previous test manifest. If using the repository's `.venv` instead, set `LIPFLOW_PYTHON=.venv/bin/python`. Fresh experiment directories must not overwrite earlier runs.

### Reserve new test speakers before selecting anything

An inspected test set cannot remain the independent acceptance set for subsequent development. Fix the sampling seed, requested counts, and speaker counts in advance. Exclude **all** speakers from the prior test, regardless of their recognition quality. The following is a reproducible example; its counts are not a claim about the currently running experiment.

```bash
"$LIPFLOW_PYTHON" scripts/fetch_chinese_lips.py \
  --accept-noncommercial-license \
  --output-dir "$LIPFLOW_DATA" \
  --seed lipflow-chinese-lips-round4 \
  --train-count 180 --train-speakers 12 \
  --dev-count 30 --dev-speakers 6 \
  --test-count 50 --test-speakers 10 \
  --exclude-test-manifest "$LIPFLOW_PRIOR_TEST" \
  --max-download-mb 500
```

The exclusion manifest must come from the same pinned dataset revision, with speaker identities consistent with the source IDs. Exclusion uses identities, not predictions. The downloader records its hash and excluded speakers, selects members by a deterministic seed hash, and checks official train/dev/test speaker separation. Insufficient independent speakers are an error; the tool does not reuse the old test as a fallback.

HTTP Range requests retrieve selected ZIP members under an explicit quota. A server that ignores ranges causes failure instead of an automatic full-archive download. The tool checks metadata hashes, member CRCs, and local video SHA256 values; it cannot claim verification of a full ZIP SHA256 when the full archive was never downloaded. Seal the new `test.json`: neither training, epoch selection, nor decoder search may use it. Further development after inspecting its results needs a new independent test protocol.

### Keep labels intact; exclude unsupported training clips explicitly

The CNVSRC vocabulary is fixed. If a train reference contains an unsupported character, exclude and record the **entire training clip**. Do not delete characters, substitute names, alter amounts, or rewrite the reference to fit the vocabulary.

```bash
"$LIPFLOW_PYTHON" scripts/prepare_chinese_training.py \
  --manifest "$LIPFLOW_DATA/train.json" \
  --vocabulary "$LIPFLOW_RESEARCH_SOURCE/VSR/datamodule/char_units.txt" \
  --exclude-unsupported-training-samples \
  --output "$LIPFLOW_DATA/train-supported.json"
```

The preparation tool rejects dev/test and existing output files. It keeps selected references unchanged, records excluded IDs/characters and input hashes, and leaves the original manifest untouched. **Every dev/test sample and reference character remains in scoring**, including out-of-vocabulary characters. Applying a train exclusion rule to evaluation would bias the result.

### Compare CTC and joint adaptation against epoch zero

[train_chinese_joint_adapter.py](../scripts/train_chinese_joint_adapter.py) accepts train/dev only. General adaptation requires separate speakers, content, and video identities; personal adaptation also requires independent real recording sessions. Before training, it evaluates the full dev set at epoch zero and writes `baseline.json`, `protocol.json`, and `report.json`.

Run both prespecified objectives independently from the verified base, with the same schedule and separate outputs. Do not choose losses, layers, rates, or epochs using test labels.

```bash
"$LIPFLOW_PYTHON" scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss ctc \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --output "$LIPFLOW_RUN/ctc"

"$LIPFLOW_PYTHON" scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss joint \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --output "$LIPFLOW_RUN/joint"
```

Only the requested final visual encoder blocks and output normalization are trainable. The visual frontend, CTC head, forward/reverse decoder parameters, and all BatchNorm statistics remain frozen. Frozen decoders still allow gradients to reach the visual encoder. Each step prepares one clip; the whole training image set is not cached in RAM.

CTC uses the complete target, normalizes loss by reference token count, and checks the frame requirement for repeated tokens. Invalid targets are not hidden with `zero_infinity`. Joint adaptation uses:

```text
L = 0.1 * CTC + 0.9 * (0.7 * forward_attention + 0.3 * reverse_attention)
```

Attention terms use the base model's label smoothing and are normalized per output token, including EOS. Reversed reference targets are allowed only in labeled **train** teacher forcing. This adaptation normalization is explicit; it is not a claim that its scalar loss equals every term in the author's original training procedure.

Choose the earliest completed epoch with strictly lower full-dev raw CER, subject to nondecreasing candidate coverage and exact-sentence rate and no baseline/candidate inference failures. Ties keep the earlier choice. Partial epochs are never selected. If nothing passes, retain epoch zero and produce no eligible adapted artifact. Saved adapters contain only the allowed encoder parameters, base binding, and metadata; loading validates the complete allowed parameter set, source, shapes, dtypes, and finite values before changing the model.

The 1800-second limit is a **soft** active-time budget beginning at training, including subsequent full-dev evaluations but excluding initial model loading and epoch-zero evaluation. It is checked before training steps. A current step or full-dev evaluation may overrun it; report actual elapsed time and overrun rather than claiming a hard cap.

### Resume a committed complete epoch without resetting the budget

After epoch zero, `training_resume.pth` is written atomically; it is updated after a complete training epoch and full-dev evaluation. It contains the boundary's current trainable parameters, AdamW state, training order, Python/NumPy/PyTorch and device RNG states, committed selection report, and cumulative active budget. The selected adapter also has an immutable `encoder_adapter_epochN.pth` boundary file and an atomically published `encoder_adapter.pth`.

Resume only an output directory with a valid recovery state. Add `--resume` to the **same command**, retaining every protocol argument. For the joint example:

```bash
"$LIPFLOW_PYTHON" scripts/train_chinese_joint_adapter.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --train-manifest "$LIPFLOW_DATA/train-supported.json" \
  --dev-manifest "$LIPFLOW_DATA/dev.json" \
  --scope general --loss joint \
  --epochs 6 --last-layers 2 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0 --max-seconds 1800 \
  --resume --output "$LIPFLOW_RUN/joint"
```

Resume checks manifests, base/source/vocabulary/configuration identities, protocol, requested device, listed local implementation files, and Python/PyTorch/NumPy/OpenCV versions. It validates parameters and optimizer tensor shapes/dtypes/finite values and the selected adapter hash. Mismatches are rejected. It continues at the next complete epoch without rerunning epoch zero. Partial-epoch parameter updates are discarded; diagnostic step counts do not make them selectable. A newly published but uncommitted adapter from an interrupted transaction does not replace the committed choice.

`active_budget.json` is updated at training-step boundaries and normal exit. Resume carries the larger elapsed value from the recovery state and budget ledger. Restarting does not grant another 1800 seconds; idle downtime and reload time are excluded. **A hard kill or power loss may omit all active time since the last ledger update, including a long step or an ongoing full-dev evaluation. That interval is not guaranteed to be short.** The ledger is therefore not a complete hard-exit timer. An exhausted recorded budget permits no further completed training epoch.

### Select decoding on dev and freeze the full configuration

[compare_chinese_decoders.py](../scripts/compare_chinese_decoders.py) rejects train, test, heldout, and test rows hidden in a dev manifest. Each clip is visually encoded once and cached in memory. Neither the frame reader nor the decoders receive reference text. Each serial search creates fresh beam/CTC scorers and clears encoder-memory attention projections.

```bash
"$LIPFLOW_PYTHON" scripts/compare_chinese_decoders.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/dev.json" \
  --adapter "$LIPFLOW_RUN/joint/encoder_adapter.pth" \
  --ctc-weights 0.1 0.3 0.5 0.7 \
  --beam-size 40 --nbest 10 --reverse-weight 0.3 \
  --cache-max-mib 256 --pure-ctc-max-workspace-mib 1024 \
  --output "$LIPFLOW_RUN/joint-dev-decoding.json"
```

Include `--adapter` only when that run actually saved an eligible artifact; remove it for epoch zero. The default grid is `[0.1, 0.3, 0.5, 0.7, 1.0]`. The 16 GB example above prespecifies omission of `1.0`: pure CTC disables prebeam and expands full-vocabulary prefix workspace. Record this resource decision before inference; do not remove a poorly scoring configuration afterward.

When CTC `1.0` is included, the estimated workspace is `6 * encoded_frames * 2 * beam_size * vocabulary_size * dtype_bytes`. Above the explicit budget, the decoder is not invoked: the row records `resource_limited`, its estimate/budget, and a blank output while remaining in every denominator. The 256 MiB encoding cache and 1024 MiB workspace estimate are separate budgets, not a bound on total process memory. A smaller prespecified dev set or changed resource protocol must be recorded explicitly.

The helper selects the base CTC weight by lowest **top-1 raw dev CER**, breaking ties by ascending weight, and evaluates reverse rescoring only for that base. It saves all configurations. Its minimum-CER choice does **not** itself enforce coverage/exact/failure gates relative to CTC `0.5`, reverse `0`; the freezing workflow must enforce them separately: strictly lower full-dev CER, no coverage or exact-sentence decline, and no baseline/candidate inference failures. Otherwise retain the original configuration.

Reverse teacher forcing uses reversed **candidate body token IDs**, then the original EOS, with the original SOS at the input start. Only encoded images and candidates enter this computation. The score is:

```text
S_new = S_beam + (1 - ctc_weight) * reverse_weight * (S_reverse - S_forward)
```

Both attention terms are unweighted sequence log scores. This replaces a forward attention share while retaining beam score scale and CTC contribution; no score softmax is interpreted as correctness probability, and reference length is not used. Keep original `beam_score` and rescored `score`. Reference-based n-best oracle CER is a **diagnostic of the candidate space only**, never a selection rule or an output available to a real user.

At the default maximum of one decoding step per encoded frame, ESPnet appends an unscored EOS in the final loop. A hypothesis can consequently have two trailing EOS tokens. The helper strips trailing EOS from rendered text/body while retaining the original token sequence, termination evidence, trailing-EOS count, and scores. If **any hypothesis in the full returned beam** is force-ended, reverse rescoring is skipped for the entire utterance, even when that hypothesis is beyond retained n-best. Original scores and ordering remain unchanged; forced endings are not mixed with teacher-forced EOS scores.

Quality and CTC greedy text are calculated once per clip and reused. All configurations use the shared `confidence.assess(policy='review', language='zh')`: unknown tokens, poor visual quality, or repetitive output can produce `retry`. Raw retry predictions still contribute to CER; clearing or polishing them cannot improve the metric. Mandarin continues to require user review.

Each configuration's reported latency includes the full preparation/encoding cost plus its own decoding, with extra reverse time where applicable. Actual comparison wall time shares encoding; dividing it by configuration count is not an independent end-to-end latency measurement. Startup/warmup are separate.

Before reading any new test predictions, freeze source commit, data/adapter/base hashes, objective and epoch, CTC/reverse weights, beam, n-best, length bonus, normalization, license, and resource restrictions. Keep reverse weight zero if rescoring fails the dev gate and epoch zero if no adaptation passes.

### Evaluate the frozen A/B/C arms on the same new test

[evaluate_cnvsrc.py](../scripts/evaluate_cnvsrc.py) runs a single supplied configuration. It performs no grid search or parameter selection from test results. All arms use the identical complete test manifest and preprocessing, beam 40, n-best 10, length bonus zero, and no external LM.

| Arm | Visual model | Decoder | Purpose |
| --- | --- | --- | --- |
| A | Original base | CTC 0.5, reverse 0 | Prespecified original pipeline |
| B | Dev-selected adapter; original base if none passes | Dev-selected and gated frozen decoder | Final candidate pipeline |
| C | Original base | Exactly the decoder used by B | Isolate model adaptation under equal decoding |

A→B measures the total pipeline change. C→B isolates model change with identical decoding. If C equals A or B, reuse the corresponding report rather than rerunning and selecting a favorable result. The values below are **examples**, not a current selection; replace them with the frozen values.

```bash
LIPFLOW_FINAL_CTC=0.5
LIPFLOW_FINAL_REVERSE=0.0
LIPFLOW_FINAL_ADAPTER=/local/chosen-run/encoder_adapter.pth

"$LIPFLOW_PYTHON" scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight 0.5 \
  --reverse-weight 0.0 --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-A-original-test.json"

"$LIPFLOW_PYTHON" scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --adapter "$LIPFLOW_FINAL_ADAPTER" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight "$LIPFLOW_FINAL_CTC" \
  --reverse-weight "$LIPFLOW_FINAL_REVERSE" --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-B-pipeline-test.json"

"$LIPFLOW_PYTHON" scripts/evaluate_cnvsrc.py \
  --accept-research-license \
  --checkpoint "$LIPFLOW_RESEARCH_CHECKPOINT" \
  --source-dir "$LIPFLOW_RESEARCH_SOURCE" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --beam-size 40 --ctc-weight "$LIPFLOW_FINAL_CTC" \
  --reverse-weight "$LIPFLOW_FINAL_REVERSE" --nbest 10 \
  --output "$LIPFLOW_RUN/frozen-C-same-decoder-base-test.json"
```

For epoch-zero B, remove `--adapter` and reuse its report for C. When B's decoder matches A, reuse A for C. Never enable an adapter that failed dev gates just to produce an adapted test result. Exit code 2 means the declared research acceptance thresholds were not met; it usually does not mean the model crashed. Failed, empty, and out-of-vocabulary cases all remain in CER denominators.

### Compare stored reports with the frozen manifest

[compare_chinese_reports.py](../scripts/compare_chinese_reports.py) performs no inference. Pass the frozen test manifest so it can verify that every intended sample appears exactly once, with matching references, speakers, video hashes, sessions, and other declared identities. Comparing two equally incomplete reports without the manifest would not establish coverage of the intended test.

```bash
"$LIPFLOW_PYTHON" scripts/compare_chinese_reports.py \
  "$LIPFLOW_RUN/frozen-A-original-test.json" \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --seed 0 --resamples 5000 \
  --output "$LIPFLOW_RUN/frozen-A-to-B-paired.json"

"$LIPFLOW_PYTHON" scripts/compare_chinese_reports.py \
  "$LIPFLOW_RUN/frozen-C-same-decoder-base-test.json" \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --seed 0 --resamples 5000 \
  --output "$LIPFLOW_RUN/frozen-C-to-B-paired.json"
```

If an arm was reused, supply the actual reused path. Deltas are B minus the baseline; negative CER differences mean fewer raw edit errors. The 95% interval resamples whole speakers 5,000 times with seed zero, pooling character counts before calculating CER each time. Reports preserve per-speaker outcomes and hashes of both reports and the manifest. Few speakers, common recording conditions, or unknown pretraining overlap limit the interval's interpretation. It does not establish silent-webcam usability or permit test-driven reselection.

For an offline breakdown of stored errors, without rerunning videos or generating corrected text:

```bash
"$LIPFLOW_PYTHON" scripts/analyze_chinese_errors.py \
  "$LIPFLOW_RUN/frozen-B-pipeline-test.json" \
  --manifest "$LIPFLOW_DATA/test.json" \
  --vocabulary "$LIPFLOW_RESEARCH_SOURCE/VSR/datamodule/char_units.txt" \
  --output "$LIPFLOW_RUN/frozen-B-pipeline-errors.json"
```

### Interpret results within their measured domain

Raw CER uses NFKC/casefold Unicode letters and numbers, ignoring spaces and punctuation. Digits in monetary amounts remain; traditional and simplified Han characters remain distinct. No translation or LLM cleanup is applied. CER is an edit error rate; beam scores, candidate coverage, and decoder agreement are not calibrated probabilities.

New test speakers are separate from this adaptation's train/dev and the previously inspected test. This reduces current selection leakage, but base-model pretraining overlap has not been fully audited. Do not set `training_overlap_checked=true` merely because local hashes and adaptation split separation were verified. Ordinary voiced mouth crops lack evidence for deliberate silent articulation and real webcam conditions. Even a statistically improved result cannot lift experimental status or establish usable unrestricted silent Mandarin input. Record actual measurements in separate benchmark reports rather than substituting example commands for evidence.
