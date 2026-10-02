# 公开数据适配与本地无声采集 / Public-data adaptation and optional silent capture

[English guide](#english-guide)

历史说明：本页固定训练实验记录于 2026-10-02，保留第三轮两轮 CTC 实验的原参数和命令以便复现，不是继续调参的推荐协议。后续研究使用[中英公开数据操作说明](chinese-public-adaptation.md)中的 train/dev-only 选择、完整 epoch 恢复、新测试说话人与冻结 A/B/C 比较；最新结果以[验证记录](VALIDATION.md)为准。下方摄像头采集为可选功能，公开数据研究不依赖本次个人录制。

目标是纯无声、自由中文句子。**目前仍未达到日常输入的可用标准。** 本流程分别记录公开数据研究与真实无声摄像头验收，训练报告不会自动替换 GUI 模型或解除中文候选确认。

## 公开数据

使用作者发布的 [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS)，固定版本 `db96948538811029011eee44602438a26710ecd9`，许可为 CC-BY-NC-SA-4.0。原始参考文本来自作者人工整理；下载工具不会重新核对每个字，也不会生成标签。数据是普通说话的嘴部裁剪，不能证明刻意不发声的口型识别能力。[作者论文](https://arxiv.org/abs/2504.15066)说明了数据来源和划分。

```bash
uv run python scripts/fetch_chinese_lips.py --accept-noncommercial-license \
  --output-dir samples/chinese_lips
```

默认按固定哈希规则选取 train 180 条/12 人、dev 30 条/6 人、test 50 条/10 人。选择不看识别结果。工具只读取嘴部视频与文本元数据，不读取音频或幻灯片。通过 HTTP Range 读取指定压缩成员；服务器忽略范围请求时直接失败，不回退下载整个多 GB 压缩包。元数据 SHA256、成员 CRC 和本地视频 SHA256 会核对；整个压缩包的发布者 SHA256 仅记录，未下载整包，因此不声称已验证整包。

`samples/` 已由 Git 忽略，视频和参考标签不随应用分发。三分区是本轮适配的独立分区；尚未完整核实基座预训练的身份/视频重叠，`training_overlap_checked` 保持 false。

## 固定训练实验（2026-10-02 第三轮历史复现）

单独下载并阅读 [CNVSRC2025 源码许可](https://github.com/liu12366262626/CNVSRC2025/blob/main/VSR/LICENSE)和[基座权重](https://huggingface.co/ReflectionL/CNVSRC2025Baseline)。源码版本、配置、词表和权重哈希由研究脚本核对。实验使用现有 ESPnet 模块实现兼容结构，不导入作者源码、不将研究权重装入产品。

CNVSRC 使用固定字符词表。先列出无法表示的训练目标；明确选择排除这些**完整训练片段**，保留原始文本和排除记录。这个操作只适用于 train，dev/test 中不支持的字符仍须保留在完整计分中：

```bash
uv run python scripts/prepare_chinese_training.py \
  --manifest samples/chinese_lips/train.json \
  --vocabulary /local/CNVSRC2025/VSR/datamodule/char_units.txt \
  --exclude-unsupported-training-samples \
  --output samples/chinese_lips/train-supported.json

uv run python scripts/train_chinese_adapter.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --train-manifest samples/chinese_lips/train-supported.json \
  --dev-manifest samples/chinese_lips/dev.json \
  --test-manifest samples/chinese_lips/test.json \
  --scope general --output samples/chinese_lips/adapter-run \
  --epochs 2 --last-layers 1 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0
```

本轮默认划分保留 160 条可表示训练片段。训练只改最后一个视觉编码块和输出归一化，冻结前端、CTC 头、解码器与 BatchNorm 统计。损失是完整字符目标的 CTC，MPS 编码器与 CPU CTC 之间保留梯度。过短目标帧数或非有限损失会停止实验，不会悄悄跳过。推理模式产生的位置编码缓存会在训练前转换为可反传张量。

实验固定两个 epoch，不按验证/测试分数选择 epoch。基线和最终候选分别在完整 dev/test 各计分一次。只有两套数据的原始 CER 都降低、整句正确率和候选覆盖率不下降且无推理失败，才保存 `encoder_adapter.pth`；否则只保存拒绝报告。相对改善仍不等于产品可用。报告保留基座、数据、排除记录与参数，部分权重保留基座研究用途限制。

观察过测试结果后，若更改参数、增强、模型或解码方式，需要新的独立测试集。不要重复使用同一测试集挑选最佳模型。

保存的部分权重可以在独立研究脚本中显式加载；基座 SHA256、语言、允许参数、形状、类型和有限值均会先核对：

```bash
uv run python scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --adapter samples/chinese_lips/adapter-run/encoder_adapter.pth \
  --manifest /local/new-independent-test/manifest.json \
  --output samples/chinese_lips/new-test-results.json
```

## 稍后采集真正无声的摄像头片段

采集不需要加载识别权重。只在你主动执行以下命令时打开摄像头；没有麦克风或云调用：

```bash
uv run lipflow collect-chinese --output samples/chinese_private/train \
  --speaker bryce --session day01 --split train
```

录制前，在终端输入你准备无声说的准确句子。按 SPACE 开始/停止；R 丢弃并重录；Esc 丢弃当前片段并退出。停止后，只有当你确实无声发音、没有耳语/发声且说的文字与提示一致时才按 Enter 保存。说错或漏字请重录。也可用 `--sentences phrases.txt` 提供独立编写的 UTF-8 每行一句文本。不要采用模型识别结果作为参考答案。

预览提示绘制在副本上，保存的是未加文字的原始摄像头画面。文件按实际采集时间戳流式重采样为 25 fps；记录人脸、光线、口型运动和时间戳用于后续审计。单段默认最多 15 秒，输出目录默认配额 512 MiB，可用参数调整。质量检查不可用时会明确记录未知质量。未确认片段在正常退出或重录时删除。

`speaker` 必须是同一人的一致标识，`session` 表示一次真实录制场次，不能每条视频虚构一个新场次。train/dev/test 用独立目录，个性化模型使用同一人在不同日期的片段；通用模型使用不同说话人。test 句子不得来自训练或调参。默认清单不声称完成重叠审计。

采集工具的合成测试已通过；真实 macOS 摄像头权限、录制操作和 Windows 界面仍需人工验证。你稍后录制的实际片段才可用于验证这个目标。批量验收见[中文纯无声验收流程](CHINESE_EVALUATION.md)，本轮结果见[验证记录](VALIDATION.md)。


## English guide

Historical snapshot: **2026-10-02, third research round**. This page retains the original fixed two-epoch CTC pilot and commands for reproduction. It is not the recommended protocol for further model selection after inspecting that pilot's test results. Use the [current bilingual public-data workflow](chinese-public-adaptation.md) for train/dev-only selection, complete-epoch recovery, new test speakers, and frozen A/B/C evaluation. [Validation records](VALIDATION.md) contain measured outcomes. Local capture is optional; the public-data experiment does not depend on a personal webcam recording.

The goal is unrestricted, deliberately silent Mandarin sentences. The historical pilot did not establish daily-input usability. Public-data research and real silent-webcam acceptance are recorded separately, and no training report automatically replaces the GUI model or removes candidate review.

### Obtain the pinned public data

Use [BAAI/Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS), revision `db96948538811029011eee44602438a26710ecd9`, under CC-BY-NC-SA-4.0. References come from the author's annotation process; the downloader neither regenerates labels nor re-verifies every character. These are normally voiced mouth crops and do not establish silent-articulation performance. The [author paper](https://arxiv.org/abs/2504.15066) explains sources and splits.

```bash
uv run python scripts/fetch_chinese_lips.py --accept-noncommercial-license \
  --output-dir samples/chinese_lips
```

The defaults deterministically sample train 180 clips/12 speakers, dev 30/6, and test 50/10. Sampling ignores recognition results. The tool reads only mouth videos and text metadata, not audio or slides. HTTP Range retrieves selected archive members; a server that ignores ranges fails rather than downloading a multi-GB archive. Metadata SHA256, member CRC, and local video SHA256 are checked. Publisher full-archive SHA256 values are recorded, but cannot be claimed verified when the full archive was not downloaded.

`samples/` is ignored by Git; video and reference-label files are not distributed with the application. The three splits are independent for this adaptation experiment, but base-model pretraining overlap has not been fully audited. Keep `training_overlap_checked=false`. For durable storage and an external `UV_PROJECT_ENVIRONMENT`, follow the current research guide and use non-iCloud paths rather than relying on `/tmp`.

### Reproduce the fixed historical CTC pilot

Read the [CNVSRC2025 source license](https://github.com/liu12366262626/CNVSRC2025/blob/main/VSR/LICENSE) and obtain the [base checkpoint](https://huggingface.co/ReflectionL/CNVSRC2025Baseline) separately. Research scripts verify source revision and configuration/vocabulary/checkpoint hashes. They construct a compatible architecture using existing ESPnet modules; they do not import author code or install the research checkpoint into the product.

For the fixed character vocabulary, explicitly exclude **whole unsupported train clips**, keeping original labels and an exclusion audit. Unsupported dev/test characters remain in complete scoring.

```bash
uv run python scripts/prepare_chinese_training.py \
  --manifest samples/chinese_lips/train.json \
  --vocabulary /local/CNVSRC2025/VSR/datamodule/char_units.txt \
  --exclude-unsupported-training-samples \
  --output samples/chinese_lips/train-supported.json

uv run python scripts/train_chinese_adapter.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --train-manifest samples/chinese_lips/train-supported.json \
  --dev-manifest samples/chinese_lips/dev.json \
  --test-manifest samples/chinese_lips/test.json \
  --scope general --output samples/chinese_lips/adapter-run \
  --epochs 2 --last-layers 1 --learning-rate 0.00001 \
  --beam-size 40 --ctc-weight 0.5 --seed 0
```

The historical default split retained 160 representable training clips. Only the final visual encoder block and output normalization were trainable; the frontend, CTC head, decoder, and BatchNorm statistics stayed frozen. Full-target CTC retained gradients between the MPS encoder and CPU CTC. Insufficient frames for target alignment or nonfinite loss stopped the experiment rather than silently skipping a sample. Inference-created positional caches were converted into tensors suitable for backpropagation.

This pilot fixed two epochs without selecting an epoch using dev/test scores. It scored the base and final candidate once on complete dev and test. It saved an adapter only when raw CER improved on both, exact-sentence rate and candidate coverage did not decline, and neither had inference failures. Otherwise it retained a rejection report. That historical post-run comparison is not permission to repeatedly tune on test: **use the newer train/dev-only workflow and a new independent test for any subsequent training or decoder change**.

Reports retain base/data/exclusion/protocol evidence. Derived tensors keep the base research restrictions. Relative improvement alone is not usability. Explicit research loading checks base SHA256, language, allowed parameters, shapes, types, and finite values before modification:

```bash
uv run python scripts/evaluate_cnvsrc.py --accept-research-license \
  --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth \
  --source-dir /local/CNVSRC2025 \
  --adapter samples/chinese_lips/adapter-run/encoder_adapter.pth \
  --manifest /local/new-independent-test/manifest.json \
  --output samples/chinese_lips/new-test-results.json
```

### Optionally capture genuine silent webcam clips

Recording needs no recognition weights. The camera opens only when the user explicitly runs the collector; it makes no microphone or cloud call.

```bash
uv run lipflow collect-chinese --output samples/chinese_private/train \
  --speaker bryce --session day01 --split train
```

Before recording, type the exact intended silent sentence in the terminal. SPACE starts/stops, R discards and repeats, and Esc discards the current clip and exits. After stopping, press Enter only if articulation was genuinely silent, with no whisper/voice, and the spoken text matched the prompt. Rerecord missed or changed words. `--sentences phrases.txt` can provide independently authored UTF-8 prompts, one sentence per line. Model predictions must not become reference answers.

Preview overlays are drawn on a copy; saved video contains the original unannotated webcam frames. Frames are streamed and resampled to 25 fps using real capture timestamps. Face visibility, light, mouth motion, and timestamps are recorded for audit. Defaults limit a clip to 15 seconds and an output directory to 512 MiB, with adjustable parameters. Unavailable quality checks are recorded as unknown. Unconfirmed clips are deleted on normal cancellation or rerecording.

Use one consistent `speaker` identity for a person. A `session` is a real recording occasion, not a fabricated new session per clip. Separate train/dev/test directories; personalization uses the same person across independent dates, while general-model tests use separate speakers. Test sentences must not come from training/tuning. Collector manifests do not claim overlap has been audited.

As of this dated snapshot, synthetic collector tests had passed, while real macOS permissions/recording interaction and Windows UI still required manual validation. Those software tests do not prove webcam capture success or silent recognition accuracy. Actual independently labeled silent clips are needed for that assessment. See the [silent acceptance protocol](CHINESE_EVALUATION.md) and [validation records](VALIDATION.md); capture remains optional for the public-data research described here.
