# 中文纯无声自由句子的验收 / Acceptance evaluation for silent Mandarin sentences

[English guide](#english-guide)

协议设定日期：2026-10-02。以下数值是验收目标，不是已取得的识别结果；最新实测状态见[验证记录](VALIDATION.md)。普通发声公开数据的独立研究流程见[中英公开数据操作说明](chinese-public-adaptation.md)。

本轮目标是摄像头直接识别普通话自由句子。验收只衡量视觉模型原始输出，不读取音频，不调用文本纠错，不把参考文本、说话者姓名或标签交给模型。参考文本仅在预测完成后用于计分。

## 数据要求

先冻结模型、解码参数及阈值，再采集独立测试数据。默认至少 50 条句子、5 位说话者，每人至少两个独立录制场次，例如不同日期。使用实际摄像头的完整人脸画面，涵盖光线、距离及真实不出声的嘴部动作。普通发声视频去掉声音后只能检验视觉信息，不足以代表刻意无声说话的动作。

每段视频需要独立人工核对的参考文本及标签来源。训练和调参片段列入 `development_samples`，不得重复作为独立测试。工具检查文件 SHA-256、重叠视频范围、重复 ID 和已声明的训练/调参说话者。身份、场次及人工标注声明仍需真实填写；程序不能替代人工审计。

这是对通用模型的初步验收。单用户个性化模型需要另建该用户不同日期的训练/测试协议，不能把训练中的 24 条练习片段重新计分来宣布可用。

## 本地标注清单

JSON 路径相对于清单文件。以下是结构示例，文件名、标签和场次需替换为真实记录，并补齐所有测试样本：

```json
{
  "schema_version": 1,
  "split": "test",
  "training_overlap_checked": true,
  "description": "独立的纯无声摄像头测试",
  "development_samples": [],
  "samples": [
    {
      "id": "speaker01-day02-001",
      "video": "videos/speaker01-day02-001.mp4",
      "reference": "人工核对后填写实际说的句子",
      "speaker": "speaker01",
      "session": "day02",
      "label_source": "人工标注记录：labels/day02.csv 第1行",
      "label_verified": true,
      "domain": "webcam",
      "mouth_roi": false,
      "articulation": "silent",
      "articulation_verified": true
    }
  ]
}
```

`training_overlap_checked` 只有在实际审计完成后才能设为 true。`domain: mouth_roi` 对应已经对齐的 96×96 嘴部裁剪片段；这种数据可用于诊断，但默认不能通过摄像头验收。公开诊断样本应声明 `split: diagnostic`，不能在看过结果、调整模型后重新声称是独立测试。

`articulation` 区分真正无声发音、普通说话（`voiced`）、耳语（`whispered`）和未知（`unknown`）。关闭视频音轨后测试，不能把普通说话改记为无声发音。默认验收要求全部片段都是人工确认的 `silent`；缺少声明、普通说话和耳语都会使这一条件失败。研究普通说话语料可明确传入 `--allow-voiced-articulation`，报告会记录放宽的门槛；这不能证明无声输入可用。

稍后录制真实无声片段时，使用 `lipflow collect-chinese`，见[公开数据适配与本地采集](CHINESE_ADAPTATION.md)。采集工具默认不声称完成训练重叠审计。

运行命令：

```bash
uv run python scripts/evaluate_chinese.py --manifest eval/manifest.json \
  --beam-size 20 --output eval/results.json
```

程序只加载一次模型并先做独立预热。热运行延迟包含视频读取、人脸/嘴部处理和解码，排除首次下载及模型加载时间；报告中同时保留加载和预热开销。样本处理失败会计入错误及失败率，不会从分母移除。缺失文件或参考文本会直接报清单错误。

## 本轮默认门槛

| 项目 | 初步验收目标 |
| --- | --- |
| 测试数据 | 至少 50 段、5 人、每人两个独立场次；真实摄像头；与训练/调参独立 |
| 原始字符错误率（CER） | 不高于 10% |
| 整句完全正确率 | 不低于 70% |
| 非拒绝覆盖率 | 不低于 80%，不能靠拒绝全部输入达到低错误率 |
| 热运行 p95 等待时间 | 不高于 2 秒 |
| 热运行 p95 实时因子 | 不高于 1，即处理不慢于片段自身长度 |

这些是本轮明确设定的工程目标，可以在测试开始前通过参数改变；不是医学或行业标准，也不是所有用户的可用性保证。改变门槛必须在报告里体现。CER 忽略标点与空白，保留汉字、拉丁字母和数字；繁简不同仍算错误，空输出按删除全部字符计分，插入过多字符时 CER 可以超过 100%。

“非拒绝”仅表示程序提供候选，不表示候选正确。自动放行覆盖率与被放行结果的 CER 单独统计；当前中文锁定确认，因此自动放行率为零。门槛报告不会自动解除产品中的确认要求。

只有全部数据条件和质量条件通过时，报告才返回 `ready`；否则列出每条 `not_ready` 原因。公开演示片段、高分候选、通过单元测试，均不能替代这一验收。


## English guide

Protocol set on **2026-10-02**. The numbers below are engineering acceptance targets, not achieved recognition results. See [validation records](VALIDATION.md) for measured outcomes and the [public-data research guide](chinese-public-adaptation.md) for a separate voiced-corpus experiment.

The target is **visual-only recognition of unrestricted silent Mandarin sentences from a webcam**. Acceptance scores raw visual-model output. It reads no audio, invokes no text cleanup, and does not provide reference text, speaker names, or labels to the model. References are used only after prediction for scoring.

### Collect independent data after freezing the candidate

Freeze the model, decoder, and thresholds before collecting acceptance data. Defaults require at least 50 sentences from 5 speakers, with at least two genuinely independent sessions per speaker, such as separate dates. Use real full-face webcam video spanning lighting, distance, and deliberate silent articulation. Removing an audio track from ordinary voiced speech only tests visual information; it does not reproduce deliberately silent mouth movements.

Every clip needs an independently human-verified reference and label source. Declare training/tuning clips under `development_samples`; they cannot also be independent test clips. The tool checks SHA256 identities, overlapping video ranges, duplicate IDs, and declared training/tuning speakers. Speaker/session/annotation declarations must be truthful; software cannot perform that human audit for you.

This is an initial general-model protocol. Personalization for one user requires a separate across-date training/test design. Rescoring that user's 24 training practice clips does not establish usability.

### Create a local annotation manifest

Video paths are relative to the manifest. Replace filenames, labels, and sessions with real records and include every intended test sample. This is a structural example, not a complete acceptance set:

```json
{
  "schema_version": 1,
  "split": "test",
  "training_overlap_checked": true,
  "description": "Independent silent-webcam acceptance set",
  "development_samples": [],
  "samples": [
    {
      "id": "speaker01-day02-001",
      "video": "videos/speaker01-day02-001.mp4",
      "reference": "人工核对后填写实际说的句子",
      "speaker": "speaker01",
      "session": "day02",
      "label_source": "Human annotation: labels/day02.csv row 1",
      "label_verified": true,
      "domain": "webcam",
      "mouth_roi": false,
      "articulation": "silent",
      "articulation_verified": true
    }
  ]
}
```

Set `training_overlap_checked=true` only after a real overlap audit. `domain: mouth_roi` denotes an already aligned 96×96 mouth crop, which can support diagnostics but fails the default webcam condition. Public diagnostic clips should declare `split: diagnostic`; after inspecting results or adjusting the model, do not relabel them as an independent test.

`articulation` distinguishes deliberate `silent`, normal `voiced`, `whispered`, and `unknown` speech. Disabling audio does not turn `voiced` into `silent`. Default acceptance requires human-verified silent articulation for every clip. Missing declarations, voiced clips, and whispers fail that condition. Research on ordinary voiced data may explicitly use `--allow-voiced-articulation`; the report records relaxed thresholds and cannot establish silent-input usability.

Optional genuine silent recording uses `lipflow collect-chinese`, described in [the capture guide](CHINESE_ADAPTATION.md). The collector does not claim that a training-overlap audit has been completed.

### Run the frozen evaluation

```bash
uv run python scripts/evaluate_chinese.py --manifest eval/manifest.json \
  --beam-size 20 --output eval/results.json
```

The program loads the model once and performs separate warmup. Warm processing latency includes video reading, face/mouth preprocessing, and decoding; initial download/model loading are excluded and reported separately with warmup costs. Processing failures remain in error and failure-rate denominators. Missing files or references cause a manifest error instead of quietly removing rows.

### Apply every data and quality threshold

| Requirement | Initial target |
| --- | --- |
| Test data | At least 50 clips, 5 speakers, two independent sessions per speaker; real webcam; separate from training/tuning |
| Raw character error rate | CER ≤10% |
| Exact-sentence rate | ≥70% |
| Non-rejected candidate coverage | ≥80%; rejecting all inputs cannot satisfy usability |
| Warm p95 processing latency | ≤2 seconds |
| Warm p95 real-time factor | ≤1; processing should not exceed clip duration |

These are declared engineering targets, not medical/industry standards or guarantees for every user. Parameters may change them **before** testing, and the report must record the change. CER ignores punctuation and whitespace while retaining Han characters, Latin letters, and digits. Traditional/simplified differences count as errors. Empty output deletes the entire reference; enough insertions can make CER exceed 100%.

“Non-rejected” means a candidate was offered, not that it was correct. Automatic coverage and CER among automatically released results are reported separately. Mandarin is locked to confirmation, so automatic coverage is zero. A report does not remove that product requirement.

Only a full pass of every data and quality condition returns `ready`; otherwise the report lists each `not_ready` reason. Public demonstration clips, high-scoring candidates, passing unit tests, and improvement on normally voiced public videos cannot replace this independent acceptance test.
