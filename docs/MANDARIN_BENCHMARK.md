# 中文唇语识别第四轮实测 / Mandarin lipreading: iteration-four measurements

## 中文说明

本轮增加可恢复的编码器训练、开发集解码比较和测试前冻结流程。在 Apple M4 / 16 GiB 上执行，仅使用视频画面，不读取音轨、不使用 LLM 纠错或参考文本提示。应用中的 CMLR 中文入口与本轮 CNVSRC 研究模型分别运行；这些数字衡量研究流程，不是 GUI 准确率。

Chinese-LiPS 固定 revision `db96948538811029011eee44602438a26710ecd9`，按说话者分成 1200 条训练、100 条开发及100条新测试。训练95条整段 OOV 样本明确排除，保留1105条原始标签；开发和测试不排除 OOV、失败、拒绝或空输出。新测试的10名说话者与训练、开发及此前已查看测试的说话者隔离。基础模型预训练是否与公开语料重叠仍未知。

### 训练和开发集选择

| Objective | Completed epochs | Selected epoch | Baseline dev CER | Selected dev CER |
| --- | --- | --- | --- | --- |
| ctc | 2 | 2 | 66.34% | 60.84% |
| joint | 2 | 2 | 66.34% | 60.82% |

两种方法分别从同一原始权重开始：最多6轮、每种1800秒软预算（含轮间开发集评测，不含加载及基线）、末2个编码器块、学习率1e-5、seed 0，其余层冻结。仅完整轮次可以被选择。CTC运行曾位于有 I/O 延迟的 iCloud 路径，联合训练改用本机缓存；预算内完成轮次数受到存储吞吐影响，因此不能把不同轮次的差距全部归因于损失函数。

训练阶段按开发集选出的候选为 `joint` 第 `2` 轮。最终冻结的 B 实际使用 `joint` 第 `2` 轮，流程来源为 `development_comparison`。最终冻结的 B 使用通过开发集门槛的适配候选。 冻结解码为 CTC 权重 `0.1`、反向权重 `0.3`、beam 40、nbest 10，无外部 LM。解码比较固定为 CTC 0.1/0.3/0.5/0.7 及其中原始 CER 最优设置的一次反向 0.3 重排。纯 CTC 权重 1 在推理前因 16 GiB 内存限制排除。分数不是正确概率，反向重排不使用参考文本。

### 开发集解码比较（全部五组）

下表来自冻结前的完整开发集，不是新测试集结果。以原始第一候选 CER 为主准则，冻结工作流还要求相对原始基座具有相同数据分母、无推理失败、整句正确率和候选覆盖率不下降；仅 CER 降低不能绕过这些门槛。

| CTC | Reverse | Errors / reference characters | Dev CER | Exact sentences / rate | Candidate coverage |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.0 | 2467 / 4147 | 59.49% | 1/100 (1.00%) | 100.00% |
| 0.3 | 0.0 | 2480 / 4147 | 59.80% | 0/100 (0.00%) | 100.00% |
| 0.5 | 0.0 | 2522 / 4147 | 60.82% | 0/100 (0.00%) | 99.00% |
| 0.7 | 0.0 | 2549 / 4147 | 61.47% | 0/100 (0.00%) | 100.00% |
| 0.1 | 0.3 | 2452 / 4147 | 59.13% | 0/100 (0.00%) | 100.00% |

无反向配置的最高整句正确数为 1/100；反向 0.3 配置为 0/100。 全部配置均列出，不能用较低 CER 掩盖整句正确率或覆盖率退步。

### 冻结后的新测试

A是原始权重、CTC0.5/反向0；B是开发集选出的固定流程；C是原始权重配合B完全相同的解码参数。所有100条数据都计分。相同权重与参数的组直接复用同一预测，并明确记录，不重复推理或伪造独立延迟。

| Arm | Errors / characters | CER | Exact sentences | Candidate coverage | Failures | Warm p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 3037 / 4087 | 74.31% | 0/100 | 88.00% | 0 | 5.85 s |
| B | 2696 / 4087 | 65.97% | 0/100 | 90.00% | 0 | 6.15 s |
| C | 3170 / 4087 | 77.56% | 0/100 | 87.00% | 0 | 6.13 s |

- A → B overall / 整体流程: ΔCER -8.34 pp; 95% speaker-cluster interval [-10.49, -5.71] pp.
- C → B matched decoder / 相同解码下适配: ΔCER -11.60 pp; 95% speaker-cluster interval [-20.50, -5.98] pp.

区间按整名说话者配对抽样，5000次、seed 0；负值表示字符错误减少。10名说话者及其共有录制条件限制区间的外推。测试结果没有被用来重新选择、重训或调参。

### 使用范围与后续优化

中文输入保留逐次候选确认。本轮验收报告 `ready=false`，未通过项目：`minimum_sessions_per_speaker, webcam_domain, silent_articulation, independent_data, cer, exact_sentence_rate, warm_processing_seconds_p95`。公开数据是普通发声时的 96×96 嘴部裁剪；即使只读画面，也不能据此证明本人刻意无声、自由中文句子的摄像头效果。后续优化需要更丰富的视觉训练、领域适配和独立多场次的真实无声数据，并按新的独立协议验证。

源权重及适配器受研究许可限制，Chinese-LiPS为CC-BY-NC-SA-4.0。仓库只提交源码、协议与汇总测量，不分发权重、视频、逐段参考文本或原始预测。操作见[中英文研究流程](chinese-public-adaptation.md)，产品安装见[中文识别使用说明](CHINESE.md)。

## English

This iteration adds resumable encoder adaptation, development-only decoder comparison and frozen test evaluation. All experiments ran on Apple M4 / 16 GiB and read video frames only: no audio, LLM cleanup or reference-derived decoding prompts. The GUI's CMLR Mandarin backend and this separate CNVSRC research model are different entry points; these measurements are research-pipeline results.

Pinned Chinese-LiPS revision: `db96948538811029011eee44602438a26710ecd9`. Speaker-separated partitions contain 1200 original training clips, 100 development clips and 100 fresh test clips. Exactly 95 whole training records with unsupported labels were excluded explicitly; 1105 retained references are unchanged. All development/test records remain in the denominator, including OOV labels, failed, rejected and empty predictions. The 10 fresh test speakers also exclude every speaker from previously inspected tests. Original base-pretraining overlap is unknown.

### Training and development selection

| Objective | Completed epochs | Selected epoch | Baseline dev CER | Selected dev CER |
| --- | --- | --- | --- | --- |
| ctc | 2 | 2 | 66.34% | 60.84% |
| joint | 2 | 2 | 66.34% | 60.82% |

Both objectives start independently from the same base: at most 6 epochs, a 1800-second soft active budget per objective including inter-epoch development evaluation, final 2 encoder blocks, learning rate 1e-5, seed 0 and frozen remaining layers. Model loading and epoch-zero baseline are excluded from the budget. Only completed epochs can be selected. CTC experienced iCloud I/O delays; joint training used local cache. Throughput can change completed epochs, so unequal-epoch results cannot isolate the loss objective.

The training-stage development winner is `joint`, epoch `2`. Frozen arm B actually uses `joint`, epoch `2`, from `development_comparison`. Frozen arm B uses the adapter that passed the development gates. Frozen decoding uses CTC `0.1`, reverse `0.3`, beam 40, n-best 10, and no external LM. The fixed development comparison evaluates CTC 0.1/0.3/0.5/0.7 and reverse 0.3 only on the raw-CER best setting. Weight 1 was omitted before inference because of the 16 GiB memory limit. Decoder scores are diagnostic values, and reverse rescoring receives hypotheses only.

### Development decoder comparison: all five configurations

These rows use the full development set before freezing; they are not fresh test results. Raw top-1 CER is the principal criterion. The freezing workflow additionally requires identical corpus denominators, no inference failures, and no decline in exact-sentence rate or candidate coverage relative to the original base. Lower CER alone cannot bypass those gates.

| CTC | Reverse | Errors / reference characters | Dev CER | Exact sentences / rate | Candidate coverage |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.0 | 2467 / 4147 | 59.49% | 1/100 (1.00%) | 100.00% |
| 0.3 | 0.0 | 2480 / 4147 | 59.80% | 0/100 (0.00%) | 100.00% |
| 0.5 | 0.0 | 2522 / 4147 | 60.82% | 0/100 (0.00%) | 99.00% |
| 0.7 | 0.0 | 2549 / 4147 | 61.47% | 0/100 (0.00%) | 100.00% |
| 0.1 | 0.3 | 2452 / 4147 | 59.13% | 0/100 (0.00%) | 100.00% |

The highest exact-sentence count among reverse-0 configurations is 1/100; the reverse-0.3 configuration has 0/100. Every configuration is shown; a lower CER must not conceal reduced exact-sentence accuracy or coverage.

### Fresh frozen test

A uses the original base with CTC 0.5/reverse 0. B uses the development-selected pipeline. C uses the original base with exactly B's decoding. All 100 records are scored. Identical arms reuse the same saved predictions explicitly; no repeated inference or independent latency is fabricated.

| Arm | Errors / characters | CER | Exact sentences | Candidate coverage | Failures | Warm p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 3037 / 4087 | 74.31% | 0/100 | 88.00% | 0 | 5.85 s |
| B | 2696 / 4087 | 65.97% | 0/100 | 90.00% | 0 | 6.15 s |
| C | 3170 / 4087 | 77.56% | 0/100 | 87.00% | 0 | 6.13 s |

- A → B overall / 整体流程: ΔCER -8.34 pp; 95% speaker-cluster interval [-10.49, -5.71] pp.
- C → B matched decoder / 相同解码下适配: ΔCER -11.60 pp; 95% speaker-cluster interval [-20.50, -5.98] pp.

Intervals resample paired whole-speaker clusters 5000 times with seed 0; negative values mean fewer raw errors. Ten speakers and shared recording conditions limit generalization. Test results did not alter model/decoder selection or trigger further training.

### Scope and continued optimization

Mandarin input keeps per-utterance candidate confirmation. The declared report returns `ready=false`; failed criteria: `minimum_sessions_per_speaker, webcam_domain, silent_articulation, independent_data, cer, exact_sentence_rate, warm_processing_seconds_p95`. Public clips contain voiced 96×96 mouth crops, so frame-only evaluation does not establish deliberately silent, free-form webcam accuracy. Further work requires richer visual training, domain adaptation, and independent multi-session silent-camera data under a new evaluation protocol.

Base and adapted parameters retain their research restrictions; Chinese-LiPS is CC-BY-NC-SA-4.0. Only source, protocols and aggregate measurements are committed. Weights, media, per-sample labels and raw predictions are not redistributed. See the bilingual [workflow](chinese-public-adaptation.md) and [installation guide](CHINESE.md).

## Provenance / 可追溯信息

- Measurement source commit: `8a7584fb4f7cefd86e9087c3057400d74b57af79`.
- Test freeze SHA-256: `329c7dddb124181c5085445d4520a70690fa0ed9cf8a3be8b764f9c1baadd9e5`.
- Input identity SHA-256: `9ab8af366f11f02c5e8ade5d50a43a7909e167dffaee07b861b66ad270f3ccde`.
- Development decoder comparison SHA-256: `9f8c0007d85fff4a3f7f428c2f14337be1c9efdb7f8fa14753a986038367a978`.
- Measurement completed at: `2026-10-03T03:06:17.898430+08:00`.
- Training-stage candidate adapter SHA-256: `e85ccabd47dc62fbcfbbfc0f4a040e322f4e2f86ba15714b5e341a2e950c2457`.
- Frozen arm B adapter SHA-256: `e85ccabd47dc62fbcfbbfc0f4a040e322f4e2f86ba15714b5e341a2e950c2457`.
- Test A/B/C SHA-256: `99099b22cd3a715ea93ee2a825e94be839b7c5f925de6f6d4bfdc9ace38ab0a8` / `a724a8189f6c72c97f01c7b3b630ba2b3f952ff2bd72ad3e5a8f5fcd62a6092d` / `77cb0fd5a6086dfd321aec5c18701e0d36105b3a69c1e85ee919e852aaedf9c8`.
- Aggregate machine-readable metrics: [mandarin-round4-summary.json](benchmarks/mandarin-round4-summary.json).
