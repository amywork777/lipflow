# Lipflow

[English](README.md) · [简体中文](README.zh-CN.md)

按住输入键，对着摄像头动嘴说话，松开后识别成文字。视觉识别不需要麦克风，模型在本机运行。可选的文本纠错支持本地模型、Ollama 或 Claude。

## 中文唇语识别

本分支增加中文视觉模型、候选确认、统一纠错检查，以及可复现的中文公开数据训练和评测工具。后续将持续优化中文识别、领域适配和真实摄像头表现。功能说明提供中英文版本。

| 功能 | 作用 |
| --- | --- |
| 中文纯视觉识别 | 使用 CMLR 中文权重与字符词表，从摄像头或视频识别中文 |
| 候选确认 | 浮窗显示原始识别、纠错建议和其他候选；按 1、2、3 选择，Esc 重说 |
| 画面质量提示 | 综合人脸可见比例、嘴部像素、亮度和对比度，提示距离、姿态或光线问题 |
| 统一纠错检查 | 本地模型、Claude、Ollama 共用数字、日期、金额、否定及配置姓名/术语变化检查 |
| 原文保留 | 候选窗、历史和 Copy raw recognition 菜单保留原始结果 |
| 中文研究训练 | Chinese-LiPS 固定分区、CNVSRC 编码器 CTC/联合训练、完整轮次恢复及开发集选择 |
| 独立评测 | 测试前冻结参数，保留失败和空输出，报告字符错误率、整句准确率及说话者配对区间 |

中文结果统一要求确认。候选分数与分数差不是正确概率，规则检查也不能证明语义完全一致。CMLR 产品入口与 CNVSRC 研究入口分别运行；研究适配权重不会自动启用到产品。公开普通说话的嘴部视频可以衡量视觉识别，但不能代替本人刻意无声说话的摄像头测试。实际结果见[验证记录](docs/VALIDATION.md)。

## 基础安装

### Apple Silicon Mac

按上游要求使用 macOS 13 或更新版本；Liquid Glass 外观需要 macOS 26。先安装基础运行环境和英文模型：

```bash
git clone --branch feat/mandarin-safe-dictation https://github.com/BryceYuuu/lipflow.git ~/code/lipflow
cd ~/code/lipflow
./setup.sh
open /Applications/Lipflow.app
```

首次启动需要摄像头、输入监控及辅助功能权限。通过安装器启动时权限属于 Lipflow；从终端运行时权限可能属于终端。权限开关已打开但检测仍未通过时，使用设置窗口的 Restart Lipflow。

个人片段、短语及模型保存在 `~/Library/Application Support/Lipflow/`，日志为 `~/Library/Logs/Lipflow.log`。可在系统设置 → 通用 → 登录项中添加 Lipflow。

### Windows

使用 Windows 10 或 11 的 64 位版本，在 PowerShell 中运行：

```powershell
git clone --branch feat/mandarin-safe-dictation https://github.com/BryceYuuu/lipflow.git $HOME\code\lipflow
cd $HOME\code\lipflow
powershell -ExecutionPolicy Bypass -File setup.ps1
```

从开始菜单启动 Lipflow，托盘中显示应用图标。默认按住 Right Ctrl 输入，可在菜单选择其他按键。检查系统摄像头权限，并使用 `uv run lipflow doctor` 诊断运行环境。无 NVIDIA GPU 时使用 CPU，速度取决于硬件。Windows 没有 Mac 的 MLX 本地纠错路径，可使用 Ollama、Claude 或离线基本格式化。

个人数据与日志位于 `%APPDATA%\Lipflow`。Windows 候选选择后的目标检查目前基于窗口 HWND；Mac 还检查应用 PID 和输入控件。真实 Windows 中文输入操作的验证范围见验证记录。

## 安装中文视觉模型

基础安装完成后，阅读 [CMLR 来源与许可](docs/CHINESE.md)，再下载可选研究模型：

```bash
uv sync --extra chinese
uv run --extra chinese lipflow install-chinese --accept-research-license
uv run lipflow run --language zh --cleanup basic --confidence-policy review
```

安装器校验权重档案 SHA-256，不分发第三方权重。该视觉模型使用汉字字符词表，不能据此宣称支持自由中英混合识别。`--cleanup basic` 避免调用文本大模型；使用自动纠错后端且环境中存在 Anthropic 凭证时，可能调用 Claude。

对本地完整人脸视频识别：

```bash
uv run lipflow file face-video.mp4 --language zh --cleanup basic
```

已经对齐的 96×96 嘴部视频需要明确声明：

```bash
uv run lipflow file mouth-video.mp4 --language zh --mouth-roi --cleanup basic
```

菜单可选择识别语言、输入模式、忠实/润色模式及候选策略，设置需要重启生效。CLI 参数优先于保存的设置。详细行为、中文练习训练与视频参数见[中文使用说明](docs/CHINESE.md)。

## 日常操作

| 操作 | 结果 |
| --- | --- |
| Mac 按住 Right Option / Windows 按住 Right Ctrl，动嘴后松开 | 开始和结束一段输入 |
| 双击输入键，再按一次 | 免持续按键输入，最长 60 秒 |
| 录制时按 Esc | 取消 |
| 候选窗按 1、2、3 | 确认所选文字 |
| 候选窗按 Esc | 取消并重说 |
| 菜单 Copy raw recognition | 复制原始识别 |

松开输入键后会继续采集约 0.4 秒，以保留最后一个词的嘴部动作。数字快捷键只在候选窗口有焦点时使用。用户等待时改变输入目标可能导致仅复制结果，需要手动粘贴。

英文可以选择综合分数差、CTC 一致及画面质量的启发式自动策略；中文保持确认流程。低质量画面、明显重复、无脸或无嘴部动作会提示调整或重说。原始结果不会因纠错建议被覆盖。

## 文本纠错与个人短语

默认自动后端按可用条件选择 Claude、本地 MLX 模型或离线规则，也可显式使用 Ollama。Claude 会把识别文本及选定上下文发送给服务；本地模型和离线基本格式化在本机执行。中文与英文均经过统一敏感改动检查，忠实模式限制改词，润色模式的措辞变化需要确认。

通过菜单编辑自定义词汇；中文按字符短语匹配。菜单可重新练习和训练，中文与英文片段、脸部模型分开保存。是否保留个人适配继续以留出数据检查；少量练习不构成跨日期、光线或说话者准确率保证。

英文个人语言模型训练及 Wispr Flow 导入仍保留：

```bash
uv run lipflow import-wispr
uv run lipflow import-wispr --from-text my-writing.txt
```

中文模式不加载英文个人语言模型，支持中文历史短语检索。参数列表可用 `uv run lipflow --help` 和各子命令的 `--help` 查看。

## 中文公开数据训练与验证

新的研究流程固定来源版本、权重/词表/配置哈希及分区，先训练和开发集选择，再冻结单个候选进入新测试集。支持 CTC 和 CTC/双向注意力联合损失、完整轮次的参数/优化器/RNG 恢复、候选分数重排与配对分析。模型处理视频帧，参考文本只用于训练目标或完成预测后的计分，不进入测试解码。

查看以下中英文说明：

- [公开数据适配工作流](docs/chinese-public-adaptation.md)：固定来源、训练恢复、开发集解码选择和冻结测试。
- [中文纯无声验收协议](docs/CHINESE_EVALUATION.md)：独立摄像头数据及明确质量门槛。
- [数据适配与可选本地采集](docs/CHINESE_ADAPTATION.md)：历史实验复现及人工确认采集工具。
- [验证记录](docs/VALIDATION.md)：已经执行的测试、指标及适用范围。

大型权重、数据及 Python 环境建议放在非 iCloud 同步路径，例如 `~/.cache/lipflow-research/`。新权重仅用于人工研究评测，公开数据及其权重有独立许可，不能因本仓库采用 MIT 就获得商用权。

另有可选中文低声模式，需要真实可听声音并使用 Whisper 音频识别与画面质量门控。它不属于纯无声唇读，安装和边界见中文使用说明。本次中文公开数据优化以纯视觉模型为目标。

## 开发与许可

```bash
uv sync --frozen
uv run pytest
```

测试范围与实际执行数量见验证记录。自动化测试不代表真实中文识别准确率或已经验证跨应用粘贴。英文原有 Auto-AVSR 模型、人脸对齐、25 fps 采样及平台组件的技术说明见 [English README](README.md)。

代码采用 MIT，见 [LICENSE](LICENSE)；第三方代码及模型许可见 [NOTICE](NOTICE)。CMLR、LRS3、CNVSRC 权重和 Chinese-LiPS 数据的限制独立适用。权重、视频及个人录制不会随此分支或 PR 分发。
