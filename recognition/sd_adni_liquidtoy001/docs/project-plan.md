# Project Plan and Feasibility Review Outline

**Historical outline below was recorded on 2 October. Current status is in the 7 October coverage ledger at the end; old pending items are not current claims.**

Recorded: 2026-10-02 (Australia/Brisbane).

This is a planning record, not a completed feasibility review or evidence of tutor check-off.

## Research question

How does domain adaptation change structural fidelity, diversity, and evidence of memorization in generated ADNI brain MRI slices?

## Intended user and scope

The user is a researcher evaluating synthetic MRI cohorts. Study 2D image generation and its limitations. Do not infer clinical safety or privacy guarantees from image appearance or nearest-neighbor checks alone.

## Proposed acceptance criteria

1. Produce an auditable patient-level split manifest, with zero known patient overlap. If identifiers are unavailable or ambiguous, resolve the split design before training.
2. Execute a real forward/backward/update step and a save/restart/resume check with finite loss and gradients. Record restored optimizer, scheduler, step, and relevant random/data state.
3. Compare an implemented simple generator, an unadapted SD model, and an adapted SD model under one documented evaluation protocol.
4. Report uncurated generated samples, distribution/coverage measurements, nearest-neighbor audits, and 3–5 representative failures. Specify how SSIM reference pairs are formed; do not treat arbitrary image pairs as a validated generation metric.
5. Measure peak GPU and host memory, step time, and checkpoint size. Demonstrate durable checkpoint storage within verified available space and completion or recovery within the verified Slurm wall-time limit.

These criteria are proposals. Numerical quality targets and resource budgets must be finalized using the course requirements and actual data/resource evidence.

## Execution sequence

1. Read-only Rangpur audit: current partition/account limits, available persistent storage, environment compatibility, and dataset layout/identity metadata.
2. Data audit: patient grouping, image dimensions/orientation, intensity handling, duplicates, and label availability. Split patients before selecting slices. Fit any learned preprocessing only on training data.
3. VAE reconstruction check: compare real slices with reconstructions and error maps to identify compression limitations.
4. Short training and recovery test: choose a versioned pretrained checkpoint and run a bounded experiment; measure resources before estimating a full run.
5. Baseline and adaptation experiments: select a simple generative baseline, fix evaluation settings, and vary one justified adaptation factor at a time.
6. Final evaluation/report: reserve held-out test data for the finalized protocol, write limitations and recommendations, verify reproduction, then prepare the PR and PDF.

## Rangpur storage and scheduling

Use CPU jobs for environment setup and preprocessing, GPU jobs for training/evaluation, and the login node for lightweight operations and job submission. Keep reconstructible caches in job scratch; keep recovery checkpoints in verified persistent storage and back up completed artifacts locally.

Historical user records report an A100 PCIe 40GB, a 12-hour course partition, and constrained home storage. These are not current resource guarantees. Verify them before writing final job resource directives. GPU memory and Slurm `--mem` host-memory allocation are distinct.

## Open decisions and risks

- Confirm the permitted pretrained-library scope and whether LoRA is sufficient for the main SD implementation.
- Determine whether the course data has reliable patient identities and a usable conditional label/slice description.
- Select the exact base-model identifier/revision and verify access and terms.
- Select and implement the simple baseline; unadapted SD alone does not replace the requested simple generative baseline.
- Check diffusion dependencies against the available PyTorch environment without disrupting other projects.
- Budget persistent storage for optimizer state and overlapping files during atomic checkpoint replacement.
- Validate signal handling for the actual launcher/process tree; previous project recovery tests do not validate this trainer.
- Confirm the midpoint consultation date and final submission time.

## Preliminary evidence

The [2 October audit](rangpur-audit-2026-10-02.md) found 30,520 grayscale JPEG slices from 1,526 scans and 680 patients, with metadata available for patient grouping. The supplied train/test directories share 216 patients and must not be described as patient-independent. Home has approximately 2.20 GiB free; a durable storage budget is still unresolved. No baseline smoke test or model training has occurred. Add the final split design and resource budget before presenting the one-page feasibility review.

## 2026-10-07 要求覆盖与下一步（当前状态）

| 要求 | 现有证据 | 状态 / 缺口 |
| --- | --- | --- |
| 患者隔离与数据协议 | 476/102/102患者，全量哈希与加载验收 | 已验收；测试集未用于评价 |
| SD与独立简单基线 | LoRA20k步，条件VAE5轮 | 已运行；LoRA/预训练库范围需教师确认 |
| 训练稳定与资源 | 8步恢复严格一致、20k记录、显存/耗时 | 已有证据；非完整20k连续对照 |
| SSIM与分布指标 | 两种配对SSIM、像素/VAE特征双向近邻 | 已计算；SD均值低于0.6，不能标为达标 |
| t-SNE/UMAP | 396点联合t-SNE，元数据/提示标签区分 | 已生成；共享编码器偏差，不代表临床标签正确 |
| 随机种子完整网格 | 三模型各64图，CFG192图 | 已检查；保留全部输出 |
| 3–5失败案例 | 灰背景、颜色伪影、基线模糊 | 3类观察；SD自身独立失败案例与根因仍需学生细化，风险探针不冒充失败 |
| 最近邻与插值 | 像素/特征近邻、噪声路径、真实/生成VAE码路径 | 有证据；不能据此证明无记忆或安全 |
| 一页可行性与check-off | E:/Documents/COMP3710/finalreport/checkoff/feasibility-review-2026-10-07.pdf | 一页英文讨论稿已制作；学生审阅及教师记录未完成 |
| 最终代码入口 | dataset.py和诊断脚本 | modules.py/train.py/predict.py未统一，复现环境清单仍需整理 |
| README与PDF、最终PR | 本地渐进提交与备份、AI_USAGE | 正式README报告/PDF/PR未完成，禁止当作已提交 |
| 最终测试评价 | test封存 | 固定协议后再执行，不为调参提前打开 |

优先顺序：先学生审阅一页材料并完成咨询；并行整理已有代码的最终入口与复现说明。
不应继续无限扩展采样搜索。后续训练应有单一假设、对照预算和明确停止条件。

### 咨询时需要你能够解释

1. 为什么216位患者跨原始目录会导致泄漏，患者划分如何解决它？
2. SD在预测什么、LoRA更新哪些参数，为什么冻结其余组件？
3. 为什么VAE更模糊却SSIM更高？为什么不能挑最高SSIM配对？
4. t-SNE图和最近邻审计各不能证明什么？
5. 明确区分观察（灰背景）与原因假设（先验/采样/训练不足）。

咨询记录仍为空；不得填入未发生的教师批准或完成日期。
