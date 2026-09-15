# EventGround-WAM：12-Week Execution Plan

## Project thesis

**Working title:** *EventGround-WAM: Do Temporally Grounded Future Representations Improve VLA/WAM Robustness on LIBERO-Plus?*

核心研究问题：

> Pixel 与 latent future targets、auxiliary 与 inference-time dynamics coupling、fixed 与 event-adaptive temporal prediction，分别如何影响 VLA/WAM 在 LIBERO-Plus distribution shifts 下的鲁棒性？

关键子问题：

1. Pixel reconstruction 和 JEPA latent prediction，哪个更适合学习机器人轨迹中的 temporal dynamics？
2. Future prediction 只作为训练 auxiliary，还是在 inference 时进入 action path，更能提升鲁棒性？
3. 固定 future horizon 和 event-adaptive horizon，哪个更适应 robot initial-state 和 layout shifts？
4. Simulator-verifiable event grounding 经 SFT/RLVR 后，能否提高 JEPA-WAM 的时间校准与任务成功率？

## Assumptions and scope

- 单人 side project，每周约 15–20 小时。
- 本地 RTX 5090 32GB 持续训练；关键阶段短租 H100。
- 唯一 shared VLM backbone：`Qwen/Qwen3.5-2B`，首次下载前固定不可变 revision 和文件 manifest。
- 冻结 vision encoder 和大部分 LLM；训练 projector、上层 LoRA、future predictor 和 action-query Transformer。
- 所有主实验使用同一个 deterministic action-query Transformer，以 normalized action-chunk L1 为目标。
- 不使用 Qwen-VLA、Qwen-RobotManip 或任何其他预训练 action-policy
  checkpoint；不将它们作为 fallback、依赖或主实验矩阵成员。
- 主 benchmark 只用 LIBERO-Plus；Standard LIBERO 只作 clean sanity check。
- 不加入 RoboTwin、外部自然视频或独立 video grounding benchmark。
- 删除原计划中的 optional latent-flow M5。
- Flow matching 和其他 generative action objectives 不阻塞任何主实验，移至 12 周核心研究之外的可选 future work。
- RLVR 只优化 EventGround head、少量 backbone LoRA 和 horizon-conditioning layers；不对连续 action decoder 做 on-policy RL。

## Final experiment matrix

| ID | Model | Future target | Future used at inference? | Event-adaptive horizon? |
|---|---|---|---:|---:|
| B0 | Pure VLA | None | No | No |
| M1 | Pixel-aux VLA | Pixel reconstruction | No | No |
| M2 | JEPA-aux VLA | JEPA latent | No | No |
| M3 | Pixel-WAM | Pixel hidden future tokens | Yes | No |
| M4 | JEPA-WAM | Predicted JEPA future tokens | Yes | No |
| M4-EG | EventGround JEPA-WAM | Event-conditioned JEPA future tokens | Yes | Yes |

M4-EG 保存两个 checkpoint：

- `M4-EG-SFT.ckpt`
- `M4-EG-RLVR.ckpt`

## Week 1 — Lock protocol and environment

**Goal:** 先证明 benchmark、simulator 和 evaluation pipeline 可靠。

Tasks:

- 安装并跑通 LIBERO-Plus。
- 下载 assets、demonstrations 和 task classification。
- 跑官方或公开 checkpoint 的 20–50 个任务。
- 实现统一 episode recorder：instruction、agent RGB、wrist RGB、proprioception、action、MuJoCo state、task success。
- 固定 image resolution、control frequency 和 action chunk length。
- 建立三档 evaluation：
  - `dev-210`：快速开发。
  - `stratified-1K`：中期模型比较。
  - `full-10K`：最终评测。
- `dev-210` 从 7 类 perturbation × 5 个 difficulty level 中分层抽样。

Deliverables:

- 可重复的 environment setup。
- Evaluation script。
- 官方 checkpoint sanity-check 结果。

**Gate:** 若无法稳定复现官方 checkpoint 的基本行为，不开始训练。

## Week 2 — Dataset and automatic robot-event labeling

**Goal:** 把 demonstration trajectory 转成统一的 CPT/SFT 数据格式。

自动提取候选事件：

- reach
- contact
- gripper close
- grasp established
- lift
- transport
- target-region entry
- release
- success

示例：

```json
{
  "instruction": "put the red mug in the cabinet",
  "events": [
    {"type": "contact", "interval": [18, 22]},
    {"type": "grasp", "interval": [22, 31]},
    {"type": "lift", "interval": [31, 45]},
    {"type": "place", "interval": [78, 91]}
  ]
}
```

同时生成 multi-horizon samples：

\[
h \in \{2,4,8,16,32,\text{next-event}\}
\]

Tasks:

- 随机人工检查 200 条轨迹。
- 可视化 event boundaries。
- 缓存 image 或 JEPA features。
- 做 train/validation split，避免相同 initial seed 泄漏。

Deliverables:

- Trajectory dataset。
- Event annotation pipeline。
- 200 条人工检查报告。

**Gate:** 关键事件自动标注准确率约 95% 以上；否则缩减为 contact、grasp、success 三个最可靠事件。

## Week 3 — Implement the shared VLA baseline

**Goal:** 完成不含 world modeling 的 B0。

**Status:** 已于 2026-09-13 完成，并由 `week3-complete-v1` 固定；包含一个明确记录的验收修订。

Architecture:

```text
language + agent RGB + wrist RGB + state
                    ↓
          Qwen3.5-2B backbone
                    ↓
  action-query Transformer + continuous L1
                  + gripper BCE
                    ↓
              action chunk
```

Tasks:

- 创建独立 B0 training environment，不修改 Week 1 reference-policy runtime。
- 固定 Qwen3.5-2B revision、processor、依赖和完整文件 manifest。
- 实现 action normalization。
- 实现 learned action queries、future-conditioning memory interface 和 normalized-action L1 loss。
- 保持 continuous action 的 L1 objective；binary gripper 使用 unit-weight BCE，
  所有后续 matched policies 复用相同 hybrid objective，不引入 flow matching。
- Single-batch overfit。
- 10 条 demonstration overfit。
- Closed-loop rollout。
- 检查 action latency 和显存占用。

Deliverables:

- B0 可训练代码。
- 固定的 Qwen3.5-2B model/runtime provenance。
- Overfit curve。
- 第一次成功 closed-loop rollout。

**Gate:** 必须先能在少量训练任务上明显 overfit。

Completion record:

- 已固定 Qwen3.5-2B revision、完整文件 manifest、processor contract 和独立的
  126-package B0 runtime；没有修改 Week 1 reference-policy 环境。
- train-only normalization 覆盖 11,521 个训练 episodes 和 1,796,601 个
  actions；全部 2,238,036 个数据集 actions 的反归一化最大误差为
  `1.12e-16`。确定性 loader 提供 1,715,954 个有效 `8 x 7` chunks。
- dual-image/state conditioning、rank-16 LoRA allowlist、action-query
  Transformer、frozen-base gradients、BF16 forward/backward 和 checkpoint
  exact reload 均通过。
- fixed-batch loss ratio 为 3.803%，decoded MAE 为 `0.048459`。最终
  ten-demonstration hybrid-objective ratio 为 `0.126083`，decoded MAE 为
  `0.027970`。
- 原定 100% gripper gate 未通过，结果仍保留为 failed。最终 checkpoint 为
  11,580/11,632（99.552957%）；经 reviewer 批准，在确认全部 52 个错误都位于
  真实 gripper transition 前后四帧内且没有 label contradiction 后，作为 Week 3
  engineering exception 接受。该修订不改变后续 evaluation threshold。
- predeclared 10-task closed-loop sweep 有 1 个 task 成功；成功 task 在 shared
  HDF5 recorder 中于第 118 个 policy step 再次成功，并通过 protocol、provenance、
  action alignment 和 terminal-signal 深度验证。
- concurrent profile 的 process-attributed GPU 峰值为 5.86 GB，Torch
  allocated/reserved 峰值为 4.59/4.65 GB，观测到的全局剩余显存至少 28.07 GB；
  15 次 query 的 median latency 为 32.94 ms。Week 3 不设置 latency target。
- 最终 191 个 tests 与 lint 全部通过。完整记录见
  [`WEEK3_STATUS.md`](WEEK3_STATUS.md) 和
  [`evaluation/week3-completion-v1.json`](evaluation/week3-completion-v1.json)。

Week 4 is unblocked. 但 1/10 cohort smoke rate 和首次 perturbed rollout 失败仍是
明确风险；它们转化为 Week 4 的 full training、Standard LIBERO、`dev-210` 和
七类 perturbation failure analysis 工作，不被 Week 3 closure 隐藏。

## Week 4 — Train and evaluate B0

**Goal:** 建立所有后续实验的共同 policy initialization。

**Status:** started on 2026-09-14; updated 2026-09-15. Weeks 1–3 are closed. The
five-epoch recipe and training/recovery/evaluation infrastructure are implemented,
with 209 tests passing locally and on Feifei. A host restart resumed the pilot;
the corrected deterministic pilot's final result awaits Feifei connectivity.
Production has not been launched by this task. Progress, inherited contracts,
pilot evidence, and the remaining clean-baseline gate decision are tracked
in [`WEEK4_STATUS.md`](WEEK4_STATUS.md). The six-step execution sequence is in
[`ROADMAP.md`](ROADMAP.md#week-4-train-evaluate-and-freeze-the-shared-b0-baseline).

Tasks:

- 完整训练 B0。
- Standard LIBERO clean evaluation。
- LIBERO-Plus `dev-210` evaluation。
- 分析七类 perturbation 的 failure profile。
- 固定后续模型的 training samples、optimizer、action steps、LoRA rank、random seed 和 evaluation task IDs。

Deliverables:

- `B0.ckpt`
- Clean/perturbed baseline table。
- Failure videos。

**Gate:** B0 在 Standard LIBERO 上应接近同配置公开 baseline，建议差距不超过约 5 个百分点；否则优先修 policy。

## Week 5 — Pixel temporal CPT

**Goal:** 训练 pixel future predictor，暂不训练 action policy。

Input and target:

\[
(o_t,l,a_{t:t+h},h) \rightarrow o_{t+h}
\]

推荐使用轻量 deterministic reconstruction，不做 video diffusion：

- L1
- SSIM
- 可选 LPIPS
- 128×128 或 160×160 future target

Tasks:

- Current-frame copy baseline。
- Constant-background baseline。
- Short/medium/event horizon mixture。
- Temporal-consistent augmentation。
- 训练 pixel future module。

Deliverables:

- `C_pixel.ckpt`
- Reconstruction metrics。
- 不同 horizon 的 qualitative rollouts。

**Gate:** 必须显著优于复制当前帧的 baseline，否则 pixel branch 尚未学到 dynamics。

## Week 6 — JEPA temporal CPT

**Goal:** 在与 Pixel CPT 完全相同的数据上训练 JEPA future predictor。

Objective:

\[
\mathcal L_{\mathrm{JEPA}} = d\left(F(z_t,a,h),\operatorname{sg}(E_{\mathrm{target}}(o_{t+h}))\right)
\]

Design:

- Frozen DINO 或 Qwen vision target encoder。
- Cosine loss + variance regularization。
- Predictor 参数量尽量与 pixel branch 的核心 transformer 接近；pixel decoder 不计入核心参数公平性比较。
- 比较 fixed horizon、multi-horizon 和 next-event horizon prediction。
- 测试 appearance perturbation 下的 feature consistency。

Deliverables:

- `C_jepa.ckpt`
- Pixel vs JEPA representation diagnostics。
- Future retrieval accuracy。

**Gate:** JEPA predictor 必须优于 copy-latent baseline，且无明显 representation collapse。

## Week 7 — Train auxiliary VLA models M1 and M2

**Goal:** 检验 future prediction 作为 representation regularizer 的价值。

M1 objective:

\[
\mathcal L = \mathcal L_{\mathrm{action}} + \lambda_p\mathcal L_{\mathrm{pixel}}
\]

M2 objective:

\[
\mathcal L = \mathcal L_{\mathrm{action}} + \lambda_j\mathcal L_{\mathrm{JEPA}}
\]

Constraints:

- M1、M2 均从 B0 初始化。
- Future predictor 参与训练。
- Action decoder 不 cross-attend future tokens。
- 推理时丢弃 future module。
- M1/M2 action path 参数完全一致。
- 只在 `dev-210` 上评测。

Deliverables:

- `M1.ckpt`
- `M2.ckpt`
- Auxiliary future loss 与 policy success 的相关性分析。

## Week 8 — Train inference-coupled WAM models M3 and M4

**Goal:** 检验 predicted future 是否必须在 inference 时进入 action path。

Design:

- M3：Action decoder cross-attend pixel predictor 的 hidden future tokens。
- M4：Action decoder cross-attend JEPA predicted future tokens。
- 两者均从 B0 初始化，并加载对应 CPT checkpoint。

Fairness constraints:

- M1/M3 共用同一个 `C_pixel`。
- M2/M4 共用同一个 `C_jepa`。
- 相同 action training steps、demonstrations、action head 和 seeds。
- Coupling 是唯一主要差异。

Deliverables:

- `M3.ckpt`
- `M4.ckpt`
- Inference latency。
- Predicted-future visualizations。

Cloud window 1：短租 H100，并行训练 M3/M4。

## Week 9 — Core 2×2 factorial study

**Goal:** 在加入 EventGround 前回答原始 controlled-study 问题。

在相同 `stratified-1K` 上评测 B0、M1–M4：

- Standard LIBERO success rate。
- LIBERO-Plus overall success rate。
- 七类 perturbation success rate。
- Difficulty L1–L5。
- Robustness ratio：

\[
\mathrm{Robustness\ Ratio}=\frac{SR_{\mathrm{LIBERO\text{-}Plus}}}{SR_{\mathrm{Clean}}}
\]

Analysis:

- Pixel vs JEPA main effect。
- Auxiliary vs coupled main effect。
- 两者 interaction。
- Paired task bootstrap confidence interval。
- Success/latency trade-off。

允许且有研究价值的结果：

- M4 > M2：future inference coupling 有价值。
- M4 ≈ M2：world prediction 主要作为 representation regularizer。
- M2 > M4：错误的 imagined future 会伤害 action generation。
- Pixel 对 clean 有利，而 JEPA 对 perturbation 更稳健。

**Gate:** 若 M4 明显差于 M2，先分析 prediction error；EventGround 的目标之一就是判断 adaptive horizon 能否修复 fixed-horizon future 的不可靠性。

## Week 10 — EventGround SFT and temporal RLVR

**Goal:** 只在 JEPA-WAM 分支上加入 event-adaptive horizon。

### SFT

训练 2K–5K 个高质量结构化样本：

```json
{
  "next_event": "grasp",
  "interval": [12, 18],
  "horizon": 16
}
```

### RLVR

Reward:

\[
R = 0.40R_{\mathrm{event\text{-}IoU}}
  + 0.25R_{\mathrm{time\text{-}error}}
  + 0.20R_{\mathrm{predicate}}
  + 0.15R_{\mathrm{format}}
\]

Constraints:

- Group size 4。
- 短结构化输出。
- 只更新 EventGround head 和 backbone LoRA。
- 不对连续 action decoder 做 on-policy RL。
- 先跑 500 prompts，再扩到 2K–5K。

Deliverables:

- `EventGround-SFT.ckpt`
- `EventGround-RLVR.ckpt`

**Gate:** 若 RLVR 未提高 validation event-IoU，最多进行一次 reward calibration；仍无改善则保留 SFT checkpoint，停止扩大 RL scope。

## Week 11 — Integrate M4-EG and targeted evaluation

**Goal:** 让预测的 event/horizon 真正决定 future prediction target。

Inference path:

```text
current observation
        ↓
predict next event + horizon
        ↓
JEPA predicts event-conditioned future latent
        ↓
Action decoder cross-attends future latent
        ↓
action chunk
```

重点评测：

- Robot initial-state perturbation。
- Layout perturbation。
- Camera perturbation。
- Contact-rich tasks。
- Fixed M4 vs adaptive M4-EG。
- SFT vs RLVR checkpoint。
- Horizon prediction calibration。
- Inference latency。

Primary comparison:

\[
M4_{\mathrm{fixed}} \quad\text{vs}\quad M4\text{-EG}_{\mathrm{adaptive}}
\]

Deliverables:

- `M4-EG-SFT.ckpt`
- `M4-EG-RLVR.ckpt`
- Targeted robustness table。
- Adaptive-horizon qualitative examples。

## Week 12 — Final evaluation, statistics, and report

**Goal:** 形成完整、可复现、可用于求职展示的研究结果。

Preferred evaluation:

- 六个模型全部跑完整 LIBERO-Plus 10,030 tasks。

Fallback if evaluation cost is too high:

- 所有模型：相同 stratified 2K tasks。
- B0、M2、M4、M4-EG：完整 10K。
- B0、M4、M4-EG：额外 3 个 training seeds。

Final analysis:

- Paired bootstrap 95% CI。
- Pixel vs JEPA effect。
- Auxiliary vs coupled effect。
- Fixed vs event-adaptive effect。
- Clean performance vs robustness。
- Inference latency。
- Future-prediction quality 与 task success 的相关性。
- Failure taxonomy：wrong object、wrong event timing、hallucinated future、failed contact、language ignored、recovery failure。

Final deliverables:

- 6–8 页 technical report。
- 完整代码和 configs。
- Checkpoints。
- Benchmark predictions。
- 3–5 分钟 demo video。
- 一篇面向求职的技术博客。
- Résumé bullet 和 project summary。

Cloud window 2：full evaluation 和核心模型 multi-seed runs。

## Compute budget

| Workload | Estimated RTX 5090 time |
|---|---:|
| B0 + debugging | 30–60 GPU hours |
| Pixel CPT | 40–80 GPU hours |
| JEPA CPT | 30–60 GPU hours |
| M1–M4 policy training | 100–180 GPU hours |
| EventGround SFT/RLVR | 20–40 GPU hours |
| Evaluation/inference | Depends on simulator parallelism |

预计总计约 220–420 个 RTX 5090 GPU 小时。双摄像头轨迹和 multi-horizon cache 可能占用数百 GB；优先缓存 vision features，避免重复解码全部图像。

## Scope guardrails

- Week 4 前不实现 world model。
- Week 6 前不实现 WAM coupling。
- Week 9 前不实现 EventGround。
- 不在主矩阵中加入外部人类视频。
- 不同时做 RoboTwin。
- 不训练 7B 模型。
- 不引入 Qwen-VLA、Qwen-RobotManip 或其他预训练 action policy。
- 不把 video diffusion 加入 pixel branch。
- RLVR 失败后最多允许一次 reward redesign。
- 完整消融集中在 shared 2B backbone；如算力不足，优先保证 B0、M2、M4、M4-EG。
- Negative result 不是项目失败；controlled evidence 本身就是主要价值。

## Success criteria

最低成功标准：

- 完成可靠的 B0 和至少 M2/M4 controlled comparison。
- 在分层 LIBERO-Plus evaluation 上得到 paired statistical evidence。
- 明确说明 future prediction 是 regularizer、inference mechanism，还是在某些 perturbation 下有害。

完整成功标准：

- 完成 B0、M1–M4、M4-EG 六模型矩阵。
- 证明 fixed vs event-adaptive horizon 在 robot initial-state/layout shifts 下存在可解释差异。
- 发布代码、模型、评测结果、技术报告和 demo。
