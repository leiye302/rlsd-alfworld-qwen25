# RLSD on ALFWorld: Qwen2.5-1.5B / 3B, 4×A100 80GB

在同一ALFWorld训练与评测协议下运行 **RLSD（Self-Distilled RLVR）** 对比实验。两个尺寸分别从官方 `Qwen2.5-1.5B-Instruct` / `Qwen2.5-3B-Instruct` 基模开始，各训练 **150 updates**。每个尺寸使用完整四卡，默认顺序运行，不需要重新采集SFT数据或配置API。

**实现来源说明：** 复用 [SDAR官方仓库提供的RLSD多轮Agent基线](https://github.com/ZJU-REAL/SDAR/blob/29dcb12a55d8fc17ba6ae03fd0a69048f43051f3/examples/rlsd_trainer/run_alfworld_3b.sh)，锁定SDAR commit `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`。RLSD原论文/作者代码见 [Self-Distilled RLVR](https://arxiv.org/abs/2604.03128)、[iie-ycx/RLSD](https://github.com/iie-ycx/RLSD)。原作者仓库面向视觉推理；这里明确采用SDAR提供的ALFWorld适配，不冒充RLSD作者发布的原生ALFWorld实验。

## 1. 训练方法与配方

Student正常生成交互轨迹，环境奖励给出GRPO序列优势。Teacher使用当前Actor的相同权重，在Student原前缀上额外获得官方静态skill，对Student已采样token重新评分。Teacher不生成替代动作、不改变环境奖励，Student始终看不到skill。

RLSD用Teacher与Student的概率差改变token更新幅度，保留环境优势的方向。下面是 **SDAR的RLSD基线实际实现**：

$$
\Delta_t=\operatorname{sg}[\log\pi_T(y_t)-\log\pi_S(y_t)],\qquad
w_t=\operatorname{clip}\!\left(\exp(\operatorname{sign}(A)\Delta_t),0.8,1.2\right).
$$

$$
\hat A_t=A[(1-\lambda(s))+\lambda(s)w_t],\qquad
\lambda(s)=0.5\max(0,1-s/100).
$$

`compute_rlsd_token_advantage`在no-grad区域计算权重/优势，再把结果传给原生PPO/GRPO Actor更新。**没有SDAR sigmoid gate，也没有额外的SDAR辅助CE项。** Reference KL、entropy和非法动作惩罚继续按共同基线设置。

| 项目 | 两个尺寸共用的固定值 |
| --- | --- |
| 初始化 | 官方Instruct基模，固定revision；全参数训练，无Route-SFT/LoRA |
| 训练池 | 官方环境中3553个可用ALFWorld train任务 |
| 每update | 16任务 × 8条rollout = 128条episode |
| RLSD | 初始lambda0.5；warmdown100；weight clip epsilon0.2；skill_all=false |
| 学习率 / PPO | 1e-6；mini-batch256；每卡micro32；PPO epoch1；token-mean |
| Reference KL / entropy | 0.01 low_var_kl；entropy coefficient0.001 |
| 非法动作惩罚 | 0.1，沿用原生实现 |
| 环境与采样 | max steps50；history2；prompt2048；response512；训练temperature1 |
| 更新与验证 | 150 updates；初始和每10步验证固定seen140 |
| 保存 | 每20步和最终150；最近两份完整断点，含最终150 |

lambda按实际update编号计算：update1=0.495、50=0.25、100及以后=0。100步后仍训练到150，但RLSD重加权已退回普通GRPO；**这是上游公开RLSD脚本的warmdown100配方**，没有擅自延长、取消或改成SDAR的beta5。配置见 [experiment.json](configs/experiment.json)。

## 2. 干净服务器准备

Linux x86-64、Python3.12、**4×A100 80GB**、兼容CUDA12.8的驱动。建议32个可用逻辑CPU、192–256GiB内存、250–300GiB空闲磁盘，供两个尺寸的环境/模型、最近两份完整断点和最终导出使用；这是部署预算，不是实测最低要求。

```bash
git clone https://github.com/leiye302/rlsd-alfworld-qwen25.git
cd rlsd-alfworld-qwen25
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

python src/setup.py --framework
python -m pip install -r configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/sdar -c configs/requirements.txt
python -m pip check

cp .env.example .env
# 编辑获配的四张GPU、CPU数量和ALFWORLD_DATA。
set -a
source .env
set +a

# 阅读模型LICENSE后下载固定版本。3B是Qwen Research License。
python src/setup.py --download-model both
alfworld-download
python src/setup.py --verify
python -m pytest tests -q
```

源码、skill、基模和ALFWorld资源均从公开来源准备，**不依赖作者电脑、私有服务器或仓库外私有文件**。模型版本见 [models.json](configs/models.json)。`flash-attn`没有匹配wheel时需要CUDA Toolkit / nvcc编译；上游依赖安装使用本仓库约束，防止torch/vLLM被升级。可以使用自己配置的HF镜像/缓存，但模型ID和revision不变。

## 3. 四卡开跑

配置预览不占GPU、不下载模型或游戏：

```bash
bash scripts/train.sh --size 1.5b --dry-run
bash scripts/train.sh --size 3b --dry-run
```

确认四张卡属于自己的作业。在tmux或调度器的持久会话里运行：

```bash
# 1.5B150 → 导出；然后3B150 → 导出。每次使用RLSD_GPUS指定的全部四卡。
bash scripts/run_both.sh
```

也可以只跑某个尺寸：

```bash
bash scripts/train.sh --size 1.5b
bash scripts/train.sh --size 3b
```

四卡机器不要同时启动两种尺寸；不是每个尺寸只分两卡。不需要外层torchrun，SDAR/Ray负责四个FSDP/vLLM worker。各作业独立本地Ray集群和短临时目录；不要执行整机ray stop。首次运行检查真实rollout、非空奖励、Teacher gap、rlsd/lambda、有限grad norm和Actor更新，再确认完整存盘。

## 4. 评测与结果

使用和 [SDAR对比仓库](https://github.com/leiye302/sdar-alfworld-qwen25)、[RETHINK主实验仓库](https://github.com/leiye302/rethink-alfworld-qwen25-1.5b) **同一份seen140文件**：

- ALFWorld `valid_seen`，140个固定任务；环境seed=`2026 + task_index`，固定游戏文件SHA与顺序。
- 一任务一条rollout，最多50步，history2，prompt2048/response512。
- temperature0.4、top_p1、top_k=-1；do_sample=true。
- 初始和每10次更新验证；最终使用update150，不用验证成绩挑最佳模型替代最终模型。
- 评测Student没有Teacher skill、Analyzer或hint；记录成功数/140与reasoning、格式及原生action-valid率。

任务见 [seen140_tasks.json](configs/seen140_tasks.json)。这是主实验的固定seen140比较协议，不宣称等于RLSD原论文任务或SDAR原始128条验证清单。保留官方动作解析器；strict_protocol_rate只做遥测，不改变执行或奖励。如果其他实验改过执行解析器，严格同协议比较需统一评测执行器。

```text
runs/rlsd_1.5b/                         # 3B为rlsd_3b
  binding.json / resolved_config.yaml
  metrics.jsonl                        # 全部原生训练/验证指标，含rlsd/lambda和gap
  training_pool.json
  evaluation/update_000/summary.json    # 成功数/140、reasoning、格式与逐任务结果
  evaluation/update_000/decisions.json  # 每步生成文本与token IDs
  evaluation/update_150/summary.json
  checkpoints/global_step_150/          # 全rank模型、optimizer、scheduler/RNG、dataloader
  hf150/                               # run_both自动导出的最终推理模型
```

手工导出/重新评测：

```bash
python src/run.py export --size 1.5b
bash scripts/eval.sh --size 1.5b --model "$PWD/runs/rlsd_1.5b/hf150" \
  --run-dir "$PWD/runs/rlsd_1.5b_final_eval"
# 3B相应修改size和路径；均在四卡空闲后执行。
python src/report.py
```

训练中已自动进行真实环境验证；重新评测不是完成训练的前置条件。汇总脚本从实际结果生成 `results/seen140.csv` 和PNG曲线，没有预填成绩。

## 5. 中断恢复与工程改动

确认旧进程退出、断点完整后：

```bash
bash scripts/train.sh --size 1.5b --resume
# 或 --size 3b --resume
```

使用上游原生续训机制；校验四rank的模型/optimizer/extra_state和dataloader文件，以及模型/配方/任务集绑定。TextWorld环境游戏迭代器未保存，所以不承诺中断续跑与连续训练逐位相同。首个20步断点前中断不能冒充已可恢复；新运行不覆盖已有目录。

相对上游RLSD 8卡脚本：适配4卡/TP1、vLLM显存0.3/eager/释放cache、官方fused torch log-prob后端；每16个独立原生环境共用一个Ray actor，以降低主机内存。每update任务/轨迹数、GRPO和RLSD公式、原生Teacher/skill、prompt、奖励与动作解析不变。验证改为共同seen140/每10步，保存改为每20步及150。

补丁只更改环境分组/固定验证和 `main_rlsd.py` 的trainer导入；`rlsd_eval.FixedSeenTrainer` 只增加评测/日志，直接调用原生 `RLSDRayTrainer.fit`。核心算法、Actor、Teacher、skill、reward与上游文件SHA在 [upstream.json](configs/upstream.json) 中锁定；没有带入RETHINK或SDAR loss的补丁。

## 6. 验证范围与许可

检查包含干净检出准备、两种尺寸配置、优势符号保留、权重裁剪、Teacher/Student detach、零gap等价GRPO、lambda warmdown、环境分组seed/顺序、完整断点与逐任务评测。GitHub Actions在全新Linux检出执行CPU检查。**尚未实测四卡A100 CUDA/NCCL/vLLM训练，不附带训练成绩。**

包装代码和所复用的SDAR源码为Apache-2.0。1.5B模型为Apache-2.0，3B为Qwen Research License；模型权重与ALFWorld资源单独遵循来源许可。来源与差异见 [THIRD_PARTY.md](THIRD_PARTY.md)。
