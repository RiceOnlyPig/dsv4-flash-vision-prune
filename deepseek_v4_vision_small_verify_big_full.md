
# DeepSeek-V4-Flash-Vision 以小验大全流程

适用环境：node-29-118、容器 l00946971-dsv4-124copy、vLLM 0.27.1+empty、vllm-ascend a0f4947、CountBenchQA 491 条。

## 1. 目标和验收分级

以小验大不是只把模型变小，而是用小模型快速验证大模型的推理和训练链路。

Level 1：多模态推理闭环

- 模型和视觉权重可以加载；
- 图片可以正确转为视觉 token；
- vLLM 服务可以生成回答；
- CountBenchQA 可以跑完；
- 可以计算 Accuracy，输出不会大量截断或循环。

Level 2：RL 训练闭环

- 训练 forward 和 loss 正常；
- backward 梯度有限；
- optimizer 确实更新参数；
- 新参数可以 Resharding 并同步到 rollout；
- 更新后的模型可以继续生成；
- reward 趋势可解释。

本次已经基本完成 Level 1 和物理裁剪闭环，Level 2 的训练/RL 仍需短跑验证。

## 2. 先理解几个概念

Tokenizer 把文字和特殊标记转成 token ID，模型接收的是 ID，不是字符串。

Chat Template 把 role、文本、思考标记和结束标记拼成模型需要的 prompt。DeepSeek-V4-Vision 的格式可能由 encoding/ 和 inference/ 提供，不一定在 tokenizer_config.json 里内嵌 chat_template。不要直接套用 Qwen/Kimi 模板。

MoE 模型有很多专家，Router 为每个 token 选择少数专家。当前模型每个 token 激活 6 个专家，后 40 层原本有 256 个候选专家。

Megatron 负责训练并行，vLLM 负责推理；Bridge 负责两者之间的结构、权重和并行布局适配。

## 3. 环境和目录契约

进入容器：

~~~bash
sudo docker start l00946971-dsv4-124copy
sudo docker exec -it l00946971-dsv4-124copy /bin/bash
~~~

主要路径：

~~~bash
MODEL_ORIG=/mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16
SCRIPT_DIR=/mnt/share/l00946971/project/scripts
DATA=/mnt/share/l00946971/CountBenchQA/CountBenchQA/data/test-00000-of-00001.parquet
OUT=/mnt/share/l00946971/outputs
~~~

记录版本：

~~~bash
python3 - <<'PY'
import torch, vllm, vllm_ascend
print(torch.__version__)
print(vllm.__version__, vllm.__file__)
print(vllm_ascend.__file__)
PY
git -C /vllm-workspace/vllm rev-parse HEAD
git -C /vllm-workspace/vllm-ascend rev-parse HEAD
~~~

镜像迁移是 commit、save、load 三步。bind mount 的 /mnt、/data、驱动、运行进程和环境变量通常不包含在镜像内，目标机器要重新挂载和配置。

## 4. 原始模型多模态基线

CountBenchQA 含图片。infer_countgeoqa.py 必须读取 images，将图片字节编码为 Base64，并用 image_url 内容块请求 vLLM。

先建立同口径的完整模型 baseline。所有模型都要固定相同的 seed、temperature、top-p、max-tokens、数据集和评分脚本。

启动原始模型并采集 TopK：

~~~bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOPK_SAVE_PATH=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer

vllm serve /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16 \
  --served-model-name auto \
  --max-num-seqs 64 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --port 20007 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --enforce-eager
~~~

采集 TopK 时使用 enforce-eager，避免动态写日志与图捕获/回放冲突。服务就绪后检查：

~~~bash
curl -sS http://127.0.0.1:20007/v1/models
~~~

跑完整数据集：

~~~bash
mkdir -p /mnt/share/l00946971/outputs
python3 /mnt/share/l00946971/project/scripts/infer/infer_countgeoqa.py \
  --parquet /mnt/share/l00946971/CountBenchQA/CountBenchQA/data/test-00000-of-00001.parquet \
  --base-url http://127.0.0.1:20007/v1 \
  --model auto --limit 491 --temperature 0 --top-p 1.0 --seed 0 \
  --max-tokens 512 --concurrency 8 --retries 0 \
  --output /mnt/share/l00946971/outputs/infer_countbenchqa_baseline_118.jsonl
~~~

评分：

~~~bash
python3 /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer/score_countbenchqa.py \
  --predictions /mnt/share/l00946971/outputs/infer_countbenchqa_baseline_118.jsonl \
  --output /mnt/share/l00946971/outputs/countbenchqa_scores_baseline_118.jsonl \
  --verbose 10
~~~

注意：HTTP 200 只表示接口成功，不表示答案正确。

## 5. TopK 采集和统计

TopK 文件形如 layer_3_0.txt，内容每行是一个 token 在该层选中的 6 个专家 ID。通常有 43 层 × 16 rank = 688 个文件。前三层是 hash 路由，不能裁剪；统计第 3 到 42 层，共 40 层，因此会合并 640 个 rank 文件。

检查日志：

~~~bash
find /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer -type f -name 'layer_*.txt' | wc -l
find /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer -type f -name 'layer_*.txt' -size 0 | wc -l
~~~

stat_moe.py 会先按真实层号合并所有 rank，再取全局 TopN，不是每个 rank 单独取 TopN。

生成 Top32：

~~~bash
python3 /mnt/share/l00946971/project/scripts/prune/stat_moe.py \
  --top-k 32 \
  --log-dir /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer \
  --start-layer 3 --end-layer 42 \
  --output /mnt/share/l00946971/project/scripts/logs/top32_experts.json
~~~

生成 Top64 时只需把 top-k 和输出文件名改为 64。

## 6. 物理裁剪

TopK 统计只产生专家列表，不会减少参数。物理裁剪才会删除权重。

先预检查：

~~~bash
python3 /mnt/share/l00946971/project/scripts/prune/prune_moe_experts_dsv4.py \
  --model-dir /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16 \
  --topk-json /mnt/share/l00946971/project/scripts/logs/top32_experts.json \
  --output-dir /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16-top32-pruned \
  --top-k 32 --skip-layers 0,1,2
~~~

预检查不会写模型。确认目标专家数为 32、完整保留层为 0/1/2、裁剪层数为 40 后，再加 execute：

~~~bash
python3 /mnt/share/l00946971/project/scripts/prune/prune_moe_experts_dsv4.py \
  --model-dir /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16 \
  --topk-json /mnt/share/l00946971/project/scripts/logs/top32_experts.json \
  --output-dir /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16-top32-pruned \
  --top-k 32 --skip-layers 0,1,2 --execute
~~~

脚本要同时处理专家权重、Router/gate 维度、专家 ID 重映射、MTP 权重以及 config/index。前三层保留 256 专家；MTP 辅助层要使用兼容的专家列表。

检查配置：

~~~bash
python3 - <<'PY'
import json
p='/mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16-top32-pruned/config.json'
c=json.load(open(p))
for k in ['num_hidden_layers','n_routed_experts','num_experts_per_tok','num_nextn_predict_layers']:
    print(k, c.get(k))
PY
~~~

Top32 应为 43、32、6、3。

## 7. 参数量和裁剪后验收

统计精确参数量：

~~~bash
python3 - <<'PY'
from pathlib import Path
from safetensors import safe_open
d=Path('/mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16-top32-pruned')
total=0
shards=0
for p in sorted(d.glob('*.safetensors')):
    with safe_open(str(p), framework='pt', device='cpu') as f:
        for n in f.keys():
            size=1
            for x in f.get_slice(n).get_shape():
                size *= x
            total += size
    shards += 1
print('读取权重分片:', shards)
print('精确参数量:', total)
print('参数规模: %.3f B' % (total/1e9))
PY
~~~

本次实际结果：

~~~text
Top16：44.893B，Accuracy 60.2851%
Top32：62.210B，Accuracy 81.2627%
~~~

启动裁剪模型前：

~~~bash
unset TOPK_SAVE_PATH
unset EXPERT_MASK_PATH
~~~

启动 Top32：

~~~bash
vllm serve /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16-top32-pruned \
  --served-model-name auto --max-num-seqs 64 --trust-remote-code \
  --tokenizer-mode deepseek_v4 --tensor-parallel-size 8 --data-parallel-size 2 \
  --port 20007 --enable-expert-parallel --gpu-memory-utilization 0.95 \
  --async-scheduling --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --max-model-len 8192 --max-num-batched-tokens 8192 --enforce-eager
~~~

跑评测和评分时必须使用新的输出文件名，避免覆盖 baseline。

## 8. SFT/LoRA 与 RL

SFT 是监督微调，用标注数据调整模型行为；它不会减少基础模型参数。LoRA/QLoRA 只减少训练时更新的参数，也不会删除基础模型专家。

如果裁剪模型已经能生成可评分答案，reward 正常，RL loss、backward 和 optimizer 正常，可以跳过 SFT 直接进入 RL。只有输出格式差、reward 恒为零或训练不稳定时，才考虑 SFT/LoRA 热身。

RL 的闭环是：

~~~text
rollout 生成
→ reward
→ training forward/loss
→ backward
→ optimizer 更新 θ0→θ1
→ Resharding
→ 同步 θ1 到 rollout
→ 用新参数继续生成
~~~

先跑 1～3 step 验证链路，再跑 20～50 step 看趋势，最后才做 200 step 或长稳。检查 reward、loss、grad norm、参数差异和同步后的新回答。

## 9. 多模态 RL 额外检查

推理时图片能够生成答案，不代表多模态训练已经打通。训练还要确认：

- raw token 和视觉展开 token 数量一致；
- response 起始位置没有整体偏移；
- logits 和 label 对齐；
- loss 只落在 response 区间；
- 路由日志 token 位置能映射回 request/sample；
- 训练和 rollout 使用同一 tokenizer、encoding、模型代码。

## 10. 当前结论和验收清单

当前已完成：

~~~text
多模态 vLLM 推理：已完成
TopK 专家采集和全局统计：已完成
物理 Top16/Top32 裁剪：已完成
Top16：44.893B，60.2851%
Top32：62.210B，81.2627%
~~~

尚需完成：

~~~text
完整模型 491 条正式 baseline
Top64 与 baseline 对比
训练 forward/loss/backward/optimizer
Resharding 和 rollout 权重同步
多模态 token、label、loss、路由对齐
RL 短跑和长稳
~~~

最终验收原则：

- [ ] 推理服务正常不等于准确率合格；
- [ ] TopK 日志必须覆盖目标层和所有 rank；
- [ ] 物理裁剪必须先预检查，再 execute；
- [ ] 裁剪后 config、index、MTP 和权重必须配套；
- [ ] 所有模型使用同一数据集和解码参数；
- [ ] RL 必须观察到 reward、梯度和参数更新。

一句话总结：以小验大是用资源可控的小模型，先证明多模态推理和 RL 训练链路都能工作，再把结论迁移到多机全层减专家模型。
## 附录 A：脚本重点速查

### A.1 推理脚本

文件：/mnt/share/l00946971/project/scripts/infer/infer_countgeoqa.py

读取 parquet 中的 prompt 和 images，把图片转 Base64，发送多模态请求，保存预测 jsonl。它不裁剪模型；TopK 是否落盘由服务端 patch 和 TOPK_SAVE_PATH 决定。

### A.2 TopK 采集 patch

主要文件：

~~~text
vllm_ascend/ops/fused_moe/routed_experts.py
vllm_ascend/ops/fused_moe/fused_moe.py
~~~

在 Router 选出专家后、log2phy 映射前保存逻辑专家 ID，并传递层号。model.py 和 moe_comm_method.py 主要保证裁剪后专家数变化时的加载、路由和通信兼容。

### A.3 统计脚本

文件：/mnt/share/l00946971/project/scripts/prune/stat_moe.py

它把所有 rank 的同一层日志合并后统计全局 TopN，输出 topN_experts.json。它不修改模型参数。

### A.4 物理裁剪脚本

文件：/mnt/share/l00946971/project/scripts/prune/prune_moe_experts_dsv4.py

它读取 topN_experts.json，裁剪专家权重、Router、专家 ID 和 MTP，并更新 config/index。默认只预检查，只有加 --execute 才会写出新模型。

### A.5 评分脚本

文件：/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer/score_countbenchqa.py

读取预测 jsonl，提取最终数字答案并计算 Accuracy。HTTP 200 不等于答案正确，最终效果以此脚本为准。

### A.6 一句话区分

~~~text
infer_countgeoqa.py：发请求
TopK patch：记专家
stat_moe.py：选名单
prune_moe_experts_dsv4.py：删权重
score_countbenchqa.py：算准确率
~~~

## 附录 B：Kimi-K3 从数据集输入到模板字符串

假设 parquet 中的一条样本是：

~~~python
row = {
    "prompt": "<image>\\nHow many dogs are in the image?",
    "images": ["/data/dog.jpg"]
}
~~~

### B.1 原始样本到 messages

调用 `_countbenchqa_messages(row, text_only=False)` 后，先移除 prompt 中的 `<image>` 占位符，再读取图片并编码成 Base64 Data URL：

~~~python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,..."
                }
            },
            {
                "type": "text",
                "text": "\\nHow many dogs are in the image?"
            }
        ]
    }
]
~~~

这一步负责传输真实图片，但还没有执行 Chat Template。

### B.2 图片尺寸到 K3 图片占位符

`render_input()` 读取图片尺寸，调用 `_navit_new_size()` 进行 K3 NaviT 缩放。假设缩放后为 448×336，再调用 `_make_image_prompt()`：

~~~text
<|media_begin|>image 448x336<|media_content|><|media_pad|><|media_end|>
~~~

这段文字只表示“图片在这里”，不包含图片像素。真实图片仍在上一步的 Base64 或 `multi_modal_data` 字段中。

### B.3 Chat Template 把 messages 变成字符串

核心调用是：

~~~python
rendered = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    image_prompts=[image_prompt]
)
~~~

内部会按照 K3 的模板依次拼接：

~~~text
用户角色标记
图片特殊占位符
文本问题
助手开始生成标记
~~~

渲染结果大致类似下面的字符串（真实角色标记由 K3 tokenizer 决定）：

~~~text
<|user|>
<|media_begin|>image 448x336<|media_content|><|media_pad|><|media_end|>
How many dogs are in the image?
<|assistant|>
~~~

`add_generation_prompt=True` 会添加 assistant 开始标记，告诉模型接下来应该生成回答。

### B.4 字符串再变成 token ID

如果设置：

~~~python
tokenize=True
~~~

模板生成的字符串会继续被 tokenizer 转成 token ID：

~~~text
结构化 messages
    ↓ Chat Template
K3 格式的字符串 prompt
    ↓ Tokenizer
token ID
~~~

因此四个环节的职责是：

~~~text
_images_messages()：真实图片 → Base64 + image_url
_navit_new_size()：计算 K3 图片缩放尺寸
_make_image_prompt()：生成 K3 图片占位符
apply_chat_template()：messages → K3 格式字符串
Tokenizer：字符串 → token ID
视觉处理器：图片 → 视觉特征/token
~~~

K3 的 `render_input()` 主要用于复刻 RL rollout_dump 中的输入格式；实际图片视觉特征仍由服务端视觉处理器生成。DeepSeek-V4-Vision 应使用自己的 encoding/inference 协议，不能直接复制 K3 的 media 标记。

## 附录 C：论文与最新方法

关于专家频次裁剪、Router/Gate/激活贡献评分、层自适应预算、动态跳过、SFT/KD 恢复训练以及 2025～2026 最新方法，见配套文献教学稿：

[以小验大与 MoE 专家裁剪：论文、方法、效果和实践教学](yixiaoyanda_moe_pruning_literature.md)

## 附录 D：低于 45B 且尽量保性能的裁剪方案

当前实测：

~~~text
Top16：44.893B，Accuracy 60.2851%
Top32：62.210B，Accuracy 81.2627%
~~~

Top16 虽然满足领导提出的“低于 45B”，但只比 45B 少约 0.1B，安全余量很小，而且性能下降明显。因此不要把“每层固定 16 个专家”当作最终方案。

### D.1 先区分两个约束

- 如果硬约束是参数量必须低于 45B，量化只能降低存储和显存，不能把 62B 参数变成 45B 参数；
- 如果真正约束是单机显存，Top32 加 INT8/INT4 可能比 Top16 的质量更好，但报告中必须把“参数量”和“存储占用”分开写。

### D.2 推荐的首轮预算

前三个 hash 层保持完整，后 40 层不再统一 Top16，而是使用层级预算。先用 mask 做小规模敏感度实验，把层分成高、中、低三类：

~~~text
高敏感层：Top20
中敏感层：Top16
低敏感层：Top12
~~~

一个保守的试验分配是：8 个高敏感层、16 个中敏感层、16 个低敏感层：

~~~text
8×20 + 16×16 + 16×12 = 608 个专家槽位
~~~

这比统一 Top16 的 640 个槽位还少，预计参数规模约 44B 左右，并留出比当前 Top16 更充足的 45B 安全余量。具体参数量必须由裁剪脚本重新统计，不能只按线性估算。

如果评测后仍有余量，可以把 32 个槽位优先加回高敏感层，回到总计 640 个槽位，而不是平均加到所有层。

### D.3 如何判断高敏感层

不要只按专家命中频次判断。对每层分别做 mask 实验，记录：

~~~text
该层从 256 → Top8/12/16/20/24 后的 Accuracy 变化
Reward 是否为零
循环输出和截断比例
图片 token 与文本 token 的错误率
~~~

优先把专家槽位给“裁少一点就明显掉分”的层；对多个预算都不敏感的层才使用 Top8/12。这样是在固定总预算下，把参数从不重要的层转移给关键层。

### D.4 专家名单也要升级

当前 TopK 日志只能做频次统计。下一版建议把专家评分改成：

~~~text
频次分数 + Router/Gate 权重 + gate × 专家输出范数
~~~

尤其要保护“低频但高贡献”的专家；这类专家可能负责停止、视觉 token 或关键推理步骤。专家数量不变，也可以通过更好的名单提升 Top16 的效果。

### D.5 裁剪后再决定是否 SFT/KD

如果层级预算和专家评分后仍低于完整模型：

~~~text
完整模型作为 teacher
→ 多模态监督数据做短 SFT/LoRA
→ 蒸馏 teacher 的 logits/答案分布
→ 再检查 reward、梯度和参数更新
~~~

SFT/LoRA/KD 不会减少基础模型参数，它们的作用是恢复裁剪造成的能力损失；参数量目标应以裁剪后的 base model 为准。不要在一个只能得到 60% 准确率、reward 还不稳定的模型上直接开始 RL。
