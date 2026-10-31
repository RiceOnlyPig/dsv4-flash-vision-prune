# DeepSeek-V4-Flash-Vision 以小验大：从多模态输入到 TopK 采集

> 日期：2026-09-10  
> 场景：Ascend 单机 16 卡，vLLM + vllm-ascend，CountBenchQA 多模态数据集  
> 当前阶段：多模态输入、运行环境和 TopK 采集链路均已跑通；下一步是统计后 40 层的高频专家并裁剪模型。

## 1. 今天真正解决的四个难点

今天不是单纯“加一个 TopK 打印”。真正的工作顺序是：

```text
让 DeepSeek-V4 正确理解图文请求
             ↓
固定并迁移一套真正能跑的镜像环境
             ↓
给新版 vllm-ascend 适配 TopK/裁剪 patch
             ↓
关闭图模式，允许逐 token 把路由结果写入文件
             ↓
用 CountBenchQA 采集专家日志
```

### 难点一：`chat_template.jinja`、DeepSeek encoding 和多模态输入

#### 1.1 Chat API 收到的不是模型可以直接计算的输入

OpenAI 风格的 Chat 请求长这样：

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请数一下图片里有几个物体"}
  ]
}
```

但模型真正接收的是一串 token ID。中间必须有人把角色、文本、轮次结束标记等拼成模型训练时使用的格式：

```text
messages JSON
    ↓ 格式转换
模型约定的 prompt
    ↓ tokenizer
token IDs
```

`chat_template.jinja` 就是一套“聊天消息排版规则”。它规定：

- system、user、assistant 分别怎样编码；
- 每轮前后加入什么特殊 token；
- 是否添加 assistant 开始生成的提示；
- 多轮对话怎样连接。

因此它不是美化文本的模板，而是**模型输入协议的一部分**。模板不匹配时，即使服务能返回，也可能出现答非所问、无法停止或者精度明显下降。

#### 1.2 为什么当时 `/v1/chat/completions` 返回 400

当时权重目录中的 `tokenizer_config.json` 没有内嵌 `chat_template`，启动命令也没有提供 `--chat-template`。Chat Completions 路线会尝试执行类似：

```python
tokenizer.apply_chat_template(messages)
```

Transformers 4.44 以后不再允许偷偷使用一个默认模板，因此服务明确报错：

```text
default chat template is no longer allowed
```

这不是模型权重损坏，也不是 NPU 推理失败，而是“JSON 聊天消息还不知道该怎样变成模型 prompt”。

#### 1.3 为什么 `/v1/completions` 的单字符串 curl 能工作

`/v1/completions` 接收的已经是 `prompt`：

```json
{"prompt": "San Francisco is a"}
```

它不需要先把 `messages` 转成一段文本，所以可以绕过 Chat Template 缺失问题。这只能证明纯文本生成链路可用，不能证明多模态 Chat 编码已经正确。

#### 1.4 DeepSeek 为什么没有普通的 Jinja，却提供 `encoding/`

这套 DeepSeek-V4-Flash-Vision 仓库把“提示编码”和“模型推理”有意分开：

```text
encoding/    负责把 OpenAI 风格 JSON 或紧凑文本协议转换成正确 prompt/token
inference/   负责加载模型并执行推理
```

官方 encoding 同时理解：

- OpenAI 风格的 `text`、`image_url` 内容块；
- 类似 `<image>path</image>` 的紧凑表示；
- DeepSeek 自己的特殊 token 和图像占位规则。

这相当于 DeepSeek 没把完整输入规则塞进一个通用 `chat_template.jinja`，而是用专用编码器完成更复杂的转换。多模态不仅要拼 user/assistant 角色，还要把图片放入视觉处理流程，所以仅靠从 Qwen 或 Kimi 复制一个 Jinja 模板并不可靠。

本次正确路线是：

```text
infer_countgeoqa.py
  读取 parquet 的 prompt 和 images
        ↓
  _countbenchqa_messages()
        ↓
  _images_messages() 把图片做成多模态 content block
        ↓
  OpenAI 风格请求发给 vLLM
        ↓
  --tokenizer-mode deepseek_v4 调用 DeepSeek 专用 encoding
        ↓
  得到模型需要的文本 token + 图像输入
```

所以 `infer_countgeoqa.py` 并不是简单地把整行转成字符串。`normalize_prompt()` 处理文字，`_images_messages()` 负责组织图片内容块，服务端 DeepSeek encoding 再把协议转换为模型输入。

需要牢记：

- `chat_template` 解决“聊天轮次怎样排版”。
- 多模态 encoding 还要解决“图片怎样进入视觉编码器并与文本 token 对齐”。
- 某个模型的 Jinja 不能随便借给另一个模型。
- 这次补齐 `encoding/` 并使用 `--tokenizer-mode deepseek_v4`，才是针对该模型的正确方向。

### 难点二：环境一致性和镜像迁移

#### 1.5 为什么 124 能跑，118 看起来版本相同却不能跑

`pip list` 看起来一样，不代表运行环境真的一样。还可能不同的内容包括：

- vLLM 和 vllm-ascend 的 Git commit；
- editable install 实际指向的源码目录；
- 源码目录里的未提交 patch；
- CANN、驱动、固件和 NPU 算子；
- 环境变量、`LD_LIBRARY_PATH` 和启动参数；
- 模型目录内容和挂载路径。

本次两台机器的 vllm-ascend 版本字符串曾经接近，但源码 commit、`model.py` 内容和工作区修改并不完全一致。因此最省时间的方式是把已跑通容器固化为镜像。

#### 1.6 `docker commit → save → load` 分别做什么

```text
正在运行且已调好的容器
        ↓ docker commit
生成本机 Docker 镜像
        ↓ docker save
生成可复制的 tar 文件
        ↓ 传到目标机器
        ↓ docker load
目标机器注册同一个镜像
        ↓ docker run
用相同镜像创建新容器
```

第一步，在 124 上把能运行的容器固化：

```bash
docker commit \
  l00946971-dsv4 \
  l00946971/dsv4-vision:124-working-20260910
```

这里很像 Git 的命名，但它不是 Git commit：Docker commit 是把容器可写层中的文件系统变化做成一个镜像快照，不会生成适合代码审查的逐行历史。

第二步，将镜像保存成 tar：

```bash
mkdir -p /mnt/share/l00946971/images

docker save \
  -o /mnt/share/l00946971/images/dsv4-vision-124-working-20260910.tar \
  l00946971/dsv4-vision:124-working-20260910
```

第三步，在 118 上加载：

```bash
docker load \
  -i /mnt/share/l00946971/images/dsv4-vision-124-working-20260910.tar
```

成功标志是：

```text
Loaded image: l00946971/dsv4-vision:124-working-20260910
```

第四步检查镜像：

```bash
docker images | grep 'l00946971/dsv4-vision'

docker image inspect \
  l00946971/dsv4-vision:124-working-20260910 \
  --format '{{.Id}} {{.RepoTags}}'
```

#### 1.7 镜像不会帮我们保存什么

下面这些内容不能简单理解为已经被 `docker commit` 打进镜像：

- `-v /mnt:/mnt`、`-v /data:/data` 等 bind mount 中的数据；
- 宿主机的 Ascend driver 和 firmware；
- 正在运行的 vLLM 进程状态；
- 当前 shell 临时 export、但未写入启动脚本的变量；
- 启动时使用的端口、设备和挂载参数。

所以迁移后仍然要用正确的 `docker run` 挂载 `/mnt`、driver、firmware，并重新执行启动脚本。

镜像解决的是“Python 包、源码和容器内依赖一致”；挂载和启动脚本解决的是“宿主硬件、模型数据和运行参数一致”。

### 难点三：几个 patch 到底分别在做什么

这次容易混淆，是因为 patch 实际包含了两类功能：

```text
A. 采集阶段：从完整模型中识别并保存 TopK 专家
B. 裁剪阶段：让前 3 层 256 专家、后 40 层 64 专家的小模型能够加载和通信
```

它们有关联，但不是每个 patch 都是“识别 TopK 必需”。

| 文件 | 作用 | 识别 TopK 是否直接需要 | 后续裁剪模型是否需要 |
|---|---|---:|---:|
| `routed_experts.py` | 在 Router 选完专家后保存 `topk_ids` | 是，核心 | 间接相关 |
| `fused_moe.py` 中 `set_topk_layer_name` | 将 `layer_name` 传给记录器，确定是第几层 | 是 | 否 |
| `fused_moe.py` 中逐层通信切换 | 每一层选择与专家数匹配的通信对象 | 否 | 是 |
| `moe_comm_method.py` | 将全局通信配置改为按 `layer_name` 保存 | 否 | 是 |
| `deepseek_v4/model.py` | 前 3 层仍建 256 专家，后层按裁剪配置建 64 专家，并兼容缺失权重 | 否 | 是 |

#### 1.8 `routed_experts.py`：真正把 TopK 写出来

它做四件核心事情：

1. 读取 `TOPK_SAVE_PATH`。
2. 取得当前 global rank，防止 16 个 worker 写同一个文件。
3. 调用 Router 得到每个 token 的 `topk_ids`。
4. 在 `log2phy` 前把逻辑专家 ID 写入 `layer_<层号>_<rank>.txt`。

核心执行顺序：

```text
router._select_experts()
        ↓
得到每个 token 的 Top6 逻辑专家 ID
        ↓
保存到 CPU 文本文件
        ↓
log2phy 映射为设备物理专家
        ↓
分发 token，执行专家 FFN
```

如果没有这个 patch，设置 `TOPK_SAVE_PATH` 只是多了一个没人读取的环境变量，不会自动生成任何日志。

#### 1.9 `fused_moe.py`：告诉记录器“这是第几层”

`routed_experts.py` 能看到专家 ID，但不一定天然知道自己属于 `layers.3` 还是 `layers.42`。因此 `AscendMoERunner` 通过：

```python
self.routed_experts.set_topk_layer_name(self.layer_name)
```

把类似 `model.layers.3.mlp` 的层名传下去。记录器解析数字 `3` 后，才知道应该写入 `layer_3_<rank>.txt`。

没有这部分，最常见的结果不是专家选择错误，而是层号为空、日志不写，或者不同层无法可靠区分。

#### 1.10 `model.py`：解决裁剪后每层专家数不同

原始代码通常假定所有 MoE 层使用同一个 `n_routed_experts`。但目标小模型是：

```text
第 0～2 层：256 专家，hash 路由，不裁剪
第 3～42 层：64 专家，高频裁剪
```

因此模型构建函数必须看当前 `prefix`/层号：前三层按 256 创建，后续层按裁剪后的专家数创建。权重加载时还要跳过小模型中不存在的参数。

这个 patch 不负责生成 TopK 日志，它负责让“根据 TopK 裁出来的模型”以后能够加载。

#### 1.11 `moe_comm_method.py` 与 `fused_moe.py` 通信 patch

MoE 不只是选择专家，还要把 token 发到保存该专家权重的设备。通信对象内部依赖专家数量、EP 划分和每卡专家布局。

原始实现用一份进程级全局通信配置；但裁剪后前三层是 256 专家、后面是 64 专家，一份全局配置无法同时准确描述两种布局。

所以 patch 将索引从：

```python
moe_comm_type
```

改成类似：

```python
(moe_comm_type, layer_name)
```

每一层进入 forward 前再激活自己的通信对象。这个 patch 是为“混合专家数量的小模型运行”准备的，不是读取 TopK 的必要条件。

一句话区分：

```text
routed_experts + 层名传递：负责看见并记录专家
model + 分层通信：负责真正运行裁剪后的专家布局
```

### 难点四：图模式为什么与 TopK 落盘冲突

图模式先录制一套固定计算流程，后续请求直接回放，以减少 Python 调度开销。回放的是计算步骤，不是上一次的结果，所以输入变化时结果仍会变化。

但是 TopK patch 每次都要执行：

```text
NPU Tensor → CPU → Python list → 追加写文件
```

这些是同步、动态和带副作用的操作，不适合被 ACL Graph 捕获；Python 写文件也不会随着设备图回放自动执行。因此出现 stream/capture 错误不是 TopK 算错了，而是调试 I/O 被放进了图录制阶段。

解决方式：

```text
加入 --enforce-eager
删除 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

Eager 会慢一些，但每次真实 token 路由都能进入 Python 保存逻辑。等 TopK 采集结束、恢复纯推理性能测试时，再重新启用图模式。

## 2. 今天最终完成了什么

在解决上面四个难点后，最终完成了“以小验大”专家裁剪方案的采集阶段：

1. 用完整的 DeepSeek-V4-Flash-Vision 模型启动 vLLM 服务。
2. 使用正确的 DeepSeek 多模态编码向服务发送 CountBenchQA。
3. 在 MoE 路由完成后记录每个 token 选中的 6 个专家。
4. 验证 TopK 日志能够按“层号 + worker rank”保存。
5. 为后续裁剪模型准备了分层专家数和分层通信兼容 patch。

已经观察到的有效结果：

- 共生成 `688` 个日志文件，即 `43 层 × 16 个 rank`。
- 文件不是空文件，总大小约 `22 MB`（当时的测试结果）。
- 每行有 6 个专家 ID，例如：

```text
124,143,175,153,79,213
64,235,68,211,190,119
```

- 单条真实请求后，`layer_3_0.txt` 有 105 行，其中 101 种不同组合，说明记录的不再只是启动预热数据。

这证明：**请求已经真实经过 MoE Router，且我们能取得 token 级的专家选择结果。**

## 3. “以小验大”的整体路线

DeepSeek-V4-Flash 有 43 个 MoE 层，每层 256 个专家，每个 token 激活其中 6 个专家。

完整模型覆盖的任务很多，所以需要 256 个专家；但针对一个特定数据集，真正经常被路由到的专家通常只是其中一部分。因此可以：

```text
完整模型跑目标数据集
        ↓
记录每个 token 在各层选择的 Top6 专家
        ↓
统计各层专家出现频次
        ↓
每层保留高频 Top64 专家
        ↓
生成较小模型并重新评测
        ↓
比较准确率、生成稳定性和后续 RL reward 趋势
```

注意：

- 第 0～2 层采用 hash 路由，不能按普通高频统计方式裁剪，先全部保留。
- 第 3～42 层才统计并保留高频 Top64 专家。
- “高频”只代表在当前采样数据上的使用频率高，不天然代表它一定是最优专家，所以裁剪后必须重新评测。

## 4. 当前环境快照

本次跑通环境：

| 项目 | 配置 |
|---|---|
| 机器 | `node-29-118` |
| NPU | 单机 16 卡 |
| vLLM | `0.27.1+empty` |
| vllm-ascend | `0.19.1rc2.dev1913+ga0f4947bb` |
| vLLM 源码 commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| vllm-ascend commit | `a0f4947bbb72942806a8ac3fcdb2772a0ef70ec7` |
| 模型服务名 | `auto` |
| 服务端口 | `20007` |
| 并行配置 | TP=8，DP=2，EP 开启 |
| 数据集 | CountBenchQA，共 491 条 |

`TP=8 × DP=2 = 16` 个 worker 使用的是一台机器的 16 张卡，并不表示用了两台机器。

## 5. 为什么旧 Wiki 的文件在新代码里找不到

旧版 Wiki 修改的是：

```text
vllm_ascend/ops/fused_moe/experts_selector.py
```

但当前版本已经重构，专家路由代码被拆到了：

```text
vllm_ascend/ops/fused_moe/router/
vllm_ascend/ops/fused_moe/routed_experts.py
```

因此不能机械复制旧版 patch。当前版本最合适的记录点是：

```python
AscendRoutedExperts._select_experts()
```

记录动作应位于以下两个动作之间：

```text
self.router._select_experts(...)       # 得到 router 选择的逻辑专家 ID
                 ↓ 在这里保存 TopK
self.log2phy[topk_ids]                 # 转为当前设备上的物理专家 ID
```

这样记录到的是模型 Router 真实选择的**全局逻辑专家 ID**，便于后续按层裁剪权重。

### 5.1 本次代码职责划分

- `routed_experts.py`：TopK 采集的核心，在路由后记录专家 ID。
- `fused_moe.py`：把层名显式传给采集模块，确保日志能正确命名为 `layer_<层号>_<rank>.txt`。
- `model.py`、`moe_comm_method.py`：主要用于后续裁剪模型、异构专家数量或缺失参数的加载兼容，不是“写出 TopK 日志”的核心条件。

也就是说，如果当前目标只是采集 TopK，应先把采集链路验证正确，不要把“能记日志”和“裁剪模型能加载”混成一个问题。

## 6. TopK 保存代码必须满足的条件

保存逻辑至少需要做到：

1. 从环境变量读取 `TOPK_SAVE_PATH`。
2. 自动创建目录，建议使用 `os.makedirs(path, exist_ok=True)`。
3. 取得当前 global rank，避免多个进程同时写同一个文件。
4. 取得当前 MoE 层号。
5. 将 `topk_ids` 展开为二维数组，每行对应一个 token。
6. 每行写 6 个用逗号分隔的逻辑专家 ID。

文件命名约定：

```text
layer_<layer_id>_<global_rank>.txt
```

例如：

```text
layer_3_0.txt
layer_3_1.txt
...
layer_42_15.txt
```

### 6.1 为什么还需要显式传递层名

Router 本身知道“选了哪些专家”，但不一定知道自己属于模型的第几层。如果只记录 `topk_ids` 而拿不到层号，所有层的数据可能混在一起。

因此由模型构建层时把类似下面的 prefix 传下去：

```text
model.layers.3.mlp.experts
```

采集模块再从 prefix 中解析出 `3`，才能得到可靠的 `layer_3_*.txt`。

## 7. 正确启动 vLLM 服务

建议先建立日志目录：

```bash
mkdir -p /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer
```

启动脚本示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TOPK_SAVE_PATH=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer

vllm serve /mnt/share/l00946971/project/models/DeepSeek-V4-Flash-Vision-Exp-bf16 \
  --served-model-name auto \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --port 20007 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 \
  --async-scheduling \
  --mm-processor-cache-gb 0 \
  --additional-config '{"enable_cpu_binding":true}' \
  --enforce-eager
```

TopK 采集阶段不要再加：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

如果默认 `max_model_len=1048576`，KV Cache 会要求至少约 8.55 GiB，而某个 worker 当时只有 6.34 GiB，因此服务无法启动。显式限制到 8192 可以避免为不需要的百万 token 上下文预留 KV Cache。

### 7.1 `--enforce-eager` 的作用

它强制逐次执行算子，不使用 ACL Graph 回放。

这会牺牲一部分推理性能，但允许每次路由后执行这些调试动作：

```text
NPU Tensor → CPU → Python/Numpy → 写日志文件
```

TopK 采集属于调试/分析任务，正确性比吞吐量更重要，所以本阶段应使用 eager 模式。

## 8. 服务启动后的标准操作顺序

### 8.1 等待服务真正就绪

必须等日志出现：

```text
Application startup complete.
API server: HTTP server started
```

然后检查：

```bash
curl -sS http://127.0.0.1:20007/health
curl -sS http://127.0.0.1:20007/v1/models
```

### 8.2 确认服务进程继承了 TopK 环境变量

`TOPK_SAVE_PATH` 必须存在于启动 vLLM 的服务进程环境里，只在另一个终端 `export` 没有作用。

```bash
VLLM_PID=$(pgrep -f 'vllm serve.*20007' | head -n 1)

tr '\0' '\n' < "/proc/$VLLM_PID/environ" \
  | grep '^TOPK_SAVE_PATH='
```

正确输出应为：

```text
TOPK_SAVE_PATH=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer
```

同时确认启动参数：

```bash
tr '\0' ' ' < "/proc/$VLLM_PID/cmdline"
```

应看到 `--enforce-eager`，并且不应看到 `FULL_DECODE_ONLY`。

### 8.3 清理启动预热产生的 TopK

vLLM 启动时会用 dummy token 做模型预热。即使还没有发送 CountBenchQA，请求日志也可能已经存在。

不要直接把这部分数据混入正式统计。服务启动完成后，将它归档：

```bash
TOPK_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer
WARMUP_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer_startup_warmup_$(date +%Y%m%d_%H%M%S)

mv "$TOPK_DIR" "$WARMUP_DIR"
mkdir -p "$TOPK_DIR"
```

服务进程仍然会向原路径写文件，新建同名目录即可，不用重启服务。

本次预热数据的典型特征是：每个文件恰好 72 行，而且很多行完全相同。这种数据不能代表 CountBenchQA。

## 9. 先跑 1 条请求做闭环验证

单条测试不要覆盖正式结果文件：

```bash
mkdir -p /mnt/share/l00946971/outputs

python3 /mnt/share/l00946971/scripts/infer/infer_countgeoqa.py \
  --parquet /mnt/share/l00946971/CountBenchQA/CountBenchQA/data/test-00000-of-00001.parquet \
  --base-url http://127.0.0.1:20007/v1 \
  --model auto \
  --limit 1 \
  --temperature 0 \
  --top-p 1.0 \
  --seed 0 \
  --max-tokens 512 \
  --concurrency 1 \
  --retries 0 \
  --output /mnt/share/l00946971/outputs/infer_countbenchqa_topk_test.jsonl
```

验证日志：

```bash
TOPK_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer

echo '文件数：'
find "$TOPK_DIR" -type f -name 'layer_*_*.txt' | wc -l

echo '空文件数：'
find "$TOPK_DIR" -type f -name 'layer_*_*.txt' -size 0 | wc -l

echo '示例内容：'
head -n 10 "$TOPK_DIR/layer_3_0.txt"

echo '总行数与不同组合数：'
wc -l "$TOPK_DIR/layer_3_0.txt"
sort -u "$TOPK_DIR/layer_3_0.txt" | wc -l
```

本次得到：

```text
105 行
101 个不同组合
```

这是合理的：一条请求不只对应一条路由记录，而是输入 token 和生成 token 都会逐层选择专家。

## 10. 再跑完整 491 条

单条测试验证成功后，再次归档测试日志，创建干净目录：

```bash
TOPK_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer
TEST_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer_single_test_$(date +%Y%m%d_%H%M%S)

mv "$TOPK_DIR" "$TEST_DIR"
mkdir -p "$TOPK_DIR"
```

执行正式采集：

```bash
python3 /mnt/share/l00946971/scripts/infer/infer_countgeoqa.py \
  --parquet /mnt/share/l00946971/CountBenchQA/CountBenchQA/data/test-00000-of-00001.parquet \
  --base-url http://127.0.0.1:20007/v1 \
  --model auto \
  --limit 491 \
  --temperature 0 \
  --top-p 1.0 \
  --seed 0 \
  --max-tokens 512 \
  --concurrency 8 \
  --retries 0 \
  --output /mnt/share/l00946971/outputs/infer_countbenchqa_topk_full_118.jsonl
```

`--max-num-seqs 64` 是服务最多并发处理多少条序列；`--concurrency 8` 是客户端同时发多少条请求。它们都不限制数据集总数，所以 491 条可以分批全部跑完。

## 11. 正式采集后的完整校验

### 11.1 验证推理请求是否成功

```bash
python3 - <<'PY'
import json
from collections import Counter

path = "/mnt/share/l00946971/outputs/infer_countbenchqa_topk_full_118.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]

print("总记录数:", len(rows))
print("HTTP 状态:", Counter(x.get("status_code") for x in rows))
errors = [x.get("error") for x in rows if x.get("error")]
print("错误数:", len(errors))
if errors:
    print("第一个错误:", errors[0])
PY
```

期望：491 条，全部 `status_code=200`，错误数为 0。

### 11.2 验证日志结构

```bash
TOPK_DIR=/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer

find "$TOPK_DIR" -type f -name 'layer_*_*.txt' | wc -l
find "$TOPK_DIR" -type f -name 'layer_*_*.txt' -size 0 | wc -l
du -sh "$TOPK_DIR"
```

在 43 层、16 rank 都记录的情况下，文件数应为：

```text
43 × 16 = 688
```

### 11.3 验证每行是否恰好 6 个合法专家 ID

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("/mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer")
bad = []
rows = 0

for file in root.glob("layer_*_*.txt"):
    with file.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                ids = [int(x) for x in line.split(",")]
            except ValueError:
                bad.append((str(file), lineno, line))
                continue
            if len(ids) != 6 or any(x < 0 or x >= 256 for x in ids):
                bad.append((str(file), lineno, line))

print("非空行数:", rows)
print("异常行数:", len(bad))
for item in bad[:10]:
    print(item)
PY
```

期望异常行数为 0。

## 12. 下一步：统计每层高频 Top64 专家

先寻找项目是否已经带有 Wiki 对应的统计脚本：

```bash
find /mnt/share/l00946971/project \
  -type f \
  \( -name 'stat_moe.py' -o -name '*topk*.py' -o -name '*expert*stat*.py' \) \
  -print
```

旧方案的调用形式通常是：

```bash
python3 scripts/stat_moe.py \
  --top-k 64 \
  --log-dir /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer
```

在运行未知脚本前先执行：

```bash
python3 scripts/stat_moe.py --help
```

确认它至少满足：

1. 按层分别统计，不能把 43 层混在一起。
2. 将一行的 6 个专家全部计数。
3. 输出每层专家 ID、次数和频率。
4. 每层按频次从高到低选前 64 个。
5. 第 0～2 层标记为“不裁剪”。
6. 保存机器可读结果，例如 JSON，而不只是打印到终端。

建议的统计结果格式：

```json
{
  "3": [124, 143, 175, 153, 79, 213],
  "4": [31, 52, 90, 117, 8, 201]
}
```

这里每层实际应包含 64 个专家 ID，上面只作格式示意。

### 12.1 一个容易忽略的统计口径：TP 重复

TP=8 时，同一 DP 副本的 8 个 TP rank 很可能处理同一批 token，因此不同 rank 的路由日志可能完全或高度重复。

先检查同一层各 rank 是否重复：

```bash
sha256sum /mnt/share/l00946971/project/scripts/logs/topk_ids_by_layer/layer_3_*.txt
```

当前布局通常可理解为：

```text
DP0：rank 0～7
DP1：rank 8～15
```

如果一个 DP 组内 8 个文件完全相同：

- 只关心专家排名时，把 16 个文件全合并通常只是等比例重复，不会改变组内排名。
- 关心准确 token 次数和频率时，应每个 DP 组只取一个代表 rank，例如 rank 0 和 rank 8。
- 最稳妥的做法是先确认日志在 TP/EP 下的语义，再决定统计脚本是否去重。

不要看到 688 个文件就默认它们代表 688 份互不重复的数据。

## 13. 如何给模型结果打分

HTTP 200 只代表请求成功，不代表回答正确。

本模型的输出常包含推理过程，例如：

```text
...reasoning...</think>10
```

评分时应从 `</think>` 后面提取最终答案，再与数据集标准答案比较。需要统计的指标至少包括：

- 请求成功率。
- 可解析答案率。
- CountBenchQA 准确率。
- 异常循环输出比例。
- 达到 `max_tokens` 仍未停止的比例。
- 平均输出长度和耗时。

裁剪后不能只看“模型能启动”。至少要比较：

| 指标 | 完整模型 | 裁剪模型 |
|---|---:|---:|
| 服务启动成功 | 是/否 | 是/否 |
| 请求成功率 |  |  |
| CountBenchQA 准确率 |  |  |
| 答案解析率 |  |  |
| 循环/不停止率 |  |  |
| 平均时延 |  |  |
| 显存/内存占用 |  |  |

如果后续用于 RL，还要比较完整模型和小模型的 reward 曲线方向、参数是否能更新、是否能收敛。小模型不要求分数完全相等，但必须保留足够的任务代表性。

## 14. 补充原理和故障解释

### 难点一：为什么 export 了，目录还是没有日志

`export` 只影响当前 shell 以及从它启动的子进程。

如果在终端 A 启动了 vLLM，却在终端 B 执行：

```bash
export TOPK_SAVE_PATH=...
```

已经运行的 vLLM 不会收到这个变量。必须先 export，再从同一个 shell 启动服务，或者把 export 写进启动脚本。

判断标准不是当前终端的 `echo $TOPK_SAVE_PATH`，而是读取服务进程的 `/proc/<pid>/environ`。

### 难点二：为什么服务启动时已经产生 TopK

vLLM 启动会执行 profiling、dummy run、缓存初始化或图预热。dummy token 也会经过 Router，所以保存代码会记录它们。

这些数据证明代码能执行，但不代表目标数据集。正确方式是服务完全启动后归档预热日志，再创建同名空目录，然后才发真实请求。

### 难点三：为什么图模式会报 stream/capture 错误

图模式会先“录制”一段固定的计算流程，之后反复回放这套流程。回放的是计算步骤，不是上一次的计算结果；每次输入不同，输出仍然可以不同。

但录图阶段要求操作可被图捕获。下面这些调试动作会破坏捕获：

```text
NPU → CPU 同步
tolist()/numpy()
Python 文件写入
动态创建目录
```

而且即使某个 Python 写文件动作侥幸出现在录制阶段，它也不会在每次图回放时自动重新执行。

因此 TopK 采集必须使用 `--enforce-eager`。服务重启、模型结构变化、并行配置变化或 shape bucket 变化时，图模式通常需要重新录制；普通请求内容变化一般不要求重录。

### 难点四：为什么要在 `log2phy` 之前保存

Router 先选“逻辑专家 ID”，分布式系统再把逻辑 ID 映射成当前设备上的“物理专家 ID”。

裁剪模型关心的是原模型第 N 层的第 M 个专家，所以应保存逻辑 ID。如果保存物理 ID，换一种 EP/TP 布局后编号可能变化，无法直接定位原权重。

### 难点五：为什么 491 条数据会产生远多于 491 行

491 是请求数。TopK 记录单位是 token：

```text
请求数 × 每条输入/输出 token 数 × MoE 层数 × 分布式记录副本
```

每个 token 在每个 MoE 层又会记录 6 个专家。因此日志行数远大于 491 完全正常。

### 难点六：为什么旧 patch 不能直接覆盖新代码

同一个功能在不同版本中可能被拆分类、改函数签名或移动目录。例如旧版的 `experts_selector.py` 在当前代码中已不存在。

迁移 patch 时要找的是“语义锚点”：

```text
Router 在哪里返回 topk_ids？
逻辑 ID 在哪里被转成物理 ID？
层名从哪里传下来？
```

而不是只寻找同名文件。

### 难点七：为什么改完源码不一定要重新编译

如果修改的是 editable install 下的纯 Python 文件，例如：

```text
/vllm-workspace/vllm-ascend/vllm_ascend/.../*.py
```

通常重启 Python/vLLM 进程即可加载新代码。

如果修改的是 `csrc`、自定义算子、C++ 或编译扩展，才需要重新编译安装。可以用下面命令确认实际导入路径：

```bash
python3 - <<'PY'
import vllm_ascend
print(vllm_ascend.__file__)
PY
```

### 难点八：为什么 200 成功仍不等于“以小验大成功”

200 只说明接口成功返回。真正目标是用更少资源保留足够的行为代表性：

- 路由行为合理。
- 输出能正常停止。
- 任务准确率下降可接受。
- 后续 reward 趋势与完整模型方向一致。
- RL 训练能够更新并表现出收敛趋势。

因此 TopK 只是裁剪依据，不是最终验收结果。

## 15. 常见故障速查

| 现象 | 主要原因 | 处理方式 |
|---|---|---|
| `TOPK_SAVE_PATH` 目录不存在 | 服务进程没继承环境变量、patch 未加载或目录未创建 | 检查 `/proc/$PID/environ`、源码导入路径，先 `mkdir -p` |
| 目录存在但为空 | 保存点没被执行、层号解析失败、服务仍使用旧进程 | grep patch、重启服务、先发 1 条请求 |
| 启动时就有大量相同行 | vLLM dummy/warmup 数据 | 启动完成后归档目录，再跑正式请求 |
| ACL Graph 报 107030/stream capture | 在图捕获过程中执行 NPU→CPU 或文件 I/O | 加 `--enforce-eager`，移除图配置 |
| KV Cache 不足 | 默认最大上下文 1,048,576 太大 | 设置 `--max-model-len 8192` 和 `--max-num-batched-tokens 8192` |
| 491 条全部 400 | Chat Completions 缺少 chat template | 使用 DeepSeek-V4 tokenizer/encoding 路线，或正确的 completions/多模态编码接口 |
| 找不到 `experts_selector.py` | vllm-ascend 版本重构 | 在 `routed_experts.py` 和 `router/` 中找语义锚点 |
| `make_custom_layer()` 参数错误 | 新版 `make_layers` 用 `prefix=` 关键字调用 | 函数签名接受 `prefix`，不要写成不匹配的参数名 |
| 同一层出现 16 份相似日志 | TP/DP/EP 分布式 worker 都在记录 | 检查 hash，统计时明确是否按 TP 去重 |

## 16. 修改和回退建议

改源码前先保存状态和补丁：

```bash
cd /vllm-workspace/vllm-ascend
git status --short
git diff > /mnt/share/l00946971/project/vllm_ascend_topk.patch
```

记录版本：

```bash
git -C /vllm-workspace/vllm rev-parse HEAD
git -C /vllm-workspace/vllm-ascend rev-parse HEAD
```

不要用 `git reset --hard` 清理包含同事改动的工作区。需要回退时，先确认目标文件和已有变更，再用备份文件或经过审核的逆向 patch 恢复。

## 17. 最终检查清单

### 服务端

- [ ] 使用正确的 vLLM/vllm-ascend commit。
- [ ] `TOPK_SAVE_PATH` 在启动服务前 export。
- [ ] 使用 `--enforce-eager`。
- [ ] 没有使用 `FULL_DECODE_ONLY` 图配置。
- [ ] 显式限制 `max_model_len`。
- [ ] 服务 `/health` 正常。

### 数据采集

- [ ] 服务启动完成后已归档 warmup 日志。
- [ ] 先跑 1 条并验证产生真实、非重复的 token 路由。
- [ ] 正式 491 条使用独立输出文件。
- [ ] 491 条请求全部 HTTP 200。
- [ ] TopK 文件非空，每行 6 个 ID，范围是 0～255。
- [ ] 已核对 43 层和 16 rank 的文件覆盖情况。

### 高频专家统计

- [ ] 按层统计，而不是跨层混合。
- [ ] 第 0～2 层不裁剪。
- [ ] 第 3～42 层各选 Top64。
- [ ] 已检查 TP rank 是否重复，统计口径明确。
- [ ] 保存专家列表、原始次数、频率和数据集配置。

### 裁剪后验收

- [ ] 小模型能够加载并启动。
- [ ] 输出能正常停止，无明显循环。
- [ ] CountBenchQA 准确率下降在可接受范围内。
- [ ] 与完整模型使用相同数据、相同解码参数评测。
- [ ] 后续 RL reward 可计算、参数可更新、趋势具有代表性。

## 18. 一句话总结

今天做的事情不是简单地“打印 TopK”，而是建立了一条可验证的数据链：**完整多模态模型接收真实数据 → Router 为每个 token 选择逻辑专家 → 按层、按 rank 落盘 → 去掉启动预热污染 → 校验日志 → 为后 40 层选取高频 Top64 专家。**
