# Teacher 按层共享 HBM 槽

`scripts/local/profile_native_single_gpu.sh` 默认开启：

```bash
TEACHER_LAYER_PIPELINE="${TEACHER_LAYER_PIPELINE:-true}"
TEACHER_LAYER_PIPELINE_MAX_MB="${TEACHER_LAYER_PIPELINE_MAX_MB:-8192}"
```

对应配置为 `actor_rollout_ref.rollout.teacher_layer_pipeline` 和
`teacher_layer_pipeline_max_mb`；通用训练配置默认关闭。开启后替代旧的
`teacher_param_prefetch` 单 shard 预取，`teacher_param_prefetch_max_mb` 不再控制这条路径。
`teacher_forward_overlap` 仍单独控制 Math 与 student forward 是否重叠。

两个 teacher 使用一份 teacher 参数容量的持久 CUDA 存储，按 FSDP handle 分槽。
目前 Math／Code 各 36 层，共享 37 个槽（36 个 decoder 层，加一个根槽），合计约
5,865 MiB。上限约束整组参数槽，不包含 activation、logits、KV cache 或 student 内存。
这些槽在 rollout 和训练更新期间也保留；关闭开关可恢复原先的显存使用方式。

1. 首次 student scoring 期间，将 Math 参数装入槽；冷启动需要这一轮完整 H2D。
2. Math 第 k 层返回后，在实际 compute stream 上记录 CUDA event。
3. copy stream 等该 event，随后将 Code 第 k 层 CPU pinned shard 写入同一个地址。
   此时 Math 后续层可继续计算。
4. Code 的 FSDP pre-unshard 等待对应槽的 copy event，再使用参数，不重复分配或拷贝。
5. Code 用完第 k 层后，同样将槽预装为下一轮 Math 参数。

Embedding、最终 norm、LM head 所在的根槽在整个 teacher forward 结束后才交接，
保证 tied embedding／head 的最后一次使用完成。两个 teacher 的 forward 和评分仍按
Math → Code 顺序执行；这里重叠的是下一 teacher 的参数加载与当前 teacher 的计算。

当前限制：单 GPU、FSDP1 NO_SHARD、CPU offload、`use_orig_params=false`、相同的
FSDP 参数布局；参数不能使用 FSDP mixed precision，`forward_prefetch=false`。
使用冻结 teacher、`no_grad`、每次评分一个固定 micro-batch（可以含多个样本）；每层每次只执行一次。
Code 的 remove-padding 路径支持；不支持 fused kernels、chat-template switching 或动态分批。
不支持的布局／超出容量在安装 hooks 前报错。forward 异常会禁止继续复用该流水线。

```bash
# 开启（launcher 默认）
TEACHER_LAYER_PIPELINE=true bash scripts/local/profile_native_single_gpu.sh

# 恢复此前的单 shard 预取方式
TEACHER_LAYER_PIPELINE=false bash scripts/local/profile_native_single_gpu.sh

# student / Math / Code 都一次 forward 处理 4 个样本；actor 更新 micro-batch 仍为 1。
OPENMOPD_TRAIN_BATCH_SIZE=4 bash scripts/local/profile_native_single_gpu.sh
```

Nsight 的 `openmopd::io::h2d::teacher_layer_prefetch::<目标teacher>::<层>`
标识实际加载的目标，而非提交 copy 时正在计算的 teacher。
`teacher_overlap.json` 的 `layer_pipeline` 分别统计两个方向的 H2D 容量、次数，
以及与前一 teacher 模型 kernel 的真实 GPU 时间交集；根槽等待和冷启动会降低重叠比例。

真实 checkpoint 的固定输入校验：

```bash
CUDA_DEVICE_MAX_CONNECTIONS=8 PYTHONPATH=training/verl \
  /home/xxf/anaconda3/envs/mopd/bin/python scripts/local/verify_teacher_layer_pipeline.py \
  --student /home/xxf/Distill/models/OPD/MixSFT \
  --math /home/xxf/Distill/models/OPD/Math \
  --code /home/xxf/Distill/models/OPD/Code \
  --output output/teacher_layer_verification.json
```

可添加 `--batch-size 4 --forward-overlap true` 比较更大的评分批次。参数 H2D 字节数不随
batch 增长；activation、logits 和输入输出传输会增长。添加 `--profile` 可用 Nsight 的
`--capture-range=cudaProfilerApi` 只采集预热后的评分。

校验覆盖 1248-token 输入、1024-token response、左 padding、Code remove-padding，
对比两个 teacher 的评分输出，交替开关 Math/student forward overlap，并检查多轮槽地址不变。
这项固定输入校验的耗时不代表完整训练 step 性能。

2026-09-24 的固定输入实测（RTX PRO 5000 72GB，PyTorch 2.8，预热后 Nsight 采样四轮，
Math/student forward overlap 开启；每个 batch 都与相同 batch 的普通 FSDP 逐位对齐）：

| Scoring batch | 每样本评分耗时 | Code H2D 与 Math kernel 重叠 | Math H2D 与 Code kernel 重叠 | 局部评分峰值显存 |
| --- | --- | --- | --- | --- |
| 1 | 301 ms | 25.4% | 29.2% | 13.83 GiB |
| 2 | 201 ms | 33.7% | 61.9% | 16.15 GiB |
| 4 | 184 ms | 45.8% | 76.7% | 21.76 GiB |

每批参数 H2D 均为 11.46 GiB（两个方向各 37 次，共用 5.73 GiB 参数槽）。表中的显存和
耗时只覆盖局部评分，未包含 vLLM rollout、optimizer 或训练更新，不能作为整步训练性能。
每样本耗时为四轮批次耗时中位数除以 batch；重叠率按 H2D 活动总时长加权。

实际 launcher 的 batch 4 另通过两步训练，但峰值 allocated memory 达 70.63 GiB，
在这张卡上余量较小。日常可先设 `OPENMOPD_TRAIN_BATCH_SIZE=2`。
