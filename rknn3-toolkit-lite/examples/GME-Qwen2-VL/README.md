# GME-Qwen2-VL RKNN 示例说明
GME-Qwen2-VL 是基于 Qwen2-VL 的统一多模态向量模型，主要用于将文本、图片、图文对编码为同一语义空间中的向量表示，适合多模态检索、图文匹配和 RAG 等场景。与常规的视觉问答/对话模型不同，GME-Qwen2-VL 的核心目标不是生成长文本回答，而是输出可用于相似度计算、召回和排序的高质量 embedding。

该模型适用于以下典型任务：
- 文本检索图片
- 图片检索图片
- 图文混合检索

本DEMO包含一个基于 `rknn3lite` 的多模态推理示例脚本 `test.py`，用于完成以下流程：
1. 加载视觉 RKNN 模型；
2. 对输入图片做预处理并提取视觉特征；
3. 将视觉特征作为 `<image>` 输入喂给 LLM RKNN 模型；
4. 通过 tokenizer / embed / output 回调完成文本生成与结果输出；
5. 打印 Vision / LLM 的耗时信息，并导出 `model_outputs.npy`。

## 模型与依赖文件

`test.py` 默认使用以下路径：

- LLM 模型：`/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.rknn`
- Vision 模型：`/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-vision_quant.rknn`
- Embedding 文件：`/userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.embed.bin`
- Tokenizer：`gme-Qwen2-VL-2B`

如果你的文件路径不同，可以通过命令行参数覆盖。

同时请确保以下配套文件存在：

- `*.rknn` 对应的 `*.weight` 文件
- 可正常加载的 tokenizer 目录或 HuggingFace 模型名
- 与脚本中 `VOCAB_SIZE = 151936` 匹配的 `embed.bin`

## 运行环境

建议准备 Python 运行环境，并安装以下依赖：

- `numpy`
- `opencv-python`
- `transformers`
- `rknn3lite`

注意：`transformers`库由于模型的版本限制最好使用4.51.3

脚本还依赖：

- `ctypes`
- 目标设备上的 RKNN / RKLLM 运行库

脚本中默认设置了：

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
```

如果你的环境不需要镜像源，可自行调整。

## 命令行参数

`test.py` 当前支持以下参数：

```bash
python test.py \
  --rknn_llm_path /path/to/GmeQwen2VL-llm_quant.rknn \
  --rknn_vision_path /path/to/GmeQwen2VL-vision_quant.rknn \
  --tokenizer_path /path/to/tokenizer_or_hf_name \
  --embed_path /path/to/GmeQwen2VL-llm_quant.embed.bin \
  --image_path /path/to/demo.jpg \
  --prompt "Describe this image."
```

参数说明：

- `--rknn_llm_path`：LLM RKNN 模型路径
- `--rknn_vision_path`：Vision RKNN 模型路径
- `--tokenizer_path`：tokenizer 本地目录或 HuggingFace 名称
- `--embed_path`：embedding 二进制文件路径
- `--image_path`：输入图片路径
- `--prompt`：用户提示词

## 运行示例

```bash
python test.py \
  --rknn_llm_path /userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.rknn \
  --rknn_vision_path /userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-vision_quant.rknn \
  --tokenizer_path gme-Qwen2-VL-2B \
  --embed_path /userdata/rknn_Qwen2_5_VL_demo/model/GmeQwen2VL-llm_quant.embed.bin
```

## 推理流程说明

### 1. 加载模型

脚本会分别加载：

- Vision RKNN 模型
- LLM RKNN 模型

并自动尝试加载与 `.rknn` 同名的 `.weight` 文件。

### 2. 初始化运行时

当前脚本默认使用：

```python
target='rk1820'
```

并分别对 vision 与 llm runtime 调用 `init_runtime()`。

如果你的目标平台不是 `rk1820`，需要修改脚本中的 target 参数。

### 3. 图片预处理

脚本会将输入图片：

- 使用 OpenCV 读取
- 转成 RGB
- resize 到 `392 x 392`

随后根据 Vision 模型输入维度选择两种处理方式：

- 如果输入张量是 4 维：按完整模型路径直接输入
- 否则：走 `prune_model_img_process()` 的裁剪版预处理逻辑

### 4. LLM 推理

视觉模型输出会被封装成 `RKNN3Image`，并通过以下特殊 token 传递给 LLM：

- `<|vision_start|>`
- `<|vision_end|>`
- `<|image_pad|>`

文本生成通过以下回调协同完成：

- `result_callback`
- `tokenizer_callback`
- `embed_callback`
- `output_callback`

## 输出内容

运行成功后，终端通常会看到：

- 模型加载日志
- runtime 初始化日志
- tensor attr 信息
- Vision 推理日志
- LLM 文本输出
- Prefill / Generate / Vision latency 性能统计

同时脚本会保存：

```bash
model_outputs.npy
```
