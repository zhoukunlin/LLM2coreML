# LLM2coreML

将大型语言模型（LLM）转换为CoreML格式，以便在苹果设备上本地运行。**只需修改模型ID，即可一键转换各种LLM模型为Apple设备可用的格式。**

## 特性

- 简单易用：只需一行命令即可完成转换
- 通用转换：支持多种Hugging Face上的LLM模型
- 自动优化：针对Apple设备性能优化的CoreML格式
- 完整流程：从模型下载到转换全流程自动化

## 当前已测试的模型

- **Microsoft Phi-3 系列模型**（默认支持）
  - `microsoft/phi-3-mini-4k-instruct` (推荐，体积小速度快，约4GB)
  - `microsoft/phi-3-mini-128k-instruct` (完整模型，但体积大，约8GB)

- **其他可转换的模型**（通过修改模型ID）
  - `google/gemma-2b` (Google Gemma模型)
  - `meta-llama/Llama-2-7b-chat-hf` (Meta Llama 2模型)
  - `mistralai/Mistral-7B-Instruct-v0.2` (Mistral模型)
  - 以及其他Hugging Face上支持的LLM模型

## 安装

### 环境要求

- macOS 13.0 或更高版本
- Python 3.8 或更高版本
- 推荐安装 [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 步骤

1. 克隆此仓库：

```bash
git clone https://github.com/zhoukunlin/LLM2coreML.git
cd LLM2coreML
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
conda create -n coreml python=3.9
conda activate coreml
```

3. 安装依赖项：

```bash
python LLM2coreML.py --install_deps
```

或者手动安装：

```bash
pip install torch>=2.0.0 transformers>=4.35.0 coremltools>=7.1 numpy>=1.23.0
```

## 使用方法

### 快速开始（使用默认模型）

使用默认参数转换phi-3-mini-4k-instruct模型：

```bash
python LLM2coreML.py
```

### 转换其他LLM模型（修改模型ID）

转换为其他模型只需添加`--model_id`参数：

```bash
# 转换Phi-3-mini-128k-instruct模型
python LLM2coreML.py --model_id "microsoft/phi-3-mini-128k-instruct"

# 转换Google Gemma 2B模型
python LLM2coreML.py --model_id "google/gemma-2b"

# 转换Llama 2模型
python LLM2coreML.py --model_id "meta-llama/Llama-2-7b-chat-hf"
```

### 查看支持的模型列表

```bash
python LLM2coreML.py --list_models
```

### 高级参数配置

```bash
python LLM2coreML.py --model_id "microsoft/phi-3-mini-128k-instruct" --use_gpu --max_seq_len 8192 --use_float16
```

### 参数说明

- `--model_id`：Hugging Face上的模型ID，默认为"microsoft/phi-3-mini-4k-instruct"
- `--output_dir`：输出目录，默认为脚本所在目录下的models子目录
- `--max_seq_len`：最大序列长度，默认为4096
- `--use_float16`：使用float16精度，可减小模型体积但可能影响精度
- `--use_gpu`：使用GPU进行转换（如有可用）
- `--cache_dir`：模型缓存目录，默认为脚本所在目录
- `--install_deps`：安装必要的依赖项
- `--list_models`：列出支持的模型示例

## 转换后的模型使用

转换后的CoreML模型（.mlpackage格式）可在iOS、macOS等苹果设备上使用。

### 模型位置

转换后的模型将保存在以下位置：

```
./models/[model-name].mlpackage
```

例如：`./models/phi-3-mini-4k-instruct.mlpackage`

### 在Apple设备上使用模型

转换后的CoreML模型可以在以下场景使用：

- **iOS/iPadOS应用**：通过Core ML框架加载模型
- **macOS应用**：使用Core ML或通过终端直接使用
- **Swift/Objective-C应用**：使用Core ML API
- **Python**：使用coremltools加载并使用模型

### 简单的Swift调用示例

```swift
import CoreML

func loadModel() {
    do {
        // 加载模型
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        
        // 准备输入
        let inputIds = MLMultiArray(...)
        let attentionMask = MLMultiArray(...)
        let input = ModelInput(input_ids: inputIds, attention_mask: attentionMask)
        
        // 运行推理
        let output = try model.prediction(from: input)
        
        // 处理输出
        // ...
    } catch {
        print("Error loading or running model: \(error)")
    }
}
```

## 复用已下载的模型文件

如果您之前已经下载过模型，可以使用`copy_model_cache.py`工具将缓存的模型文件复制到工作目录，避免重复下载：

```bash
python copy_model_cache.py --model_id "microsoft/phi-3-mini-4k-instruct"
```

### 查看已下载的模型列表

```bash
python copy_model_cache.py --list
```

### 参数说明

- `--model_id`：需要复制的模型ID，默认为"microsoft/phi-3-mini-4k-instruct"
- `--target_dir`：目标目录，默认为以模型名称命名的文件夹
- `--cache_dir`：自定义Hugging Face缓存目录，默认自动检测
- `--list`：列出缓存中的可用模型

这个工具会自动查找Hugging Face的缓存目录，并将模型文件复制到指定位置。在macOS上，缓存通常位于`~/.cache/huggingface/`目录。

## 常见问题

### 内存不足问题
如果转换过程中出现内存不足错误，请尝试以下解决方案：
- 使用`--use_float16`参数减少内存占用
- 使用更小的模型（如从7B模型改为2B模型）
- 增加系统虚拟内存/交换空间

### 转换失败问题
如果特定模型转换失败，可能是因为模型架构不完全兼容，请尝试：
- 检查错误日志中的具体错误信息（保存在输出目录中）
- 尝试使用较新版本的依赖库（特别是transformers和coremltools）

## 注意事项

- 转换过程可能需要较长时间（取决于您的设备性能和模型大小）
- 建议使用较小的模型（如phi-3-mini-4k-instruct）进行初次尝试
- 转换过程中可能会下载模型文件，请确保网络连接畅通
- 使用`copy_model_cache.py`工具可以重复利用已下载的模型文件，节省带宽和时间
- 某些较大的模型（如70B以上）可能需要特殊的硬件配置才能成功转换

## 贡献

欢迎通过Issue和Pull Request提供反馈和贡献。如果您成功转换了其他模型并想分享经验，请在Issue中告诉我们。

## 许可证

MIT 