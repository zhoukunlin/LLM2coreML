# LLM2coreML

将大型语言模型（LLM）转换为CoreML格式，以便在苹果设备上本地运行。

## 当前支持的模型

- Microsoft Phi-3 系列模型
  - phi-3-mini-4k-instruct (强烈推荐，体积小速度快)
  - phi-3-mini-128k-instruct (完整模型，但体积大)

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
python phi3_to_coreml.py --install_deps
```

或者手动安装：

```bash
pip install torch>=2.0.0 transformers>=4.35.0 coremltools>=7.1 numpy>=1.23.0
```

## 使用方法

### 基本命令

使用默认参数（转换phi-3-mini-4k-instruct模型）：

```bash
python phi3_to_coreml.py
```

### 高级选项

```bash
python phi3_to_coreml.py --model_id "microsoft/phi-3-mini-128k-instruct" --use_gpu --max_seq_len 8192
```

### 参数说明

- `--model_id`：Hugging Face上的模型ID，默认为"microsoft/phi-3-mini-4k-instruct"
- `--output_dir`：输出目录，默认为脚本所在目录下的models子目录
- `--max_seq_len`：最大序列长度，默认为4096
- `--use_float16`：使用float16精度，可减小模型体积但可能影响精度
- `--use_gpu`：使用GPU进行转换（如有可用）
- `--cache_dir`：模型缓存目录，默认为脚本所在目录
- `--install_deps`：安装必要的依赖项

## 转换后的模型使用

转换后的CoreML模型（.mlpackage格式）可在iOS、macOS等苹果设备上使用。

### 模型位置

转换后的模型将保存在以下位置：

```
./models/phi-3-mini-4k-instruct.mlpackage
```

### 模型大小

- phi-3-mini-4k-instruct: 约4GB（CoreML格式）
- phi-3-mini-128k-instruct: 约8GB（CoreML格式）

## 注意事项

- 转换过程可能需要较长时间（取决于您的设备性能和模型大小）
- 建议使用较小的模型（如phi-3-mini-4k-instruct）进行初次尝试
- 转换过程中可能会下载模型文件，请确保网络连接畅通

## 贡献

欢迎通过Issue和Pull Request提供反馈和贡献。

## 许可证

MIT 