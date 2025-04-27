#!/usr/bin/env python3
# Phi-3到CoreML转换脚本

import os
import torch
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import time
import datetime
import sys
import subprocess
from pathlib import Path

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

def install_dependencies():
    """安装必要的依赖项"""
    print("正在检查并安装必要的依赖项...")
    
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "coremltools>=7.1",
        "numpy>=1.23.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"正在安装 {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        
        print("✅ 所有依赖项安装完成!")
        return True
    except Exception as e:
        print(f"❌ 安装依赖项时出错: {str(e)}")
        return False

def convert_phi3_to_coreml(
    model_id="microsoft/phi-3-mini-4k-instruct",
    output_dir=None,
    max_seq_len=4096,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    use_float16=False,
    cache_dir=None
):
    """
    将Phi-3模型转换为CoreML格式
    
    参数:
        model_id (str): Hugging Face模型ID
        output_dir (str): 输出目录，默认为"./models"
        max_seq_len (int): 最大序列长度
        compute_units: CoreML计算单元 (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ...)
        use_float16 (bool): 是否使用float16精度
        cache_dir (str): 模型缓存目录
    
    返回:
        (bool, str): (成功状态, 模型路径或错误信息)
    """
    try:
        start_time = time.time()
        
        # 设置默认输出目录
        if output_dir is None:
            model_name = model_id.split("/")[-1]
            output_dir = DEFAULT_OUTPUT_DIR
        
        # 设置默认缓存目录
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        
        print(f"使用模型: {model_id}")
        print(f"输出目录: {output_dir}")
        print(f"最大序列长度: {max_seq_len}")
        print(f"计算单元: {compute_units}")
        print(f"使用float16: {use_float16}")
        print(f"缓存目录: {cache_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载预训练分词器
        print("加载分词器...")
        tokenizer_kwargs = {"local_files_only": False}
        if cache_dir:
            tokenizer_kwargs["cache_dir"] = cache_dir
            
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        
        # 加载模型
        print("加载模型...")
        model_kwargs = {
            "torch_dtype": torch.float16 if use_float16 else torch.float32,
            "trust_remote_code": True,
            "local_files_only": False
        }
        
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        
        # 创建示例输入
        sample_text = "你好"
        inputs = tokenizer(sample_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        print(f"示例输入形状: {input_ids.shape}")
        
        # 简化的模型类
        class SimpleModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
        
        # 包装模型
        wrapped_model = SimpleModel(model)
        
        # 使用JIT追踪
        print("使用JIT追踪模型...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model,
                (input_ids, attention_mask)
            )
        
        # 输入规格
        input_specs = [
            ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32)
        ]
        
        # 设置输出路径
        model_name = model_id.split("/")[-1]
        output_path = os.path.join(output_dir, f"{model_name}.mlpackage")
        
        # 转换为CoreML
        print("转换为CoreML格式...")
        mlmodel = ct.convert(
            traced_model,
            inputs=input_specs,
            compute_units=compute_units,
            minimum_deployment_target=ct.target.macOS13
        )
        
        # 保存模型
        print(f"保存CoreML模型到: {output_path}")
        mlmodel.save(output_path)
        
        # 计算转换时间
        end_time = time.time()
        conversion_time = end_time - start_time
        
        # 保存成功信息
        success_info = {
            "status": "success",
            "model_id": model_id,
            "output_path": output_path,
            "conversion_time_seconds": conversion_time,
            "conversion_time_formatted": str(datetime.timedelta(seconds=int(conversion_time))),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        success_path = os.path.join(output_dir, "conversion_success.json")
        with open(success_path, "w") as f:
            json.dump(success_info, f, indent=2)
        
        print(f"✅ 转换成功! 用时: {str(datetime.timedelta(seconds=int(conversion_time)))}")
        return True, output_path
    
    except Exception as e:
        error_message = str(e)
        print(f"❌ 转换失败: {error_message}")
        
        # 保存错误信息
        if output_dir:
            error_info = {
                "status": "error",
                "model_id": model_id,
                "error_message": error_message,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            error_path = os.path.join(output_dir, "conversion_error.json")
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
        
        return False, error_message

def main():
    parser = argparse.ArgumentParser(description="将Phi-3模型转换为CoreML格式")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-3-mini-4k-instruct",
                      help="Hugging Face模型ID (默认: microsoft/phi-3-mini-4k-instruct)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--max_seq_len", type=int, default=4096,
                      help="最大序列长度 (默认: 4096)")
    parser.add_argument("--use_float16", action="store_true",
                      help="使用float16精度 (默认: False)")
    parser.add_argument("--use_gpu", action="store_true",
                      help="使用GPU进行转换 (默认: 仅使用CPU)")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help=f"模型缓存目录 (默认: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--install_deps", action="store_true",
                      help="安装依赖项 (默认: False)")
    
    args = parser.parse_args()
    
    # 安装依赖项
    if args.install_deps:
        if not install_dependencies():
            print("依赖项安装失败，退出程序")
            sys.exit(1)
    
    # 设置计算单元
    compute_units = ct.ComputeUnit.CPU_AND_GPU if args.use_gpu else ct.ComputeUnit.CPU_ONLY
    
    success, result = convert_phi3_to_coreml(
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        compute_units=compute_units,
        use_float16=args.use_float16,
        cache_dir=args.cache_dir
    )
    
    if success:
        print(f"\n转换成功! 模型保存在: {result}")
        sys.exit(0)
    else:
        print(f"\n转换失败: {result}")
        sys.exit(1)

if __name__ == "__main__":
    main() 