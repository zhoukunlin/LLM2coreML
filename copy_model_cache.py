#!/usr/bin/env python3
# LLM模型缓存复制工具
# 将Hugging Face缓存的模型文件复制到当前目录，避免重复下载

import os
import shutil
import argparse
from pathlib import Path
import sys

# 从主转换脚本中导入模型列表（如果可用）
try:
    from LLM2coreML import SUPPORTED_MODELS
    has_models_list = True
except ImportError:
    has_models_list = False

def find_huggingface_cache():
    """查找Hugging Face缓存目录"""
    # 常见的缓存位置
    potential_locations = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/Library/Caches/huggingface"),
        os.path.expanduser("~/AppData/Local/huggingface/cache")
    ]
    
    for location in potential_locations:
        if os.path.exists(location):
            return location
    
    return None

def find_model_in_cache(cache_dir, model_id):
    """在缓存中查找特定模型的文件"""
    model_name = model_id.split("/")[-1]
    
    # 查找模型目录
    model_paths = []
    
    # 检查hub目录
    hub_model_dir = os.path.join(cache_dir, "hub", f"models--{model_id.replace('/', '--')}")
    if os.path.exists(hub_model_dir):
        model_paths.append(hub_model_dir)
    
    # 检查modules目录
    modules_model_dir = os.path.join(cache_dir, "modules", "transformers_modules", model_id)
    if os.path.exists(modules_model_dir):
        model_paths.append(modules_model_dir)
    
    return model_paths

def copy_model_files(model_paths, target_dir):
    """复制模型文件到目标目录"""
    if not model_paths:
        print("❌ 未在缓存中找到模型文件")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    total_size = 0
    file_count = 0
    
    print(f"📦 正在复制模型文件到 {target_dir}...")
    print(f"⏳ 请稍候，这可能需要一些时间...")
    
    for model_path in model_paths:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                src_file = os.path.join(root, file)
                file_size = os.path.getsize(src_file) / (1024 * 1024)  # 转换为MB
                
                # 创建相对路径
                rel_path = os.path.relpath(root, model_path)
                target_subdir = os.path.join(target_dir, rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                dst_file = os.path.join(target_subdir, file)
                
                # 如果文件已存在且大小相同，则跳过
                if os.path.exists(dst_file) and os.path.getsize(dst_file) == os.path.getsize(src_file):
                    print(f"⏭️ 跳过已存在的文件: {dst_file} ({file_size:.2f} MB)")
                    continue
                
                print(f"📄 复制文件: {file} ({file_size:.2f} MB)")
                shutil.copy2(src_file, dst_file)
                
                total_size += file_size
                file_count += 1
    
    print(f"\n✅ 复制完成! 共复制了 {file_count} 个文件，总大小约 {total_size:.2f} MB")
    print(f"💡 提示: 现在可以使用'python LLM2coreML.py --model_id {model_id} --cache_dir {target_dir}'来转换模型")
    return True

def list_available_models(cache_dir):
    """列出缓存中可用的模型"""
    if not cache_dir:
        print("❌ 无法找到Hugging Face缓存目录")
        return
    
    print("\n📋 本地缓存中的可用模型:")
    
    # 检查hub目录中的模型
    hub_dir = os.path.join(cache_dir, "hub")
    if os.path.exists(hub_dir):
        model_dirs = [d for d in os.listdir(hub_dir) if d.startswith("models--") and os.path.isdir(os.path.join(hub_dir, d))]
        
        if model_dirs:
            for model_dir in model_dirs:
                model_id = model_dir.replace("models--", "").replace("--", "/")
                print(f"  - {model_id}")
        else:
            print("  (在hub目录中未找到缓存的模型)")
    
    print("\n💡 使用--model_id参数指定要复制的模型")

def main():
    parser = argparse.ArgumentParser(description="LLM模型缓存复制工具 - 将下载的模型文件复制到指定目录")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-3-mini-4k-instruct",
                      help="Hugging Face模型ID (默认: microsoft/phi-3-mini-4k-instruct)")
    parser.add_argument("--target_dir", type=str, default=None,
                      help="目标目录 (默认: 当前目录下以模型名创建的目录)")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Hugging Face缓存目录 (默认: 自动检测)")
    parser.add_argument("--list", action="store_true",
                      help="列出缓存中的可用模型")
    
    args = parser.parse_args()
    
    # 获取缓存目录
    cache_dir = args.cache_dir or find_huggingface_cache()
    if not cache_dir:
        print("❌ 无法找到Hugging Face缓存目录")
        sys.exit(1)
    
    print(f"🔍 使用缓存目录: {cache_dir}")
    
    # 列出缓存中的模型
    if args.list:
        list_available_models(cache_dir)
        sys.exit(0)
    
    # 设置模型ID和目标目录
    model_id = args.model_id
    
    if args.target_dir is None:
        model_name = model_id.split("/")[-1]
        args.target_dir = model_name
    
    print(f"🔄 将复制模型 '{model_id}' 到目录 '{args.target_dir}'")
    
    # 查找并复制模型文件
    model_paths = find_model_in_cache(cache_dir, model_id)
    if not model_paths:
        print(f"❌ 在缓存中未找到模型 {model_id}")
        print("💡 提示: 运行'--list'选项查看可用的模型")
        sys.exit(1)
    
    print(f"✓ 在缓存中找到模型目录:")
    for path in model_paths:
        print(f"  - {path}")
    
    if copy_model_files(model_paths, args.target_dir):
        print("\n🎉 模型复制成功！现在可以使用此缓存进行模型转换")
        print(f"💻 示例命令: python LLM2coreML.py --model_id {model_id} --cache_dir {os.path.abspath(args.target_dir)}")
    else:
        print("\n❌ 模型复制失败！")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 80)
    print("🍎 LLM模型缓存复制工具 - LLM2CoreML项目的辅助工具")
    print("=" * 80)
    main() 