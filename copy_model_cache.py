#!/usr/bin/env python3
# 将Hugging Face缓存的模型文件复制到当前目录

import os
import shutil
import argparse
from pathlib import Path

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
    
    print(f"正在复制模型文件到 {target_dir}...")
    
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
                    print(f"跳过已存在的文件: {dst_file} ({file_size:.2f} MB)")
                    continue
                
                print(f"复制文件: {file} ({file_size:.2f} MB)")
                shutil.copy2(src_file, dst_file)
                
                total_size += file_size
                file_count += 1
    
    print(f"\n✅ 复制完成! 共复制了 {file_count} 个文件，总大小约 {total_size:.2f} MB")
    return True

def main():
    parser = argparse.ArgumentParser(description="将Hugging Face缓存的模型文件复制到当前目录")
    parser.add_argument("--model_id", type=str, default="microsoft/phi-3-mini-4k-instruct",
                      help="Hugging Face模型ID (默认: microsoft/phi-3-mini-4k-instruct)")
    parser.add_argument("--target_dir", type=str, default=None,
                      help="目标目录 (默认: 当前目录下以模型名创建的目录)")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Hugging Face缓存目录 (默认: 自动检测)")
    
    args = parser.parse_args()
    
    # 获取缓存目录
    cache_dir = args.cache_dir or find_huggingface_cache()
    if not cache_dir:
        print("❌ 无法找到Hugging Face缓存目录")
        return
    
    print(f"使用缓存目录: {cache_dir}")
    
    # 设置目标目录
    if args.target_dir is None:
        model_name = args.model_id.split("/")[-1]
        args.target_dir = model_name
    
    # 查找并复制模型文件
    model_paths = find_model_in_cache(cache_dir, args.model_id)
    if not model_paths:
        print(f"❌ 在缓存中未找到模型 {args.model_id}")
        return
    
    print(f"在缓存中找到模型目录:")
    for path in model_paths:
        print(f" - {path}")
    
    copy_model_files(model_paths, args.target_dir)

if __name__ == "__main__":
    main() 