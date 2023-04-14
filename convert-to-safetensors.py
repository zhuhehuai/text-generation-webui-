'''

Converts a transformers model to safetensors format and shards it.

This makes it faster to load (because of safetensors) and lowers its RAM usage
while loading (because of sharding).

Based on the original script by 81300:

https://gist.github.com/81300/fe5b08bff1cba45296a829b9d6b0f303

'''

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument('MODEL', type=str, default=None, nargs='?', help="输入模型的路径。")
parser.add_argument('--output', type=str, default=None, help='输出文件夹的路径（默认：models/{model_name}_safetensors）。')
parser.add_argument("--max-shard-size", type=str, default="2GB", help="分片的最大大小（以 GB 或 MB 为单位）（默认值：%(default)s）。")
parser.add_argument('--bf16', action='store_true', help='以 bfloat16 精度加载模型。需要 NVIDIA Ampere GPU。')
args = parser.parse_args()

if __name__ == '__main__':
    path = Path(args.MODEL)
    model_name = path.name

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(path)

    out_folder = args.output or Path(f"models/{model_name}_safetensors")
    print(f"将转换后的模型保存到 {out_folder}最大分片大小为 {args.max_shard_size}...")
    model.save_pretrained(out_folder, max_shard_size=args.max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(out_folder)
