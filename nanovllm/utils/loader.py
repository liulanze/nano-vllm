import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# The trained weights live on disk as .safetensors files inside the model
# directory. For Qwen3-0.6B you'd typically see:
'''
/path/to/Qwen3-0.6B/
  ├── config.json
  ├── tokenizer.json
  ├── model.safetensors          (or split into multiple parts)
  ├── model-00001-of-00002.safetensors
  └── model-00002-of-00002.safetensors
'''
# Each safetensors file is a dictionary mapping weight names to tensors:
'''
"model.embed_tokens.weight"                        → tensor [151936, 1024]
"model.layers.0.self_attn.q_proj.weight"           → tensor [1024, 1024]
"model.layers.0.self_attn.k_proj.weight"           → tensor [512, 1024]
"model.layers.0.self_attn.v_proj.weight"           → tensor [512, 1024]
"model.layers.0.self_attn.o_proj.weight"           → tensor [1024, 1024]
"model.layers.0.mlp.gate_proj.weight"              → tensor [2816, 1024]
"model.layers.0.mlp.up_proj.weight"                → tensor [2816, 1024]
"model.layers.0.mlp.down_proj.weight"              → tensor [1024, 2816]
"model.layers.0.input_layernorm.weight"            → tensor [1024]
"model.layers.0.post_attention_layernorm.weight"   → tensor [1024]
... (repeat for layers 1-27) ...
"model.norm.weight"                                → tensor [1024]
"lm_head.weight"                                   → tensor [151936, 1024]
'''

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

# Loading Weights from Disk into the Empty Model.
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
