import importlib.metadata
import os
import platform
import random
import subprocess
import sys


def seed_everything(seed):
    from .common import read_json
    runtime = read_json('configs/runtime.json')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = runtime['cublas_workspace_config']
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(runtime['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = runtime['allow_tf32']
    torch.backends.cudnn.allow_tf32 = runtime['allow_tf32']
    torch.backends.cudnn.benchmark = runtime['cudnn_benchmark']
    torch.backends.cudnn.deterministic = runtime['cudnn_deterministic']


def load_model(config, adapter=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    seed_everything(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable. QLoRA has not started; use a documented CUDA launch.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable; record and freeze an explicit FP16 candidate before running')
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map={'':0}, low_cpu_mem_usage=True,
        attn_implementation=config['attention_implementation'])
    # Apply the same nonquantized-parameter casting to both baseline and adapted models.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    elif training:
        model = get_peft_model(model, LoraConfig(**config['lora'], task_type='CAUSAL_LM'))
    model.config.use_cache = not training
    packages = {}
    for name in ['torch','transformers','peft','bitsandbytes','accelerate','numpy','scipy','scikit-learn','mlflow','huggingface-hub','tokenizers','safetensors']:
        packages[name] = importlib.metadata.version(name)
    hardware = {'gpu':torch.cuda.get_device_name(), 'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'bf16_supported':torch.cuda.is_bf16_supported(), 'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
        'python':sys.version, 'platform':platform.platform(), 'packages':packages,
        'nvidia_smi':gpu_summary(), 'cpu_count':os.cpu_count(),
        'lora_modules':[n for n,m in model.named_modules() if hasattr(m,'lora_A')],
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'nonquantized_cast':'PEFT prepare_model_for_kbit_training (identical baseline and tuned)',
        'runtime':{'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32},
        'determinism':'seeded; TF32 and cuDNN benchmark disabled; consult outputs/smoke/resume_verified.json for measured resumption agreement and tolerance; no bitwise guarantee across hardware'}
    return model, tokenizer, hardware


def gpu_summary():
    """Record reproducibility fields without other users' process names or IDs."""
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
        '--format=csv,noheader',
    ], text=True, timeout=10)
