"""Reference inference: original prompt/head, optional native NF4, one option order.

This wrapper has not established historical JevBench runtime parity.
"""
import json
import time
from pathlib import Path
import torch
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from peft import PeftModel
from ._inputs import prompt

BASE = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def find_core(model):
    seen = set()
    for _ in range(16):
        if id(model) in seen:
            break
        seen.add(id(model))
        if model.__class__.__name__ == 'Qwen3_5Model' or (
                hasattr(model, 'visual') and hasattr(model, 'language_model')
                and hasattr(model, 'get_placeholder_mask')):
            return model
        if hasattr(model, 'base_model') and model.base_model is not model:
            model = model.base_model
        elif hasattr(model, 'model') and model.model is not model:
            model = model.model
        else:
            break
    raise RuntimeError("Qwen3_5Model core not found")


class Runtime:
    def __init__(self, model_dir, *, load_in_4bit, device, max_length):
        root = Path(model_dir).expanduser().resolve()
        required = ['release_config.json', 'adapter/adapter_config.json',
                    'adapter/adapter_model.safetensors', 'processor/tokenizer.json']
        missing = [p for p in required if not (root / p).is_file()]
        if not (root/'decision_head.pt').is_file():
            missing.append('decision_head.pt')
        if missing:
            raise FileNotFoundError(f"Extract the XAYA checkpoint first. Missing: {', '.join(missing)}")
        meta = json.loads((root/'release_config.json').read_text('utf-8'))
        if meta.get('model_id') != BASE or meta.get('model_revision') != REVISION:
            raise ValueError("Checkpoint config does not match the pinned XAYA base")
        self.max_options = int(meta.get('max_options', 256))
        self.max_length = int(max_length)
        if not 1 <= self.max_options <= 256 or self.max_length <= 0:
            raise ValueError("Invalid option or context limit")
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError("This reference loader requires an NVIDIA GPU and CUDA-enabled PyTorch")
        self.device = torch.device('cuda', self.device.index or 0)
        self.processor = AutoProcessor.from_pretrained(root/'processor')
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        kwargs = dict(revision=REVISION, dtype=torch.float16, low_cpu_mem_usage=True,
                      device_map={'': str(self.device)})
        if load_in_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(BASE, **kwargs)
        self.backbone = PeftModel.from_pretrained(base, root/'adapter').eval()
        hidden = int(base.config.text_config.hidden_size)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, self.max_options))
        # Keep the original serialization and tensor values for this reference path.
        self.head.load_state_dict(torch.load(root/'decision_head.pt', map_location='cpu', weights_only=True))
        self.head = self.head.to(self.device).eval()
        self.core = find_core(self.backbone)

    @torch.inference_mode()
    def decide(self, state, question, options, primitive, image):
        content = []
        if image is not None:
            path = Path(image).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            content.append({'type': 'image', 'path': str(path)})
        content.append({'type': 'text', 'text': prompt(state, question, options, primitive)})
        messages = [[{'role': 'user', 'content': content}]]
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors='pt', processor_kwargs={'padding': True})
        if batch['input_ids'].shape[-1] > self.max_length:
            raise ValueError(f"Input exceeds {self.max_length} tokens; shorten it explicitly")
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        output = self.core(**batch, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        index = batch['attention_mask'].sum(-1).long()-1
        logits = self.head(hidden[torch.arange(hidden.shape[0], device=hidden.device), index].float())
        probabilities = logits[0, :len(options)].softmax(-1).float().cpu().tolist()
        torch.cuda.synchronize(self.device)
        latency = (time.perf_counter()-started)*1000
        if not all(torch.isfinite(torch.tensor(probabilities))):
            raise RuntimeError("Non-finite model probabilities")
        selected = max(range(len(options)), key=probabilities.__getitem__)
        return {'index': selected, 'option': options[selected], 'probabilities': probabilities,
                'latency_ms': round(latency, 3)}
