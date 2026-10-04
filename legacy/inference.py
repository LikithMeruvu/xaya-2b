"""Minimal XAYA 2B inference loader for an exported adapter + decision head."""
from __future__ import annotations
import json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from peft import PeftModel


def find_core(m):
    x=m;seen=set()
    for _ in range(12):
        if id(x) in seen:break
        seen.add(id(x))
        if x.__class__.__name__=='Qwen3_5Model' or (hasattr(x,'visual') and hasattr(x,'language_model') and hasattr(x,'get_placeholder_mask')):return x
        if hasattr(x,'base_model') and x.base_model is not x:x=x.base_model;continue
        if hasattr(x,'model') and x.model is not x:x=x.model;continue
        break
    raise RuntimeError('Qwen3_5Model core not found')

class XayaHead(nn.Module):
    def __init__(self, backbone, hidden, max_options=256):
        super().__init__();self.backbone=backbone;self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,max_options))
    def forward(self,batch):
        out=find_core(self.backbone)(**batch,use_cache=False,return_dict=True);h=out.last_hidden_state;idx=batch['attention_mask'].sum(-1).long()-1
        return self.head(h[torch.arange(h.size(0),device=h.device),idx].float())

class XayaDecisionModel:
    def __init__(self, export_dir, device='cuda', dtype=torch.float16):
        export_dir=Path(export_dir);meta=json.loads((export_dir/'release_config.json').read_text());self.meta=meta;self.device=torch.device(device)
        self.processor=AutoProcessor.from_pretrained(export_dir/'processor')
        base=Qwen3_5ForConditionalGeneration.from_pretrained(meta['model_id'],revision=meta.get('model_revision'),dtype=dtype,low_cpu_mem_usage=True).to(self.device)
        bb=PeftModel.from_pretrained(base,export_dir/'adapter')
        self.model=XayaHead(bb,int(base.config.text_config.hidden_size),int(meta.get('max_options',256))).to(self.device)
        self.model.head.load_state_dict(torch.load(export_dir/'decision_head.pt',map_location='cpu'));self.model.eval()
    def _option_text(self,o):
        if isinstance(o,str):return o
        return str(o.get('label','')) if not o.get('description') else f"{o.get('label','')} - {o.get('description')}"
    @torch.inference_mode()
    def decide(self,state,question,options,primitive='choice',image_path=None):
        lines='\n'.join(f'Option {i+1}: {self._option_text(o)}' for i,o in enumerate(options));text=f"Decision type: {primitive.upper()}\n\nContext:\n{state}\n\nQuestion:\n{question}\n\nAllowed options:\n{lines}\n\nChoose exactly one option."
        content=[]
        if image_path:content.append({'type':'image','path':str(Path(image_path).resolve())})
        content.append({'type':'text','text':text});msgs=[[{'role':'user','content':content}]]
        batch=self.processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True});batch={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        logits=self.model(batch)[0,:len(options)];p=F.softmax(logits,dim=-1).cpu();i=int(p.argmax());return {'index':i,'option':options[i],'probabilities':p.tolist()}
