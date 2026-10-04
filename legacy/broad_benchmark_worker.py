from __future__ import annotations
import argparse, json, math, os, time, traceback
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from peft import PeftModel


def find_core(m):
    x=m;seen=set()
    for _ in range(16):
        if id(x) in seen: break
        seen.add(id(x))
        if x.__class__.__name__=='Qwen3_5Model' or (hasattr(x,'visual') and hasattr(x,'language_model') and hasattr(x,'get_placeholder_mask')):
            return x
        if hasattr(x,'base_model') and x.base_model is not x:
            x=x.base_model; continue
        if hasattr(x,'model') and x.model is not x:
            x=x.model; continue
        break
    raise RuntimeError('Qwen3_5Model core not found')

class XayaHead(nn.Module):
    def __init__(self, backbone, hidden, max_options=256):
        super().__init__(); self.backbone=backbone; self.head=nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,max_options))
    def forward(self,batch):
        out=find_core(self.backbone)(**batch,use_cache=False,return_dict=True)
        h=out.last_hidden_state
        idx=batch['attention_mask'].sum(-1).long()-1
        return self.head(h[torch.arange(h.size(0),device=h.device),idx].float())

def load_model(root: Path):
    meta=json.loads((root/'release_config.json').read_text())
    proc=AutoProcessor.from_pretrained(root/'processor')
    tok=getattr(proc,'tokenizer',None)
    if tok is not None:
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        tok.padding_side='right'
    qcfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
    base=Qwen3_5ForConditionalGeneration.from_pretrained(
        meta['model_id'], revision=meta.get('model_revision'), quantization_config=qcfg,
        device_map={'':0}, dtype=torch.float16, low_cpu_mem_usage=True)
    bb=PeftModel.from_pretrained(base, root/'adapter')
    model=XayaHead(bb,int(base.config.text_config.hidden_size),int(meta.get('max_options',256)))
    model.head.to('cuda:0')
    model.head.load_state_dict(torch.load(root/'decision_head.pt',map_location='cpu'))
    model.eval()
    return model,proc,meta

def opt_text(o:Any):
    if isinstance(o,str): return o
    lab=str(o.get('label',''))
    desc=o.get('description')
    return lab if not desc else f'{lab} - {desc}'

def clip_by_tokens(tok,text,max_tokens):
    text=str(text or '')
    ids=tok.encode(text,add_special_tokens=False)
    if len(ids)<=max_tokens: return text
    a=max_tokens//2; b=max_tokens-a
    return tok.decode(ids[:a]+ids[-b:],skip_special_tokens=True)

def build_messages(rec,order,proc,max_length):
    tok=proc.tokenizer
    options=[rec['options'][i] for i in order]
    # Clip long candidate responses so pairwise reward benchmarks cannot OOM.
    per_opt=max(160,min(1400,(max_length-800)//max(1,len(options))))
    rendered=[]
    for j,o in enumerate(options):
        txt=clip_by_tokens(tok,opt_text(o),per_opt)
        rendered.append(f'Option {j+1}: {txt}')
    q=clip_by_tokens(tok,rec.get('question',''),min(2200,max_length//3))
    state=clip_by_tokens(tok,rec.get('state',''),min(2600,max_length//3))
    prim=str(rec.get('primitive','choice')).upper()
    text=(f'Decision type: {prim}\n\nContext:\n{state}\n\nQuestion:\n{q}\n\nAllowed options:\n' + '\n'.join(rendered) + '\n\nChoose exactly one option.')
    content=[]
    if rec.get('image'):
        content.append({'type':'image','path':str(Path(rec['image']).resolve())})
    content.append({'type':'text','text':text})
    return [[{'role':'user','content':content}]]

def make_orders(n,rotations):
    if isinstance(rotations,list): return rotations
    k=int(rotations or 1)
    if k<=1 or n<=1: return [list(range(n))]
    if k==2: return [list(range(n)),list(reversed(range(n)))]
    # cyclic rotations for robustness
    return [[(i+s)%n for i in range(n)] for s in range(min(k,n))]

@torch.inference_mode()
def score_one(model,proc,rec,max_length):
    n=len(rec['options']); orders=make_orders(n,rec.get('rotations',2))
    total=torch.zeros(n,device='cuda:0',dtype=torch.float32)
    lat=[]
    for order in orders:
        msgs=build_messages(rec,order,proc,max_length)
        t0=time.perf_counter()
        batch=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'padding':True})
        batch={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in batch.items()}
        logits=model(batch)[0,:n]
        p=F.softmax(logits,dim=-1).float()
        torch.cuda.synchronize()
        lat.append(time.perf_counter()-t0)
        back=torch.zeros(n,device='cuda:0',dtype=torch.float32)
        for j,old in enumerate(order): back[old]=p[j]
        total += back
    p=(total/len(orders)).cpu()
    return p,lat

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-root',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--rank',type=int,required=True)
    ap.add_argument('--world',type=int,default=2)
    ap.add_argument('--max-length',type=int,default=8192)
    ap.add_argument('--deadline-epoch',type=float,default=0)
    args=ap.parse_args()
    torch.cuda.set_device(0)
    model,proc,meta=load_model(Path(args.model_root))
    rows=[json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    outp=Path(args.output);outp.parent.mkdir(parents=True,exist_ok=True)
    done=set()
    if outp.exists():
        for line in outp.open():
            try: done.add(json.loads(line)['id'])
            except Exception: pass
    assigned=[r for i,r in enumerate(rows) if i%args.world==args.rank and r['id'] not in done]
    total_assigned=sum(1 for i in range(len(rows)) if i%args.world==args.rank)
    mode='a' if outp.exists() else 'w'
    with outp.open(mode,buffering=1) as f:
        for j,rec in enumerate(assigned,1):
            if args.deadline_epoch and time.time()>args.deadline_epoch:
                print(f'[rank{args.rank}] deadline stop before {rec["id"]}',flush=True); break
            try:
                p,lats=score_one(model,proc,rec,args.max_length)
                gold=int(rec['answer'])
                pred=int(p.argmax())
                one=F.one_hot(torch.tensor(gold),num_classes=len(p)).float()
                brier=float(((p-one)**2).sum())
                nll=float(-math.log(max(1e-12,float(p[gold]))))
                obj={
                    'id':rec['id'],'benchmark':rec.get('benchmark'),'gold':gold,'pred':pred,
                    'correct':int(pred==gold),'probabilities':[float(x) for x in p],
                    'confidence':float(p[pred]),'gold_prob':float(p[gold]),'brier':brier,'nll':nll,
                    'latencies_s':lats,'latency_s':float(sum(lats)),'meta':rec.get('meta',{})
                }
                f.write(json.dumps(obj,ensure_ascii=False)+'\n')
            except Exception as e:
                obj={'id':rec['id'],'benchmark':rec.get('benchmark'),'error':repr(e),'trace':traceback.format_exc(limit=3),'meta':rec.get('meta',{})}
                f.write(json.dumps(obj,ensure_ascii=False)+'\n')
                print(f'[rank{args.rank}] ERROR {rec["id"]}: {e}',flush=True)
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
            if j==1 or j%50==0 or j==len(assigned):
                print(f'[rank{args.rank}] new {j}/{len(assigned)} total_target={total_assigned}',flush=True)
    print(f'[rank{args.rank}] done',flush=True)

if __name__=='__main__': main()
