"""Measure warmed, single-request end-to-end latency with synchronized CUDA."""
import argparse
import json
import statistics
from xaya import XAYA

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--image')
parser.add_argument('--runs', type=int, default=20)
args = parser.parse_args()
if args.runs < 2:
    parser.error('--runs must be at least 2')
model = XAYA.from_pretrained(args.model)
query = dict(state='', question='What is shown?' if args.image else 'Which queue?',
             options=['Cat', 'Dog', 'Other'] if args.image else ['Billing', 'Technical', 'Account'],
             image=args.image)
import torch
for _ in range(3):
    model.choice(**query)
torch.cuda.reset_peak_memory_stats()
latencies = [model.choice(**query)['latency_ms'] for _ in range(args.runs)]
print(json.dumps({'gpu': torch.cuda.get_device_name(), 'mode': 'NF4', 'runs': args.runs,
                  'image': bool(args.image), 'p50_ms': statistics.median(latencies),
                  'p95_ms': statistics.quantiles(latencies, n=20, method='inclusive')[18],
                  'peak_allocated_gb': torch.cuda.max_memory_allocated()/1e9,
                  'scope': 'processor + transfer + forward + probabilities; excludes load/warmup'}, indent=2))
