import argparse
import json
from xaya import XAYA

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Extracted checkpoint directory')
parser.add_argument('--image', help='Optional local image')
args = parser.parse_args()
model = XAYA.from_pretrained(args.model)
result = model.choice('A customer says their invoice was charged twice.',
                      'Which support queue should handle this?',
                      ['Billing', 'Technical', 'Account'])
print(json.dumps(result, indent=2))
if args.image:
    result = model.choice('', 'What is shown in the image?', ['Cat', 'Dog', 'Other'], image=args.image)
    print(json.dumps(result, indent=2))
