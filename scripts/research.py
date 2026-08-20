import json
from pathlib import Path
from edgeai.evaluation import robustness_campaign
p=Path('results/research');p.mkdir(parents=True,exist_ok=True);d=robustness_campaign(range(10));(p/'quantization_robustness.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps(d['summary'],indent=2))
