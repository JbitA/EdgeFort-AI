import json
from pathlib import Path
from edgeai.evaluation import evaluate
p=Path('results/evaluation');p.mkdir(parents=True,exist_ok=True);d=evaluate(p/'work');(p/'reference_metrics.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps(d,indent=2,allow_nan=False))
