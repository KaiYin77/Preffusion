from pathlib import Path
import yaml

''' Yaml Parser
'''
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_path_dir = Path('data_path/')
data_path_dir.mkdir(parents=True, exist_ok=True)

splits = ['validation', 'training']
for split in splits:
    f = open('data_path/' + split + '.txt', 'w')
    target_dir = Path(config['data']['root']) / Path('raw/'+split)
    file_path = sorted(target_dir.rglob("*.parquet"))
    for p in file_path:
        f.writelines(str(p)+'\n')
    f.close()
