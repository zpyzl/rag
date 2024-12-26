import sys
from pathlib import Path

from tqdm import tqdm

files = Path(sys.argv[1]).rglob('*')
i = 0
for f in tqdm(files):
    i += 1
    if '英雄志' in str(f):
        print(i)
