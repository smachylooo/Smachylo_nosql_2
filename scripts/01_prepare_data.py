"""Reservoir sample papers reproducibly, avoiding early-year/field ordering bias."""
import json, os, random, re
from email.utils import parsedate_to_datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

source=Path(os.getenv('ARXIV_JSONL','arxiv-metadata-oai-snapshot.json'))
target=int(os.getenv('MAX_RECORDS','10000')); rng=random.Random(42)
if not 5000 <= target <= 10000: raise ValueError('MAX_RECORDS має бути від 5000 до 10000')
seen=0; records=[]
with source.open(encoding='utf-8') as f:
    for line in tqdm(f,desc='Читаємо JSONL'):
        if not line.strip(): continue
        p=json.loads(line); title=' '.join(p.get('title','').split()); abstract=' '.join(p.get('abstract','').split())
        cats=(p.get('categories') or '').split()
        if not p.get('id') or not title or not abstract or not cats: continue
        try: year=parsedate_to_datetime(p['versions'][0]['created']).year
        except (KeyError,IndexError,TypeError,ValueError):
            try: year=int(p['update_date'][:4])
            except (KeyError,TypeError,ValueError): continue
        parsed=p.get('authors_parsed') or []
        authors=', '.join(' '.join(filter(None,[a[0].strip(),a[1].strip() if len(a)>1 else ''])) for a in parsed[:10] if a and a[0].strip()) or ' '.join((p.get('authors') or '').split())
        row=dict(id=str(p['id']),title=title,abstract=abstract,authors=authors,year=year,category=cats[0])
        seen+=1
        if len(records)<target: records.append(row)
        else:
            j=rng.randrange(seen)
            if j<target: records[j]=row
if len(records)<5000: raise ValueError(f'У джерелі лише {len(records)} придатних записів')
df=pd.DataFrame(records).sort_values('id').reset_index(drop=True)
if df.id.duplicated().any(): raise ValueError('Дубльовані arXiv ID')
Path('data').mkdir(exist_ok=True); df.to_parquet('data/arxiv_subset.parquet',index=False)
print('Придатних у джерелі:',seen,'Відібрано:',len(df)); print(df.category.value_counts().head(10));print(df.year.describe());print(df.iloc[0].to_dict())
