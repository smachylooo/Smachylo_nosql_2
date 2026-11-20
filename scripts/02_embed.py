from pathlib import Path
import numpy as np
import pandas as pd
from common import get_model, encode, fulltext

df=pd.read_parquet('data/arxiv_subset.parquet');model=get_model()
vectors=encode(model,[fulltext(r) for _,r in df.iterrows()])
assert vectors.shape==(len(df),768) and np.isfinite(vectors).all()
Path('embeddings').mkdir(exist_ok=True);np.save('embeddings/embeddings.npy',vectors)
print('Текстів:',len(df),'Розмірність:',vectors.shape[1], 'Норма першого:',np.linalg.norm(vectors[0]))
