import numpy as np
import pandas as pd
from tqdm import tqdm
from common import index,PAPERS

df=pd.read_parquet('data/arxiv_subset.parquet').reset_index(drop=True);v=np.load('embeddings/embeddings.npy')
if v.shape!=(len(df),768): raise ValueError('Кількість або розмірність ембеддингів не відповідає parquet')
ix=index(create=True)
for start in tqdm(range(0,len(df),100),desc='Upsert papers'):
    batch=[]
    for i in range(start,min(start+100,len(df))):
        r=df.iloc[i];batch.append({'id':f'paper_{i}','values':v[i].tolist(),'metadata':{'arxiv_id':str(r.id),'title':str(r.title),'abstract':str(r.abstract)[:500],'authors':str(r.authors)[:200],'year':int(r.year),'category':str(r.category)}})
    ix.upsert(vectors=batch,namespace=PAPERS)
print('Статистика індексу (зміни можуть відображатися із затримкою):',ix.describe_index_stats())
print('Очікується у namespace papers:',len(df),'унікальних векторів; повторний upsert замінює ті самі ID.')
