from datetime import datetime
import numpy as np
import pandas as pd
from common import index,get_model,encode,PAPERS

ix=index();model=get_model();df=pd.read_parquet('data/arxiv_subset.parquet').reset_index(drop=True);v=np.load('embeddings/embeddings.npy')
if v.shape!=(len(df),768):raise ValueError('Невідповідність векторів і даних')
def search(query, where=None):
 q=encode(model,[query])[0];args=dict(vector=q.tolist(),top_k=5,namespace=PAPERS,include_metadata=True)
 if where:args['filter']=where
 result=ix.query(**args); print('\nЗапит:',query,'фільтр:',where)
 for x in result.matches:
  m=x.metadata;print(x.id,round(x.score,4),m['title'],m['category'],int(m['year']),m['abstract'][:180])
 if not result.matches:print('Збігів не знайдено: перевірте представництво категорій і років у підмножині.')
search('teaching machines to recognize objects in pictures')
search('reinforcement learning',{'$and':[{'year':{'$gte':datetime.now().year-5}},{'category':{'$eq':'cs.LG'}}]})
search('reinforcement learning',{'year':{'$lt':2015}})
q=encode(model,['teaching machines to recognize objects in pictures'])[0];dot=v@q
norm=np.linalg.norm(v,axis=1)*np.linalg.norm(q);cos=dot/np.maximum(norm,1e-12);l2=np.linalg.norm(v-q,axis=1)
for name,order in [('cosine',np.argsort(-cos)[:5]),('dot',np.argsort(-dot)[:5]),('L2',np.argsort(l2)[:5])]:
 print('\n',name,[(int(i),str(df.iloc[i].title)) for i in order])
print('cosine/dot top-5:',np.array_equal(np.argsort(-cos)[:5],np.argsort(-dot)[:5]))
