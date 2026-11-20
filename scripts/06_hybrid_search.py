import re
import pandas as pd
from rank_bm25 import BM25Okapi
from common import index,get_model,encode,PAPERS

TOP=10
words=lambda t: re.findall(r'\w+',t.lower())
df=pd.read_parquet('data/arxiv_subset.parquet').reset_index(drop=True)
bm=BM25Okapi([words(f'{r.title} {r.abstract} {r.authors}') for r in df.itertuples()])
ix=index();model=get_model()
def rrf(lists,k=60):
 scores={}
 for seq in lists:
  for rank,item in enumerate(seq,1):scores[item]=scores.get(item,0)+1/(k+rank)
 return sorted(scores.items(),key=lambda x:(-x[1],x[0]))
for q in ['BERT fine-tuning','Yann LeCun convolutional networks','making computers understand human emotions from text']:
 scores=bm.get_scores(words(q));lex=sorted(range(len(df)),key=lambda i:(-scores[i],i))[:TOP]
 query=encode(model,[q])[0];hits=ix.query(vector=query.tolist(),top_k=TOP,include_metadata=False,namespace=PAPERS).matches
 dense=[int(h.id.split('_',1)[1]) for h in hits]
 print('\nQUERY:',q)
 for label,ids in [('BM25',lex[:5]),('vector',dense[:5]),('RRF',[i for i,_ in rrf([lex,dense])[:5]])]:
  print(label)
  fusion=dict(rrf([lex,dense]))
  for i in ids: print(' ',i,df.iloc[i].title,round(fusion.get(i,0),5) if label=='RRF' else '')
