import re
import pandas as pd
from tqdm import tqdm
from common import index,get_model,encode

WORDS=180;OVERLAP=30

def fixed(text):
    words=text.split();step=WORDS-OVERLAP
    return [' '.join(words[i:i+WORDS]) for i in range(0,len(words),step) if i==0 or i+OVERLAP<len(words)]
def sentence_chunks(text):
    # Sentence-boundary grouping, not embedding-based semantic segmentation.
    sentences=re.split(r'(?<=[.!?])\s+',text.strip());chunks=[];buf=[];n=0
    for s in sentences:
        words=s.split()
        if not words:continue
        if n+len(words)>WORDS and buf: chunks.append(' '.join(buf));buf=[];n=0
        # Extremely long sentences are split to prevent token truncation.
        while len(words)>WORDS:
            chunks.append(' '.join(words[:WORDS]));words=words[WORDS:]
        if words:buf.append(' '.join(words));n+=len(words)
    if buf:chunks.append(' '.join(buf))
    return chunks

if __name__=='__main__':
 df=pd.read_parquet('data/arxiv_subset.parquet').copy();df['words']=df.abstract.str.split().str.len();df=df.nlargest(30,'words')
 ix=index();model=get_model();print('30 найдовших анотацій; слова:',df.words.min(),'–',df.words.max())
 for name,splitter in [('chunks-fixed',fixed),('chunks-semantic',sentence_chunks)]:
  records=[];texts=[]
  for _,r in df.iterrows():
   for j,chunk in enumerate(splitter(r.abstract)):
    records.append({'id':f"{r.id.replace('/','_')}_{j}",'metadata':{'arxiv_id':str(r.id),'title':str(r.title),'chunk':chunk[:8000],'chunk_number':j,'year':int(r.year),'category':str(r.category)}});texts.append(f'{r.title} [SEP] {chunk}')
  vectors=encode(model,texts)
  for start in tqdm(range(0,len(records),100),desc=name):
   batch=[dict(records[i],values=vectors[i].tolist()) for i in range(start,min(start+100,len(records)))]
   ix.upsert(vectors=batch,namespace=name)
  print(name,'чанків:',len(records),'середній розмір:',round(sum(len(t.split()) for t in texts)/len(texts),1))
  for q in ['machine learning for images','quantum field theory','physics of black holes']:
   vec=encode(model,[q])[0];matches=ix.query(vector=vec.tolist(),top_k=5,namespace=name,include_metadata=True).matches
   print('\n',name,q)
   for m in matches:print(round(m.score,4),m.metadata['title'],m.metadata['chunk'][:150])
