"""Shared configuration; namespaces isolate article and chunk records."""
import os, time
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, models

load_dotenv()
MODEL = 'allenai/specter2_base'
INDEX = 'arxiv-papers'
DIM = 768
PAPERS = 'papers'

def get_model():
    # Explicit CLS pooling follows the model card. SentenceTransformer's
    # fallback for a bare Transformers model otherwise uses mean pooling.
    transformer = models.Transformer(MODEL, max_seq_length=512)
    pooling = models.Pooling(transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=True, pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False)
    return SentenceTransformer(modules=[transformer, pooling])

def encode(model, texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True).astype('float32')

def index(create=False):
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    if INDEX not in pc.list_indexes().names():
        if not create:
            raise RuntimeError('Індекс відсутній: спочатку виконайте 03_load_to_pinecone.py')
        pc.create_index(name=INDEX, dimension=DIM, metric='cosine', spec=ServerlessSpec(
            cloud=os.getenv('PINECONE_CLOUD','aws'), region=os.getenv('PINECONE_REGION','us-east-1')))
    desc = pc.describe_index(INDEX)
    if desc.dimension != DIM or desc.metric != 'cosine':
        raise ValueError('Наявний індекс має іншу розмірність або метрику. Оберіть інший індекс.')
    for _ in range(120):
        if pc.describe_index(INDEX).status['ready']:
            return pc.Index(INDEX)
        time.sleep(2)
    raise TimeoutError('Індекс не готовий протягом 4 хвилин')

def fulltext(row):
    return f"{row['title']} [SEP] {row['abstract']}"
