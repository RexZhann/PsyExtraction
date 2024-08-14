from dashvector import Client, Doc
from dashscope import TextEmbedding


def generate_embeddings(text):
    rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v2,
                             input=text)
    embeddings = [record['embedding'] for record in rsp.output['embeddings']]
    return embeddings if isinstance(text, list) else embeddings[0]


def init_coll(emb_api, endpoint, col_name='sample1'):

    client = Client(
    api_key=emb_api,
    endpoint=endpoint
    )


    collection = client.get('sample1')
    collection.delete(delete_all=True)  #empty collection
    assert collection
    return collection


def batch_generator(items, batchsize):
    for i in range(0, len(items), batchsize):
        yield items[i:i + batchsize]

def process_phrases(phrases_llm, collection, batchsize=6):
    for batch in batch_generator(phrases_llm, batchsize):
        
        for sent in batch:
            embeddings = [generate_embeddings(phrase) for phrase in sent]

            
            docs = [Doc(id=str(len(phrase)), vector=embedding, fields={"title": phrase}) 
                    for phrase, embedding in zip(sent, embeddings)]
        
        rsp = collection.insert(docs)
        assert rsp


def get_phrase_embs(collection, keyword, ex_words):
    rsp = collection.query(generate_embeddings(keyword), output_fields=['title'], topk=15)



    related_phrases = [doc.fields['title'] for doc in rsp.output]


    for ex in ex_words:
        rsp1 = collection.query(generate_embeddings(ex), output_fields=['title'], topk=10)
        related_phrases += [doc.fields['title'] for doc in rsp1.output]
    
    related_phrases_embeddings = [generate_embeddings(phrase) for phrase in related_phrases]


    return related_phrases, related_phrases_embeddings
