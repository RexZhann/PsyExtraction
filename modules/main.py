import re
import os
import dashscope
import sys
from PyQt5.QtWidgets import QApplication
from modules import *
from dotenv import load_dotenv

load_dotenv()

api = [os.getenv('MODEL_API'), os.getenv('VECTOR_API'), os.getenv('ENDPOINT')]

def search_process(keyword, dep_key, api, win_size=5, topk=15, n_char=6, modl='qwen-max', corpus='nlp\\PsyExtraction\\papers', sum=False):

    # preprocecss
    print(f"Searching for {keyword} in the corpora...")
    # read the txt file
    cur_dir = os.getcwd()
    abs_path = os.path.join(cur_dir, corpus)

    # load in files in corpora
    txt_texts = preprocess.read_txts_to_list(abs_path)
    articles = preprocess.find_related_sent(keyword, txt_texts, win_size=win_size)
    if len(articles) == 0:
        print("No related sentences found in the corpora")
        return None, None, None

    #set api key for generation task
    dashscope.api_key = api[0]
    stop_words = tokenization.stop_words
    ex_words = set([word for word in keyword.split() if word not in stop_words])

    # tokenization

    # perform summarization when true
    if sum:
            articles = tokenization.para_sum(articles, keyword, dep_key, modl=modl)
    #perform tokenization
    print("Tokenizing the articles...")
    res = tokenization.tokenizer_batch(articles, keyword, dep_key, modl=modl) 
    phrases_llm = tokenization.remove_sw(res)
    phrases_res = tokenization.exclude_key(phrases_llm, keyword, stop_words)
    
    # dependency
    print("Extracting dependency phrases...")
    dep_res = tokenization.dep_reco_batch(articles, phrases_res, keyword, dep_key, modl=modl)
    desc_phrases = tokenization.obtain_dep_phrases(phrases_res, dep_res, ex_words=ex_words)

    # embedding
    print("Embedding the phrases...")
    collection = embedding.init_coll(emb_api=api[1], endpoint=api[2])

    embedding.process_phrases(desc_phrases, collection)
    rel_ph, rel_emb = embedding.get_phrase_embs(collection, keyword, ex_words, topk=topk)

    # cluster
    print("Plotting the clusters...")
    similarity_matrix = cluster_eval.plot_heatmap(rel_emb, rel_ph, keyword)
    cluster_eval.plot_H_cluster(similarity_matrix, keyword)
    target_emb = cluster_eval.generate_embeddings(keyword)

    rep_phrases, rep_phrase_emb = cluster_eval.get_rep_phrases_target(similarity_matrix, rel_ph, rel_emb, target_emb, k=n_char)
    ts_phrases = cluster_eval.translate_cn(rep_phrases)

    cos = cluster_eval.average_cos_sim(rep_phrase_emb)
    print(f'The average cosine sim score: {cos}')
    dot = cluster_eval.average_dot_product(rep_phrase_emb)
    print(f'the average dot product is : {dot}')

    collection = embedding.init_coll(emb_api=api[1], endpoint=api[2])
    '''retr_res = cluster_eval.retriever_batch(articles, rep_phrases, batchsize=6, modl=modl)
    re1, sco1 = cluster_eval.eval_retrieval_llm(retr_res, keyword, collection)
    print(f"retrieval result from LLMs is : {re1}, with similarity score {sco1}")
    re2, sco2 = cluster_eval.eval_retrieval_que(retr_res, keyword, collection)
    print(f"\nretrieval result from query is : {re2}, with similarity score {sco2}")'''

    exp, tr_df, ld_df, ess_features = cluster_eval.pca_features(ts_phrases, rep_phrase_emb)
    print("Transformed Embeddings:")
    print(tr_df)

    print("\nExplained Variance Ratio:")
    print(exp)

    print("\nLoadings (Contribution of Original Dimensions to Principal Components):")
    print(ld_df)

    print("Loadings (Contribution of Original Dimensions to Principal Components):")
    print(ld_df)

    print("\nArgmax of each Principal Component:")
    for i, feature in enumerate(ess_features):
        print(f"主成分 {i+1}: {feature}")

    return ts_phrases, rep_phrases, rep_phrase_emb


keyword = 'vascular dementia'
dep_key = 'diagnostic guidelines'


def main():
    if __name__ == '__main__':
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        mywin = Ui_winlogic.MyMainWindow()
        mywin.show()
        sys.exit(app.exec_())

main()
