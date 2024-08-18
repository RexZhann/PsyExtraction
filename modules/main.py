import re
import os
import dashscope
import sys
from PyQt5.QtWidgets import QApplication
from modules import *
from dotenv import load_dotenv


api = [os.getenv('MODEL_API'), os.getenv('VECTOR_API'), os.getenv('ENDPOINT')]

def search_process(keyword, dep_key, api, modl='qwen-max', corpus='PsyExtraction/papers', sum=False):

    # preprocecss

    # read the txt file
    cur_dir = os.getcwd()
    abs_path = os.path.join(cur_dir, corpus)

    # load in files in corpora
    txt_texts = preprocess.read_txts_to_list(abs_path)
    articles = preprocess.find_related_sent(keyword, txt_texts)

    #set api key for generation task
    dashscope.api_key = api[0]
    stop_words = tokenization.stop_words
    ex_words = ex_words = set([word for word in keyword.split() if word not in stop_words])

    # tokenization

    # perform summarization when true
    if sum:
            articles = tokenization.para_sum(articles, keyword, dep_key, modl=modl)
    #perform tokenization
    res = tokenization.tokenizer_batch(articles, keyword, dep_key, modl=modl) 
    phrases_llm = tokenization.remove_sw(res)
    phrases_res = tokenization.exclude_key(phrases_llm, keyword, stop_words)
    
    # dependency
    dep_res = tokenization.dep_reco_batch(articles, phrases_res, keyword, dep_key, modl=modl)
    desc_phrases = tokenization.obtain_dep_phrases(phrases_res, dep_res, ex_words=ex_words)

    # embedding
    collection = embedding.init_coll(emb_api=api[1], endpoint=api[2])

    embedding.process_phrases(desc_phrases, collection)
    rel_ph, rel_emb = embedding.get_phrase_embs(collection, keyword, ex_words)

    # cluster_eval
    similarity_matrix = cluster_eval.plot_heatmap(rel_emb, rel_ph, keyword)
    cluster_eval.plot_H_cluster(similarity_matrix, keyword)
    target_emb = cluster_eval.generate_embeddings(keyword)

    rep_phrases, rep_phrase_emb = cluster_eval.get_rep_phrases_target(similarity_matrix, rel_ph, rel_emb, target_emb, k=6)
    

    return rep_phrases, rep_phrase_emb



keyword = 'vascular dementia'
dep_key = 'diagnostic guidelines'


def main():
    if __name__ == '__main__':
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        mywin = Ui_winlogic.MyMainWindow()
        mywin.show()
        res_ph, re_emb = search_process(keyword, dep_key, api)
        print(res_ph)
        sys.exit(app.exec_())

main()
