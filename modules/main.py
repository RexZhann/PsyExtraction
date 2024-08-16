from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import dashscope
from dashscope import TextEmbedding
from dashvector import Client, Doc
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from dashscope import Generation
import fitz
from pysbd import Segmenter
from tqdm import tqdm
import sys
from PyQt5.QtWidgets import QApplication
from modules import *


api = ['sk-b8ba4c6e0c9b4e7697fd3cdaaebe63f7', 
       'sk-0R9jI2212bV7Ma7u7ba4UmX8QQ8Gt2E2060C14D5411EF9A1EB61E393DC850', 
       'vrs-cn-0mm3ulqnq00032.dashvector.cn-hangzhou.aliyuncs.com'
       ]

def search_process(keyword, dep_key, api, modl='qwen-max', corpus='PsyExtraction\\papers', sum=False):

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
    phrases_llm = [re.findall(r'"(.*?)"', phrase) for phrase in res]
    phrases_llm = tokenization.remove_sw(phrases_llm)
    phrases_res = tokenization.exclude_key(phrases_llm)
    dep_res = tokenization.dep_reco_batch(articles, phrases_res, keyword, dep_key, modl=modl)
    desc_phrases = tokenization.obtain_dep_phrases(phrases_res, dep_res, ex_words)
    # dependency
    # embedding
    # cluster_eval



def main():
    if __name__ == '__main__':
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        mywin = Ui_winlogic.MyMainWindow()
        mywin.show()
        sys.exit(app.exec_())

main()
