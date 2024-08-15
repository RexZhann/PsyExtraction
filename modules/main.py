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
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from Ui_search import Ui_Dialog
from modules import cluster_eval


def search_process(keyword, dep_key, corpora, modl='qwen-max'):
    pass

class MyMainWindow(QMainWindow,Ui_Dialog): #这里也要记得改
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
    
        self.uploadButton.clicked.connect(self.on_upload_clicked)
        self.searchButton.clicked.connect(self.on_search_clicked)

    def on_upload_clicked(self):
        # 当上传按钮被点击时执行
        # 获取lineEdit的文本
        search_text = self.lineEdit.text()
        # 获取comboBox_2的当前选中文本
        
        
        # 执行上传操作...
        print(f"Uploading with search text: {search_text}")

    def on_search_clicked(self):
        # 当搜索按钮被点击时执行
        # 获取lineEdit的文本
        keyword = self.keyword.text()
        # 获取comboBox的当前选中文本
        
        selected_corpus = self.currentCorpora.items()
        dep_key = self.dep_key.currentText()
        
        # 执行搜索操作...
        print(f"Searching for: {keyword} in corpus: {selected_corpus}")



def main():
    if __name__ == '__main__':
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        mywin = MyMainWindow()
        mywin.show()
        sys.exit(app.exec_())

main()
