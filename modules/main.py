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
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from Ui_search import Ui_Dialog
from modules import *


def search_process(keyword, dep_key, corpus, api, modl='qwen-max'):
    
    pass

class MyMainWindow(QMainWindow,Ui_Dialog): 

    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
    
        self.uploadButton.clicked.connect(self.on_upload_clicked)
        self.searchButton.clicked.connect(self.on_search_clicked)

    def on_upload_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                    "上传文件",  # 对话框标题
                                                    "",  # 起始目录，默认打开的文件夹路径
                                                    "All Files (*);;Text Files (*.txt)")  # 文件过滤器
        
        # 检查用户是否选择了文件
        if file_path:  # 如果用户没有取消操作，file_path将不是空的
            # 在这里处理文件，例如获取文件路径
            print(f"Selected file: {file_path}")

            # 可以在这里添加代码来处理上传的文件
            # 例如，使用Python的内置open函数来读取文件内容
            # with open(file_path, 'r') as file:
            #     content = file.read()
            #     print(content)
        

    def on_search_clicked(self):


        keyword = self.keyword.text()
        corpus = self.currentCorpora.items()
        dep_key = self.dep_key.currentText()
        # 临时placeholder变量
        api = [0, 0, 0]
        modl = 'qwen-max'
        
        search_process(keyword, dep_key, corpus, api, modl)

        print(f"Searching for: {keyword} in corpus: {corpus}")



def main():
    if __name__ == '__main__':
        # 创建Qt应用程序实例
        app = QApplication(sys.argv)
        mywin = MyMainWindow()
        mywin.show()
        sys.exit(app.exec_())

main()
