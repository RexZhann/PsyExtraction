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
from .modules import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget


if __name__ == '__main__':
    # 创建Qt应用程序实例
    app = QApplication(sys.argv)

    # 创建一个QWidget对象，作为主窗口
    w = QWidget()
    w.resize(250, 150)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()

    # 运行Qt应用程序
    sys.exit(app.exec_())
