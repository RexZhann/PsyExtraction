from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDirModel, QListView
from Ui_search import Ui_Dialog
import os
from PyQt5 import QtWidgets, QtCore


class MyMainWindow(QMainWindow,Ui_Dialog): 

    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.currentlist = []  # 存储当前选择的文件列表
        self.local_folder = "PsyExtraction\\papers"  # 设置本地文件夹路径
        self.target_folder = "PsyExtraction\\papers"  # 设置目标文件夹路径
        self.populate_corpus_list()  # 初始加载文件夹列表
        self.connect_signals()


    def connect_signals(self):
        self.newCorpus.currentIndexChanged.connect(self.on_corpus_changed)
        self.uploadButton.clicked.connect(self.on_upload_clicked)
        self.searchButton.clicked.connect(self.on_search_clicked)
    

    def populate_corpus_list(self):
        pass

    def on_corpus_changed(self, index):
        # 下拉列表变更时的槽函数
        if self.newCorpus.itemData(index) == "CLEAR":
            self.currentlist = []  # 清空当前选择的文件列表
            self.update_current_corpora_list()  # 更新显示
        else:
            # 获取选中的文件或文件夹路径
            path = self.newCorpus.itemData(index)
            if os.path.isfile(path):
                self.add_to_current_list(path)  # 添加到当前选择的文件列表

    def add_to_current_list(self, path):
        # 将文件添加到currentlist
        if path not in self.currentlist:
            self.currentlist.append(path)
        self.update_current_corpora_list()  # 更新显示

    def update_current_corpora_list(self):
        # 更新currentCorpora的显示
        self.textBrowser.clear()
        for file in self.currentlist:
            self.textBrowser.append(file)  # 将文件名添加到textBrowser

    def on_upload_clicked(self):
        # 上传按钮点击时的槽函数
        target_folder = "PsyExtraction\\papers"  # 设置目标文件夹
        for file in self.currentlist:
            src_file = os.path.join(self.local_folder, file)
            dst_file = os.path.join(target_folder, file)
            QtCore.QDir().copy(src_file, dst_file)  # 复制文件到目标文件夹
            print(f"Uploaded: {src_file} to {dst_file}")

    def on_search_clicked(self):
        # 当搜索按钮被点击时执行
        # 获取lineEdit的文本
        search_text = self.lineEdit.text()
        # 获取comboBox的当前选中文本
        selected_corpus = self.comboBox.currentText()
        
        # 执行搜索操作...
        print(f"Searching for: {search_text} in corpus: {selected_corpus}")
