from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDirModel, QListView
from .Ui_search import Ui_Dialog
import os
from PyQt5 import QtWidgets, QtCore
from .main import search_process, api
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from .Ui_setting import Ui_Dialog_setting


class MyMainWindow(QMainWindow,Ui_Dialog): 

    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.local_folder = os.path.join(os.getcwd(),"nlp\\PsyExtraction\\papers")  # 设置本地文件夹路径
        self.currentlist = os.listdir(self.local_folder)  # 存储当前选择的文件列表
        self.populate_corpus_list()  # 初始加载文件夹列表
        self.connect_signals()
        self.update_current_corpora_list()
        self.is_searching = False
        self.win_size = 5
        self.topk = 15
        self.n_char = 6


    _translate = QtCore.QCoreApplication.translate

    def connect_signals(self):
        self.newCorpus.currentIndexChanged.connect(self.on_corpus_changed)
        self.uploadButton.clicked.connect(self.on_upload_clicked)
        self.searchButton.clicked.connect(self.on_search_clicked)
        self.apiButton.clicked.connect(self.on_api_clicked)
        self.advanceButton.clicked.connect(self.on_advance_clicked)

    def populate_corpus_list(self):
        self.currentlist = os.listdir(self.local_folder)

    def on_corpus_changed(self, index):
        # 下拉列表变更时的槽函数
        _translate = QtCore.QCoreApplication.translate
        if self.newCorpus.itemData(index) == "CLEAR":
            self.currentlist = []  # 清空当前选择的文件列表
            self.update_current_corpora_list()  # 更新显示
        else:
            # 获取选中的文件或文件夹路径
            for file in self.currentlist:
                self.newCorpus.addItem("")
                self.newCorpus.setItemText(0, _translate("Dialog", file)) # 添加到当前选择的文件列表

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
        if self.is_searching:  # 检查是否已经在搜索
            return
        
        self.is_searching = True

        try:
            if self.keyword.text() != None:
                search_text = self.keyword.text()
                # 获取comboBox的当前选中文本
                selected_dep = self.dep_key.currentText()
                # 执行搜索操作...
                ts_phrases, rep_phrases, rep_phrase_emb = search_process(search_text, selected_dep, api, win_size=self.win_size, topk=self.topk, n_char=self.n_char)
                plot_window = PlotWindow()
                plot_window.show()
            else:
                raise KeyError("no key provided")
        
        finally:
            self.is_searching = False
        
    
    def on_api_clicked(self):
        
        env_path = os.path.join(os.getcwd(), "nlp\\PsyExtraction\\.env")
        if os.path.exists(env_path):
            # 使用系统的默认文本编辑器打开文件
            subprocess.run(['notepad.exe', env_path], check=True)
        else:
            # 如果文件不存在，可以打印一条消息或者进行其他错误处理
            print(f"The file {env_path} does not exist.")
            
    
    def on_advance_clicked(self):
        settings_dialog = Ui_Dialog_setting(self)  # 'self' 是当前主窗口的实例
        if settings_dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.win_size, self.topk, self.n_char = settings_dialog.get_settings()
        else:
            pass
            


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.initUI()

    def initUI(self):
        # 创建一个垂直布局
        self.layout = QtWidgets.QVBoxLayout(self)

        # 创建 matplotlib 图形
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # 创建 DataFrame 显示控件
        self.table = QtWidgets.QTableWidget()
        self.layout.addWidget(self.table)

        # 假设你有一些数据要显示
        # 这里使用随机数据作为示例
        self.populate_table(pd.DataFrame({
            'Column1': np.random.rand(10),
            'Column2': np.random.rand(10)
        }))

    def populate_table(self, df):
        # 将 DataFrame 填充到 QTableWidget
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                self.table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(f"{value:.2f}"))