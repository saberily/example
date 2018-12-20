# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'heat_window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from mysql import MySQL # 导入自定义数据库文件
from PyQt5.QtGui import QPalette, QPixmap, QColor

# 创建自定义数据库对象
mysql = MySQL()
# 连接数据库
sql = mysql.connection_sql()
# 创建游标
cur = sql.cursor()
class Heat_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 开启自动填充背景
        self.centralwidget.setAutoFillBackground(True)
        palette = QPalette()  # 调色板类
        palette.setBrush(QPalette.Background, QtGui.QBrush(QtGui.QPixmap('img/rankings_bg.png')))  # 设置背景图片
        self.centralwidget.setPalette(palette)  # 为控件设置对应的调色板即可
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 801, 71))
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        # 获取热评排行前100名数据信息
        row, column, results = mysql.query_top100_rankings(cur, 'heat_rankings')

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(0, 69, 800, 530))
        # 设置表格内容不可编辑
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.verticalHeader().setHidden(True)  # 隐藏行号
        self.tableWidget.setRowCount(row)  # 根据数据库内容设置表格行
        self.tableWidget.setColumnCount(column)  # 设置表格列
        self.tableWidget.setStyleSheet("background-color:rgba(0,0,0,0)")  # 设置背景透明
        # 设置表格头部
        self.tableWidget.setHorizontalHeaderLabels(['排名', '书名', '京东价', '定价', '出版社'])
        # 根据窗体大小拉伸表格
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        for i in range(row):
            for j in range(column):
                temp_data = results[i][j]  # 临时记录，不能直接插入表格
                data = QtWidgets.QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                self.tableWidget.setItem(i, j, data)
        # 设置表格内容文字大小
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tableWidget.setFont(font)
        mysql.close_sql()  # 提取完数据以后关掉数据库
        self.tableWidget.setObjectName("tableWidget")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "热评排行榜"))
        self.label.setText(_translate("MainWindow", "计算机与互联网图书热评排行榜"))

