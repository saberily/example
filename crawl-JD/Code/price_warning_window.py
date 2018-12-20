# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'price_warning_window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette
class Price_Warning_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 177)
        MainWindow.setMinimumSize(QtCore.QSize(600, 177))
        MainWindow.setMaximumSize(QtCore.QSize(600, 177))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 开启自动填充背景
        self.centralwidget.setAutoFillBackground(True)
        palette = QPalette()  # 调色板类
        palette.setBrush(QPalette.Background, QtGui.QBrush(QtGui.QPixmap('img/price_warning_bg.png')))  # 设置背景图片
        self.centralwidget.setPalette(palette)  # 为控件设置对应的调色板即可
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 600, 177))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tableWidget.setFont(font)
        self.tableWidget.setLineWidth(1)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers) # 表格内容不可编辑
        self.tableWidget.setRowCount(3)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setStyleSheet("background-color:rgba(0,0,0,0)")  # 设置背景透明
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)   # 字体加粗
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)   # 字体加粗
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.tableWidget.setItem(2, 1, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(299) # 默认列宽度
        self.tableWidget.horizontalHeader().setMinimumSectionSize(25)  # 最小列宽度
        self.tableWidget.horizontalHeader().setStretchLastSection(False)  # 关闭拉伸
        self.tableWidget.verticalHeader().setVisible(False)           # 不显示行号
        self.tableWidget.verticalHeader().setDefaultSectionSize(50)    # 默认行高度
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "价格预警窗口"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "关注图书的名称"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "价格变化信息"))
        __sortingEnabled = self.tableWidget.isSortingEnabled()
        self.tableWidget.setSortingEnabled(False)   # 关闭排序
        item = self.tableWidget.item(0, 0)
        item.setText(_translate("MainWindow", "无"))
        item = self.tableWidget.item(0, 1)
        item.setText(_translate("MainWindow", "无"))
        item = self.tableWidget.item(1, 0)
        item.setText(_translate("MainWindow", "无"))
        item = self.tableWidget.item(1, 1)
        item.setText(_translate("MainWindow", "无"))
        item = self.tableWidget.item(2, 0)
        item.setText(_translate("MainWindow", "无"))
        item = self.tableWidget.item(2, 1)
        item.setText(_translate("MainWindow", "无"))
        self.tableWidget.setSortingEnabled(__sortingEnabled)

