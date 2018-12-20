# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'attention_window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Attention_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 200)
        MainWindow.setMinimumSize(QtCore.QSize(400, 200))  # 主窗体最小值
        MainWindow.setMaximumSize(QtCore.QSize(400, 200))  # 主窗体最大值
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(50, 70, 301, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_no = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_no.setGeometry(QtCore.QRect(250, 125, 100, 50))
        self.pushButton_no.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_no.setObjectName("pushButton_no")
        self.pushButton_yes = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_yes.setGeometry(QtCore.QRect(50, 125, 100, 50))
        self.pushButton_yes.setMinimumSize(QtCore.QSize(100, 50))
        self.pushButton_yes.setObjectName("pushButton_yes")


        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "关注窗体"))

