# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluation_chart_window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from chart import PlotCanvas # 导入自定义图表文件中的画布类
from mysql import MySQL   # 导入自定义数据库文件


class Press_Bar_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(390, 310)
        MainWindow.setMinimumSize(QtCore.QSize(390, 310))
        MainWindow.setMaximumSize(QtCore.QSize(390, 310))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 390, 310))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayoutWidget_0 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_0.setGeometry(QtCore.QRect(0, 0, 390,310))
        self.horizontalLayoutWidget_0.setObjectName("horizontalLayoutWidget_0")
        self.horizontalLayout_0 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_0)
        self.horizontalLayout_0.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_0.setObjectName("horizontalLayout_0")
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayoutWidget_1 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_1.setGeometry(QtCore.QRect(0, 0, 390, 310))
        self.horizontalLayoutWidget_1.setObjectName("horizontalLayoutWidget_1")
        self.horizontalLayout_1 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_1)
        self.horizontalLayout_1.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_1.setObjectName("horizontalLayout_1")
        self.tabWidget.addTab(self.tab_2, "")

        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 390,310))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "关注图书出版社占有比例"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "关注图书-1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "关注图书-2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "关注图书-3"))
        self.evaluation_chart()

    def evaluation_chart(self):
        # 销量榜数据表名称
        sales_volume_rankings_table_name = 'sales_volume_rankings'
        # 创建自定义数据库对象
        mysql = MySQL()
        # 连接数据库
        sql = mysql.connection_sql()
        # 创建游标
        cur = sql.cursor()
        # 获取关注商品的出版社名称与图书名称
        attention_message = mysql.query_attention(cur, 'press,book_name',
                                                  sales_volume_rankings_table_name, "attention = '1'")
        for i in range(len(attention_message)):
            query_sql = "select press,count(*) from sales_volume_rankings " \
                        "group by press having press = '{name}'".\
                format(name =attention_message[i][0])
            # 查询关注商品出版社占有比例
            number, press = mysql.query_press_proportion(cur,query_sql)
            # 计算其它比例并添加至列表中
            number.append((100-number[0]))
            press.append('其它')
            # 关注的第一个商品
            if i == 0:
                plt1 = PlotCanvas()  # 创建如表画布类对象
                # 显示出版社占有比例图
                plt1.bar(number,press,attention_message[i][1])
                # 将出版社占有比例图添加至布局中
                self.horizontalLayout_0.addWidget(plt1)
            # 关注的第二个商品
            if i == 1:
                plt2 = PlotCanvas()
                plt2.bar(number, press, attention_message[i][1])
                self.horizontalLayout_1.addWidget(plt2)
            # 关注的第三个商品
            if i == 2:
                plt3 =PlotCanvas()
                plt3.bar(number, press, attention_message[i][1])
                self.horizontalLayout_2.addWidget(plt3)
        mysql.close_sql() #关闭数据库




