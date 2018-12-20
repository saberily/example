# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window_main.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QPixmap, QColor # 导入调色板、位图、颜色

from mysql import MySQL  # 导入自定义数据库文件
from crawl import Crawl,rankings_list  # 导入自定义爬取文件
from chart import *  # 导入自定义图表文件

# 销量榜数据表名称
sales_volume_rankings_table_name = 'sales_volume_rankings'
# 热评榜数据表名称
heat_rankings_table_name = 'heat_rankings'
# 计算机与互联网图书销量榜地址
sales_volume_url = \
    'http://book.jd.com/booktop/0-0-0.html?category=3287-0-0-0-10001-{page}'
# 计算机与互联网图书热评排行榜地址
heat_rankings_url = \
    'http://book.jd.com/booktop/0-0-1.html?category=3287-0-0-1-10001-{page}'
# 创建自定义数据库对象
mysql = MySQL()
# 创建爬去对象
mycrawl = Crawl()
# 连接数据库
sql = mysql.connection_sql()
# 创建游标
cur = sql.cursor()

attention_warning_message_list = [] # 关注预警信息

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1070, 713)
        MainWindow.setMinimumSize(QtCore.QSize(1070, 713))  # 主窗体最小值
        MainWindow.setMaximumSize(QtCore.QSize(1070, 713))  # 主窗体最大值
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 开启自动填充背景
        self.centralwidget.setAutoFillBackground(True)
        palette = QPalette()  # 调色板类
        palette.setBrush(QPalette.Background, QtGui.QBrush(QtGui.QPixmap('img/window_main_bg.png')))  # 设置背景图片
        self.centralwidget.setPalette(palette)  # 为控件设置对应的调色板即可

        self.treeWidget = QtWidgets.QTreeWidget(self.centralwidget)
        self.treeWidget.setGeometry(QtCore.QRect(0, 50, 230, 561))
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.setStyleSheet("background-color:rgba(244,244,244,2)")  # 设置背景透明
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_0.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_0.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_0.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_0 = QtWidgets.QTreeWidgetItem(self.treeWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_0.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(13)
        item_1.setFont(0, font)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        font = QtGui.QFont()
        font.setPointSize(12)
        item_1.setFont(0, font)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1070, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        # 显示评价分析与出版社占有比例饼图布局区域
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(230, 50, 841, 271))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        # 开启自动填充背景
        self.horizontalLayoutWidget.setAutoFillBackground(True)
        palette = QPalette()  # 调色板类
        palette.setColor(QPalette.Background, QtCore.Qt.red)  # 设置背景颜色
        self.horizontalLayoutWidget.setPalette(palette)  # 为控件设置对应的调色板即可

        self.pie_horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.pie_horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.pie_horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.pie_horizontalLayout.setSpacing(5)
        self.pie_horizontalLayout.setObjectName("pie_horizontalLayout")


        # 显示前10名价格走势折线图布局区域
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(230, 320, 551, 371))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.line_horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.line_horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.line_horizontalLayout.setSpacing(10)
        self.line_horizontalLayout.setObjectName("line_horizontalLayout")

        # 更新预警信息按钮显示区域
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(-1, 610, 231, 80))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(21, 10, 190, 60))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")

        self.pushButton.setStyleSheet("background-color:rgba(244,244,244,2)")  # 设置背景透明
        self.pushButton.setIcon(QtGui.QIcon('img/update_btn_bg.png'))  # 设置更新预警信息按钮的背景图片
        self.pushButton.setIconSize(QtCore.QSize(190, 60))  # 设置按钮背景图大小
        self.treeWidget.expandAll()  # 树形菜单全部展开

        # 显示前十名图书名称的列表
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(780, 320, 291, 371))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.listView.setFont(font)
        self.listView.setObjectName("listView")
        # 设置列表内容不可编辑
        self.listView.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.listView.setWordWrap(True) # 自动换行



        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1070, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.action_about = QtWidgets.QAction(MainWindow)
        self.action_about.setObjectName("action_about")
        self.action_exits = QtWidgets.QAction(MainWindow)
        self.action_exits.setObjectName("action_exits")
        self.menu.addAction(self.action_about)
        self.menu.addAction(self.action_exits)
        self.menubar.addAction(self.menu.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 指定菜单栏退出的事件处理方法,实现关闭当前窗体
        self.action_exits.triggered.connect(MainWindow.close)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "看店宝（京东商城版）"))
        self.treeWidget.headerItem().setText(0, _translate("MainWindow", "功能列表"))
        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        self.treeWidget.topLevelItem(0).setText(0, _translate("MainWindow", "商品排行功能"))
        self.treeWidget.topLevelItem(0).child(0).setText(0, _translate("MainWindow", "图书销量排行--Top100"))
        self.treeWidget.topLevelItem(0).child(1).setText(0, _translate("MainWindow", "图书热评排行--Top100"))
        self.treeWidget.topLevelItem(1).setText(0, _translate("MainWindow", "商品营销预警功能"))
        self.treeWidget.topLevelItem(1).child(0).setText(0, _translate("MainWindow", "关注商品中、差评预警"))
        self.treeWidget.topLevelItem(1).child(1).setText(0, _translate("MainWindow", "关注商品价格变化预警"))
        self.treeWidget.topLevelItem(2).setText(0, _translate("MainWindow", "关注商品图表分析"))
        self.treeWidget.topLevelItem(2).child(0).setText(0, _translate("MainWindow", "评价分析饼图"))
        self.treeWidget.topLevelItem(2).child(1).setText(0, _translate("MainWindow", "出版社占有比例"))
        self.treeWidget.topLevelItem(3).setText(0, _translate("MainWindow", "关注商品"))
        self.treeWidget.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("MainWindow", "看店宝（京东商城版）"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.action_about.setText(_translate("MainWindow", "关于"))
        self.action_exits.setText(_translate("MainWindow", "退出"))

        # 查询关注商品的预警信息
        attention_warning_message =mysql.query_attention\
            (cur, 'jd_id,middle_time,poor_time,attention_price,attention',
             sales_volume_rankings_table_name, "attention = '1'")
        # 遍历并将预警信息添加至列表当中
        for a in attention_warning_message:
            attention_warning_message_list.append(a)
        # 获取销量排行数据
        id_str = mycrawl.get_rankings(sales_volume_url)
        # 发送网络请求获取价格
        mycrawl.get_price(id_str)
        # 清理数据表
        mysql.query_empty(cur, sales_volume_rankings_table_name)
        # 将销量排行数据添加至数据库中
        mysql.insert(cur,rankings_list,sales_volume_rankings_table_name)
        # 获取热品排行数据
        id_str1 = mycrawl.get_rankings(heat_rankings_url)
        # 发送网络请求获取价格
        mycrawl.get_price(id_str1)
        # 清理数据表
        mysql.query_empty(cur, heat_rankings_table_name)
        # 将热评排行数据添加至数据库中
        mysql.insert(cur,rankings_list,heat_rankings_table_name)
        # 查看销量排行数据为多少条
        sales_number = mysql.query_is_number(cur, sales_volume_rankings_table_name)
        print(sales_number)
        # 遍历预警信息，
        for index, item in enumerate(attention_warning_message_list):
            middle_time = item[1] # 中评时间信息
            poor_time = item[2]   # 差评时间信息
            price = item[3]       # 京东价格信息
            attention = item[4]   # 关注标记
            # 数据表中需要更新的字段
            up = "middle_time='{mi_time}',poor_time='{p_time}'," \
                 "attention_price='{price}',attention='{attention}'" \
                .format(mi_time=middle_time, p_time=poor_time,
                        price=price, attention=attention)
            # 更新关注商品的预警信息
            mysql.update_attention(cur,sales_volume_rankings_table_name,up,
                                   "jd_id={jd_id}".format(jd_id = item[0]))
        # 查询已经关注的图书数量
        attention_number = mysql.query_attention(cur, 'count(*)',
                                                 sales_volume_rankings_table_name, "attention = '1'")[0][0]
        print('排行数据库已更新！')
        # 查看热评排行数据为多少条
        heat_number = mysql.query_is_number(cur, heat_rankings_table_name)
        if  sales_number!=0 and heat_number!=0:
            print('显示数据')
            self.show_message()  # 显示数据
            self.show_attention_book_name(attention_number)
        else:
            print('数据库信息异常！')




    #
    def show_message(self):
        # 获取排行第一商品的京东id
        top1_id = mysql.query_top1_id(cur)
        # 销量前十名图书名称的列表
        list = mysql.query_top10_book_name(cur)
        good, time = mycrawl.get_evaluation(0, top1_id)
        # 创建饼图画布
        pie = PlotCanvas()
        # 销量第一图书名称
        top1_name = list[0]
        # 显示销量第一名的评价分析图
        pie.pie_chart(good, (100 - good), top1_name)
        # 将评价分析图添加至布局中
        self.pie_horizontalLayout.addWidget(pie)
        # 创建水平条形图画布
        bar = PlotCanvas()
        # 查询出版社名称及数量的sql语句
        query_sql = "select press,count(*) from sales_volume_rankings group by press"
        # 查询出版社名称及对应的比例数量
        number, press = mysql.query_press_proportion(cur,query_sql)
        # 显示前100名出版社占有比例图
        bar.bar(number, press,"前100名出版社占有比例")
        # 将出版社占有比例的水平条形图添加至布局中
        self.pie_horizontalLayout.addWidget(bar)
        # 获取前10名图书的京东价格
        str_y = mysql.query_top10_jd_price(cur)
        # 将数据库中的价格字符串列表转换为float类型的列表
        y = [float(f) for f in str_y]
        # 创建画布对象
        line = PlotCanvas()
        # 显示前10名价格折线图
        line.broken_line(y)
        self.line_horizontalLayout.addWidget(line) # 将折线图添加至水平布局当中
        model = QtCore.QStringListModel()  # 创建字符串列表模式
        model.setStringList(list)  # 设置字符串列表
        self.listView.setModel(model)  # 设置模式
    def show_attention_book_name(self,attention_number):
        self.treeWidget.topLevelItem(3).child(0).setText(0, "无")
        self.treeWidget.topLevelItem(3).child(1).setText(0, "无")
        self.treeWidget.topLevelItem(3).child(2).setText(0, "无")
        # 关注图书的数据库中如果存在数据，就获取关注的图书名称并显示出来
        if attention_number!=0:
            for i in range(attention_number):
                name = mysql.query_attention(cur, 'book_name',
                            sales_volume_rankings_table_name, "attention = '1'")[i][0]
                self.treeWidget.topLevelItem(3).child(i).setText(0, name)
        mysql.close_sql() # 关闭数据库

