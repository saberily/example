from PyQt5.QtWidgets import QMainWindow, QApplication

from sales_window import Sales_MainWindow  # 导入销量排行榜窗体文件中的ui类
from heat_window import Heat_MainWindow  # 导入热评排行榜窗体文件中的ui类

from window_main import Ui_MainWindow  # 导入主窗体文件中的ui类
from attention_window import Attention_MainWindow  # 导入关注窗体文件中的ui类
from evaluate_warning_window import Evaluate_Warning_MainWindow  # 导入评价预警窗体中的ui类
from price_warning_window import Price_Warning_MainWindow  # 导入价格预警窗体中的ui类

from evaluation_chart_window import Evaluation_Chart_MainWindow #导入关注商品评价分析窗体中的ui类

# 导入关注商品出版社占有比例窗体中的ui类
from press_bar_window import Press_Bar_MainWindow
from about_window import About_MainWindow
import sys
from crawl import Crawl  # 导入自定义爬取文件
from mysql import MySQL  # 导入自定义数据库文件
from PyQt5 import QtWidgets, QtCore, QtGui
import requests  # 导入网络请求模块
from PyQt5.QtGui import QPalette  # 导入调色板


# 销量榜数据表名称
sales_volume_rankings_table_name = 'sales_volume_rankings'
# 热评榜数据表名称
heat_rankings_table_name = 'heat_rankings'
# 计算机与互联网图书销量榜地址
sales_volume_url = 'http://book.jd.com/booktop/0-0-0.html?category=3287-0-0-0-10001-{page}'
# 计算机与互联网图书热评排行榜地址
heat_rankings_url = 'http://book.jd.com/booktop/0-0-1.html?category=3287-0-0-1-10001-{page}'


# 显示消息提示框，参数title为提示框标题文字，message为提示信息
def messageDialog(title, message):
    msg_box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, title, message)
    msg_box.exec_()


# 出窗体初始化类
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

    # 左侧功能列表的事件处理方法
    def tree_itemClicked(self):
        # 树形菜单item对象
        item = self.treeWidget.currentItem()
        # 树形菜单item对象
        item = self.treeWidget.currentItem()
        if item.text(0) == '图书销量排行--Top100':
            sales.open()  # 打开销量排行榜窗体

        if item.text(0) == '图书热评排行--Top100':
            heat.open()  # 打开热评排行榜窗体
        if item.text(0) == '关注商品中、差评预警':
            evaluate.__init__()  # 初始化
            evaluate.warning()  # 处理评价预警信息
            evaluate.open()  # 打开评价预警窗体
        if item.text(0) == '关注商品价格变化预警':
            price.__init__()  # 初始化
            price.price()  # 处理价格预警信息
            price.open()  # 打开价格预警窗体

        if item.text(0) == '评价分析饼图':
            evaluation.__init__()  # 初始化
            evaluation.open()  # 打开评价分析饼图窗体

        if item.text(0) == '出版社占有比例':
            press_bar.__init__()  # 初始化
            press_bar.open()  # 打开出版社占有比例窗体

        # 查询已经关注的图书数量
        attention_number = \
            mysql.query_attention(cur, 'count(*)',
                                  sales_volume_rankings_table_name, "attention = '1'")[0][0]
        # 查询已经关注的图书信息在数据库中的id
        attention_id = mysql.query_attention(cur, 'id',
                                             sales_volume_rankings_table_name, "attention = '1'")
        if attention_number != 0:
            for i in range(attention_number):
                # 查询已经关注的图书名称
                name = mysql.query_attention(cur, 'book_name',
                                             sales_volume_rankings_table_name, "attention = '1'")[i][0]
                # 判断选中的名称与关注的名称是否相同
                if item.text(0) == name:
                    # 显示取消关注的图书名称
                    cancel_attention.lineEdit.setText(item.text(0))
                    # 开启自动填充背景
                    cancel_attention.centralwidget.setAutoFillBackground(True)
                    palette = QPalette()  # 调色板类
                    palette.setBrush(QPalette.Background, QtGui.QBrush(
                        QtGui.QPixmap('img/cancel_attention_bg.png')))  # 设置背景图片
                    cancel_attention.centralwidget.setPalette(palette)  # 为控件设置对应的调色板即可
                    cancel_attention.pushButton_yes. \
                        setStyleSheet("background-color:rgba(244,244,244,2)")  # 设置背景透明
                    cancel_attention.pushButton_yes. \
                        setIcon(QtGui.QIcon('img/yes_btn.png'))  # 设置确认关注按钮的背景图片
                    cancel_attention.pushButton_yes. \
                        setIconSize(QtCore.QSize(100, 50))  # 设置按钮背景图大小
                    cancel_attention.pushButton_no. \
                        setStyleSheet("background-color:rgba(244,244,244,2)")  # 设置背景透明
                    cancel_attention.pushButton_no. \
                        setIcon(QtGui.QIcon('img/no_btn.png'))  # 设置确认关注按钮的背景图片
                    cancel_attention.pushButton_no. \
                        setIconSize(QtCore.QSize(100, 50))  # 设置按钮背景图大小
                    # 打开取消关注的窗体
                    cancel_attention.open()
                    # 确认取消关注的按钮的单击事件
                    cancel_attention.pushButton_yes.clicked.connect(
                        lambda: cancel_attention.pushButton_yes_click(attention_id[i][0]))
                    cancel_attention.pushButton_no.clicked.connect(
                        cancel_attention.pushButton_no_click)
                    print('已取消关注图书' + item.text(0) + '!')
                    break

    # 更新关注图书名称的显示
    def up_show_attention_name(self):
        attention_number = mysql.query_attention(cur, 'count(*)',
                                                 sales_volume_rankings_table_name, "attention = '1'")[0][0]
        main.treeWidget.topLevelItem(3).child(0).setText(0, "无")
        main.treeWidget.topLevelItem(3).child(1).setText(0, "无")
        main.treeWidget.topLevelItem(3).child(2).setText(0, "无")
        # 关注图书的数据库中如果存在数据，就获取关注的图书名称并显示出来
        if attention_number != 0:
            for i in range(attention_number):
                name = mysql.query_attention(cur, 'book_name',
                                             sales_volume_rankings_table_name, "attention = '1'")[i][0]
                main.treeWidget.topLevelItem(3).child(i).setText(0, name)

    # 更新预警信息按钮的单击事件处理方法
    def up(self):
        warningDialog = QtWidgets.QMessageBox.warning(self,
                                                      '警告', '关注商品的预警信息更新后，将以新的信息进行对比并预警！',
                                                      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if warningDialog == QtWidgets.QMessageBox.Yes:
            # 查询已经关注的图书信息在数据库中的id
            attention = mysql.query_attention(cur, 'id,jd_id', sales_volume_rankings_table_name, "attention = '1'")

            for i in attention:
                print(i)
                # 获取好评率与中评最新的时间
                good_rate, middle_time = mycrawl.get_evaluation(2, i[1])
                # 获取差评最新的时间
                good_rate, poor_time = mycrawl.get_evaluation(1, i[1])
                price = self.get_attention_price(i[1])  # 获取关注商品价格
                up = "middle_time='{mi_time}',poor_time='{p_time}',attention_price='{price}'".format(
                    mi_time=middle_time,
                    p_time=poor_time, price=price)
                # 更新关注商品的预警信息
                mysql.update_attention(cur, sales_volume_rankings_table_name, up, " id = {id}".format(id=i[0]))
            print('已更新预警信息！')

    # 获取关注商品价格
    def get_attention_price(self, id):
        price_url = 'http://p.3.cn/prices/mgets?type=1&skuIds={id_str}'  # 获取价格的网络请求地址
        response = requests.get(price_url.format(id_str=id))  # 将京东id作为参数发送获取前100名图书价格
        price = response.json()  # 获取价格json数据，该数据为list类型
        for p in price:
            return p['op']


# 销量榜窗体初始化类
class Sales(QMainWindow, Sales_MainWindow):
    def __init__(self):
        super(Sales, self).__init__()
        self.setupUi(self)

    # 销量榜窗体双击事件处理方法
    def sales_itemDoubleClicked(self):
        item = self.tableWidget.currentItem()  # 表格item对象
        # 判断是否是书名列
        if item.column() == 1:
            # 获取书名，并将书名显示在关注窗体的编辑框内
            attention.lineEdit.setText(item.text())
            # 开启自动填充背景
            attention.centralwidget.setAutoFillBackground(True)
            palette = QPalette()  # 调色板类
            palette.setBrush(QPalette.Background, QtGui.QBrush(
                QtGui.QPixmap('img/attention_bg.png')))  # 设置背景图片
            attention.centralwidget.setPalette(palette)  # 为控件设置对应的调色板即可
            # 设置背景透明
            attention.pushButton_yes.setStyleSheet("background-color:rgba(0,0,0,0)")
            # 设置确认关注按钮的背景图片
            attention.pushButton_yes.setIcon(QtGui.QIcon('img/yes_btn.png'))
            # 设置按钮背景图大小
            attention.pushButton_yes.setIconSize(QtCore.QSize(100, 50))
            # 设置背景透明
            attention.pushButton_no.setStyleSheet("background-color:rgba(0,0,0,0)")
            # 设置确认关注按钮的背景图片
            attention.pushButton_no.setIcon(QtGui.QIcon('img/no_btn.png'))
            # 设置按钮背景图大小
            attention.pushButton_no.setIconSize(QtCore.QSize(100, 50))
            attention.open()  # 显示关注窗体

    # 打开销量榜窗体
    def open(self):
        self.show()


# 热评榜窗体初始化类
class Heat(QMainWindow, Heat_MainWindow):
    def __init__(self):
        super(Heat, self).__init__()
        self.setupUi(self)

    # 打开热评榜窗体
    def open(self):
        self.show()


# 关注窗体初始化类
class Attention(QMainWindow, Attention_MainWindow):
    def __init__(self):
        super(Attention, self).__init__()
        self.setupUi(self)

    # 打开关注窗体
    def open(self):
        self.show()

    # 按钮“是”的单击事件处理
    def pushButton_yes_click(self):
        self.insert_attention_message()

    # 按钮“否”的单击事件
    def pushButton_no_click(self):
        self.close()  # 关闭关注窗体

    # 获取关注图书的预警信息,并进行关注
    def insert_attention_message(self):
        # 关注表格中图书所对应的id,因为表格数据与数据库内容相同，表格中的第0行是数据库中id为1的数据
        rows = sales.tableWidget.currentItem().row() + 1
        # 准备关注的图书名称
        name = sales.tableWidget.currentItem().text()
        # 查询已经关注的图书数量
        attention_number = \
            mysql.query_attention(cur, 'count(*)', sales_volume_rankings_table_name, "attention = '1'")[0][0]
        # 根据id查询图书是否已关注
        is_attention = \
            mysql.query_attention(cur, 'attention', sales_volume_rankings_table_name, "id={id}".format(id=rows))[0][0]
        # 查询已关注图书的jd_id
        jd_id = mysql.query_attention(cur, 'jd_id', sales_volume_rankings_table_name, "id={id}".format(id=rows))[0][0]
        # 获取好评率与中评最新的时间
        good_rate, middle_time = mycrawl.get_evaluation(2, jd_id)
        # 获取差评最新的时间
        good_rate, poor_time = mycrawl.get_evaluation(1, jd_id)
        # 获取关注商品的现在价格
        price = mysql.query_attention(cur, 'jd_price', sales_volume_rankings_table_name, "id={id}".format(id=rows))[0][
            0]
        # 判断是否有已经关注的图书
        if is_attention != '1':
            if attention_number <= 2:
                up = "middle_time='{mi_time}',poor_time='{p_time}',attention_price='{an_price}',attention='1'".format(
                    mi_time=middle_time,
                    p_time=poor_time, an_price=price)
                # 更新数据库中的关注信息
                mysql.update_attention(cur, sales_volume_rankings_table_name, up, " id = {id}".format(id=rows))
                main.treeWidget.topLevelItem(3).child(attention_number).setText(0, name)
                print('已关注图书《' + name + '》!')
                self.close()
            else:
                messageDialog('警告！', '仅可以关注3本图书！')
                print('仅可以关注3本图书！')
                self.close()
        else:
            messageDialog('警告！', '不可以关注相同的图书！')
            self.close()
            print('不可以关注相同的图书！')


# 取消关注窗体初始化类
class Cancel_Attention(QMainWindow, Attention_MainWindow):
    def __init__(self):
        super(Cancel_Attention, self).__init__()
        self.setupUi(self)

        # 打开窗体

    def open(self):
        self.show()

    # 按钮是的单击事件处理
    def pushButton_yes_click(self, id):
        # 在数据库中取消关注标记
        mysql.update_attention(cur, sales_volume_rankings_table_name, "attention='0'", " id = {id}".format(id=id))
        main.up_show_attention_name()
        self.close()

    # 按钮否的单击事件处理
    def pushButton_no_click(self):
        self.close()


# 评价预警窗体初始化类
class Evaluate_Warning(QMainWindow, Evaluate_Warning_MainWindow):
    def __init__(self):
        super(Evaluate_Warning, self).__init__()
        self.setupUi(self)

    # 打开窗体
    def open(self):
        self.show()

    def warning(self):
        warning_list = []  # 保存评价分析后得数据
        # 查询关注图书的信息，其中包含图书名称，中评时间与差评时间
        attention_message = mysql.query_attention(cur,
                                                  'book_name,middle_time,poor_time,jd_id',
                                                  sales_volume_rankings_table_name, "attention = '1'")
        # 判断是否有关注图书的信息
        if len(attention_message) != 0:
            middle_time = ''
            poor_time = ''
            for i in range(len(attention_message)):
                # 获取好评率与中评最新的时间
                good_rate, new_middle_time = mycrawl.get_evaluation(2,
                                                                    attention_message[i][3])
                # # 获取差评最新的时间
                good_rate, new_poor_time = mycrawl.get_evaluation(1,
                                                                  attention_message[i][3])
                if attention_message[i][1] == new_middle_time:
                    middle_time = '无'
                else:
                    middle_time = '有'
                if attention_message[i][2] == new_poor_time:
                    poor_time = '无'
                else:
                    poor_time = '有'
                warning_list.append((attention_message[i][0], middle_time, poor_time))
            for i in range(len(attention_message)):
                for j in range(3):
                    temp_data = warning_list[i][j]  # 临时记录，不能直接插入表格
                    data = QtWidgets.QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                    data.setTextAlignment(QtCore.Qt.AlignCenter)
                    evaluate.tableWidget.setItem(i, j, data)


# 价格预警窗体初始化类
class Price_Warning(QMainWindow, Price_Warning_MainWindow):
    def __init__(self):
        super(Price_Warning, self).__init__()
        self.setupUi(self)

    # 打开窗体
    def open(self):
        self.show()

    # 价格信息处理
    def price(self):
        price_list = []  # 保存价格分析后的数据
        # 查询关注图书的信息，其中包含图书的京东价格以及京东id
        attention_message = mysql.query_attention(cur, 'attention_price,jd_id,book_name',
                                                  sales_volume_rankings_table_name, "attention = '1'")
        # # # 判断是否有关注图书的信息
        if len(attention_message) != 0:
            jd_id_str = ''
            for i in range(len(attention_message)):
                jd_id = 'J_' + attention_message[i][1] + ','
                jd_id_str = jd_id_str + jd_id
            price_url = 'http://p.3.cn/prices/mgets?type=1&skuIds={id_str}'
            response = requests.get(price_url.format(id_str=jd_id_str))  # 将京东id作为参数发送获取前100名图书价格
            price_json = response.json()  # 获取价格json数据，该数据为list类型
            change = ''
            for index, item in enumerate(price_json):
                # 京东价格
                new_jd_price = item['op']
                if float(attention_message[index][0]) < float(new_jd_price):
                    change = '上涨'
                if float(attention_message[index][0]) == float(new_jd_price):
                    change = '无'
                if float(attention_message[index][0]) > float(new_jd_price):
                    change = '下浮'
                price_list.append((attention_message[index][2], change))
            for i in range(len(attention_message)):
                for j in range(2):
                    temp_data = price_list[i][j]  # 临时记录，不能直接插入表格
                    data = QtWidgets.QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                    data.setTextAlignment(QtCore.Qt.AlignCenter)
                    price.tableWidget.setItem(i, j, data)


# 评价分析图窗体初始化类
class Evaluation_Chart(QMainWindow, Evaluation_Chart_MainWindow):
    def __init__(self):
        super(Evaluation_Chart, self).__init__()
        self.setupUi(self)

    # 打开窗体
    def open(self):
        self.show()


# 出版社占有比例窗体初始化类
class Press_Bar(QMainWindow, Press_Bar_MainWindow):
    def __init__(self):
        super(Press_Bar, self).__init__()
        self.setupUi(self)

    # 打开窗体
    def open(self):
        self.show()

# 关于窗体初始化类
class About_Window(QMainWindow, About_MainWindow):
    def __init__(self):
        super(About_Window, self).__init__()
        self.setupUi(self)

    # 打开窗体
    def open(self):
        self.show()


if __name__ == "__main__":
    # 创建自定义数据库对象
    mysql = MySQL()
    # 创建爬去对象
    mycrawl = Crawl()
    # 连接数据库
    sql = mysql.connection_sql()
    # 创建游标
    cur = sql.cursor()

    app = QApplication(sys.argv)
    # 主窗体对象
    main = Main()
    # 显示主窗体
    main.show()
    # 销量排行窗体对象
    sales = Sales()
    # 热评排行窗体对象
    heat = Heat()

    # 关注窗体对象
    attention = Attention()
    # 取消关注窗体对象
    cancel_attention = Cancel_Attention()
    # 评价预警窗体对象
    evaluate = Evaluate_Warning()
    # 价格预警窗体对象
    price = Price_Warning()
    # 关注图书评价分析图窗体对象
    evaluation = Evaluation_Chart()

    # 出版社占有比例窗体对象
    press_bar = Press_Bar()

    # 关于窗体对象
    about = About_Window()
    # 指定菜单栏关于选项单击事件处理方法
    main.action_about.triggered.connect(about.open)

    # 指定左侧树形菜单的事件处理方法
    main.treeWidget.itemClicked['QTreeWidgetItem*', 'int'].connect(main.tree_itemClicked)
    # 指定更新商品排行信息按钮的事件处理方法
    main.pushButton.clicked.connect(main.up)

    # 指定销量榜表格的双击事件处理方法
    sales.tableWidget.itemDoubleClicked.connect(sales.sales_itemDoubleClicked)
    # 指定关注窗体按钮(是)单击事件处理方法
    attention.pushButton_yes.clicked.connect(attention.pushButton_yes_click)
    # 指定关注窗体按钮（否）单击事件处理方法
    attention.pushButton_no.clicked.connect(attention.pushButton_no_click)


    sys.exit(app.exec_())
