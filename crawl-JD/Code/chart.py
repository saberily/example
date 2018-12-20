
# 图形画布
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib  # 导入图表模块
import matplotlib.pyplot as plt # 导入绘图模块


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=0, height=0, dpi=100):
        # 避免中文乱码
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 创建图形
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        # 初始化图形画布
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)  # 设置父类

    # 显示评价饼图
    def pie_chart(self, good_size, general_poor_size, title):
        """
        绘制饼图
        explode：设置各部分突出
        label:设置各部分标签
        labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径
        autopct：设置圆里面文本
        shadow：设置是否有阴影
        startangle：起始角度，默认从0开始逆时针转
        pctdistance：设置圆内文本距圆心距离
        返回值
        l_text：圆内部文本，matplotlib.text.Text object
        p_text：圆外部文本
        """
        label_list = ['好评', '中差评']  # 各部分标签
        size = [good_size, general_poor_size]  # 各部分大小
        color = ['lightblue', 'red']  # 各部分颜色
        explode = [0.05, 0]  # 各部分突出值
        plt.pie(size, colors=color, labels=label_list, explode=explode, labeldistance=1.1,
                autopct="%1.1f%%", shadow=True, startangle=0, pctdistance=0.6)
        plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
        plt.title(title, fontsize=12)
        plt.legend()  # 显示图例

    # 显示前十名价格趋势的折线图
    def broken_line(self, y):
        '''
        y:y轴折线点，也就是价格
        linewidth:折线的宽度
        color：折线的颜色
        marker：折点的形状
        markerfacecolor：折点实心颜色
        markersize：折点大小
        '''
        x = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # X轴折线点，也就是排名
        plt.plot(x, y, linewidth=3, color='r', marker='o',
                 markerfacecolor='blue', markersize=8)  # 绘制折线，并在折点添加蓝色圆点
        plt.xlabel('排名')
        plt.ylabel('价格')
        plt.title('前10名价格走势图')  # 标题名称
        plt.grid()  # 显示网格

    # 显示出版社占有比例的条形图
    def bar(self, number, press, title):
        """
        绘制水平条形图方法barh
        参数一：y轴
        参数二：x轴
        """
        # 设置图表跨行跨列
        plt.subplot2grid((12, 12), (1, 2), colspan=12, rowspan=10)
        plt.barh(range(len(number)), number, height=0.3, color='r', alpha=0.8)  # 从下往上画水平条形图
        plt.yticks(range(len(number)), press) # Y轴出版社名称显示
        plt.xlim(0, 100)                      # X轴的数量0~100
        plt.xlabel("比例")                    # 比例文字
        plt.title(title)                      # 表标题文字
        # 显示百分比数量
        for x, y in enumerate(number):
            plt.text(y + 0.1, x, '%s' % y + '%', va='center')
