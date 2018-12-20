import pymysql  # 导入操作MySQL数据库模块

# 数据库字段
'''
id          编号
book_name   图书名称
jd_price    京东价格
ding_price  定价
press       出版社
item_url    图书对应的地址
jd_id       京东商品id
middle_time 中评最新的时间
poor_time   差评最新的时间
attention_price  关注价格
attention   关注
'''


class MySQL(object):
    # 连接数据库
    def connection_sql(self):
        # 连接数据库
        self.db = pymysql.connect(host="192.168.0.104", user="root",
                                  password="123456", db="jd_data", port=3306,charset="utf8")
        return self.db

    # 关闭数据库
    def close_sql(self):
        self.db.close()

    # 排行数据插入方法,该方法可以根据更换表名插入排行数据
    def insert(self, cur, value, table):
        # 插入数据的sql语句
        sql_insert = "insert into  {table} (id,book_name,jd_price,ding_price," \
                     "press,item_url,jd_id) values(%s,%s,%s,%s,%s,%s,%s)on duplicate" \
                     " key update book_name=values(book_name),jd_price=values(jd_price)," \
                     "ding_price=values(ding_price),press=values(press),item_url=" \
                     "values(item_url),jd_id=values(jd_id)".format(table=table)
        try:
            # 执行sql语句
            cur.executemany(sql_insert, value)
            # 提交
            self.db.commit()
        except Exception as e:
            # 错误回滚
            self.db.rollback()
            # 输出错误信息
            print(e)

    # 获取销量榜前10名价格
    def query_top10_jd_price(self, cur):
        y = []  # 保存前10名京东价格的列表
        query_sql = "select jd_price from sales_volume_rankings where id<=10"
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        for row in results:
            y.append(row[0])  # 将京东价格添加至列表中
        return y  # 将前10名的京东价格列表返回

    # 获取销量榜前10名书名
    def query_top10_book_name(self, cur):
        name = []  # 书名列表
        query_sql = "select book_name from sales_volume_rankings where id<=10"
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        i = 1  # 定义排名变量
        for row in results:
            i = str(i)  # 转换变量类型为字符串类型
            name.append('第' + i + '名：  ' + row[0])  # 将排名与图书名称添加至列表中
            i = int(i)  # 由于字符串类型无法进行计算，所以转换为int类型
            i += 1      # 改变排名
        return name    # 将保存书名的列表返回

    # 获取出版社比例
    def query_press_proportion(self, cur, query_sql):
        press_list = []  # 出版社列表
        number_list = []  # 数量
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        for row in results:
            press_list.append(row[0].strip('出版社'))  # 去除出版社三个字，然后添加至对应的列表中
            number_list.append(row[1])  # 将出版社占有数量添加至对应的列表中
        return number_list, press_list # 将出版社列表与数量列表返回

    # 获取排行第一的京东id
    def query_top1_id(self, cur):
        query_sql = "select jd_id from sales_volume_rankings where id=1"
        cur.execute(query_sql)  # 执行sql语句
        jd_id = cur.fetchone()  # 获取查询的内容
        return jd_id[0]        # 返回京东id

    # 获取排行100名的图书信息，这里仅需要查询图书的id，图书名称，京东价，定价，以及出版社
    def query_top100_rankings(self, cur, table):
        query_sql = "select id,book_name,jd_price,ding_price,press from {table}".format(table=table)
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        row = len(results)  # 获取信息条数，作为表格的行
        column = len(results[0])  # 获取字段数量，作为表格的列
        return row, column, results  # 返回信息行与信息列（字段对应的信息）

    # 获取数据表中有多少条数据
    def query_is_number(self, cur, table):
        query_sql = "select count(*) from {table}".format(table=table)
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        return results[0][0]     # 返回多少条数据

    # # 更新关注商品信息
    # def update_attention(self, cur, table, column, id):
    #     sql_update = "update {table} set {column} where id = {id}".format(table=table, column=column, id=id)
    #     try:
    #         cur.execute(sql_update)  # 执行sql语句
    #         # 提交
    #         self.db.commit()
    #     except Exception as e:
    #         # 错误回滚
    #         self.db.rollback()
    #         # 输出错误信息
    #         print(e)

    # 更新关注商品信息
    def update_attention(self, cur, table, up, where):
        sql_update = "update {table} set {up} where  {where}".format(table=table, up=up, where=where)
        try:
            cur.execute(sql_update)  # 执行sql语句
            # 提交
            self.db.commit()
        except Exception as e:
            # 错误回滚
            self.db.rollback()
            # 输出错误信息
            print(e)

    # 获取关注图书的信息
    def query_attention(self, cur, column, table, where):
        query_sql = "select {column} from {table} where {where} ".format(column=column, table=table, where=where)
        cur.execute(query_sql)  # 执行sql语句
        results = cur.fetchall()  # 获取查询的所有记录
        return results           # 返回查询信息

    # 清空数据表
    def query_empty(self,cur,table):
        sql_delete = "truncate table {table}".format(table = table)
        try:
            cur.execute(sql_delete)  # 像sql语句传递参数
            # 提交
            self.db.commit()
        except Exception as e:
            # 错误回滚
            self.db.rollback()
            # 输出错误信息
            print(e)




