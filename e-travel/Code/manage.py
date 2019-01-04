# _*_ codding:utf-8 _*_
from app import create_app, db
from app.models import *
from flask_script import Manager, Shell
from flask_migrate import Migrate, MigrateCommand
from flask import render_template

app = create_app('default')
manager = Manager(app)
migrate = Migrate(app, db)

def make_shell_context():
    return dict(app=app, db=db)

#shell的作用执行python manage.py shell进入python shell命令行同时导入指定的这里是app和db对象可直接调用
#shell有点类似预处理的作用
#添加shell命令并自定义shell上下文，即预导入的对象
manager.add_command("shell", Shell(make_context=make_shell_context))
#添加db命令
#python manage.py db init      #迁移仓库创建
#python manage.py db migrate   #根据迁移仓库创建迁移脚本
#python manage.py db upgrade   #使用迁移脚本更新数据库
manager.add_command('db', MigrateCommand)

@app.errorhandler(404)
def page_not_found(error):
    """
    404
    """
    return render_template("home/404.html"), 404

if __name__ == '__main__':
    #python manage.py runserver
    manager.run()