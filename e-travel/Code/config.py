# -*- coding=utf-8 -*-
import os
class Config:
    SECRET_KEY = 'mrsoft'
    UP_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "app/static/uploads/")  # 文件上传路径
    FC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "app/static/uploads/users/")  # 用户头像上传路径

    @staticmethod
    def init_app(app):
        pass

# the config for development
class DevelopmentConfig(Config):
    #SQLALCHEMY_DATABASE_URI flask_sqlalchemy 数据库配置
    #其他flask_sqlalchemy不常用配置
    #SQLALCHEMY_ECHO SQLALCHEMY_RECORD_QUERIES SQLALCHEMY_NATIVE_UNICODE
    #数据库+数据库驱动类型://用户名:密码@hostip:port/数据库名
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@192.168.0.106:3306/travel'
    SQLALCHEMY_TRACK_MODIFICATIONS = True

    #flask app 运行debug模式
    DEBUG = True

# define the config
config = {
    'default': DevelopmentConfig
}
