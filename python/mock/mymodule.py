import os
import os.path


def rm(filename):
    if os.path.isfile(filename):
        os.remove(filename)


class RemovalService(object):

    def rm(self, filename):
        if os.path.isfile(filename):
            #获取当前工作路径
            #print(os.getcwd())
            os.remove(filename)
            

class UploadService(object):

    def __init__(self, removal_service):
        self.removal_service = removal_service

    def upload_complete(self, filename):
        self.removal_service.rm(filename)