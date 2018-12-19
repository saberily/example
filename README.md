#anaconda添加清华的镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes

#anaconda创建新环境
conda create -n tensorflow-gpu python=3.6

#tensorflow安装
pip install --ignore-installed --upgrade tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu
