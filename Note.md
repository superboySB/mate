# 自博弈实现方式复现 
```sh
# （Optional）required on Windows 建议测试 
git config --global core.symlinks true 
# 拉取代码 
git clone https://github.com/superboySB/mate.git 
# 暂时还是当做一台机器训练与推理来处理 (但目前感觉python3.7很难装上nashpy)
conda create -n mate python=3.7 && conda activate mate
cd mate && pip install -r requirements.txt && pip install -e . 
```