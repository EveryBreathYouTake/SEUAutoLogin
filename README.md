# 基于注意力机制的调制模型



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#介绍">介绍</a>
    </li>
    <li>
      <a href="#环境准备">环境准备</a>
    </li>
    <li><a href="#数据准备">数据准备</a>
    <ul>
        <li><a href="#coco数据集">COCO数据集</a></li>
        <li><a href="#布匹瑕疵数据集">布匹瑕疵数据集</a></li>
    </ul>    
    </li>
    <li><a href="#预训练模型加载">预训练模型加载</a>
    <ul>
        <li><a href="#骨干网络模型加载">骨干网络模型加载</a></li>
        <li><a href="#本项目模型加载">本项目模型加载</a></li>
    </ul> 
    </li>
    <li><a href="#cuda环境依赖">Cuda环境依赖</a></li>    
    <li><a href="#训练和测试">训练和测试</a></li>
    <li><a href="#感谢">感谢</a></li>
  </ol>
</details>

<!-- INTRODUCTION -->
## 介绍
本项目是基于COCO数据集和布匹瑕疵数据集的小样本目标检测项目。代码基于Faster RCNN模型以及DAnA模型并进行改进。


<!-- GETTING STARTED -->
## 环境准备
* Python 3.6及以上
* Cuda 10.0 or 10.1
* Pytorch 1.2.0 及以上

pip安装以下内容：

    - cycler==0.10.0
    - cython==0.29.16
    - easydict==1.9
    - future==0.18.2
    - kiwisolver==1.2.0
    - matplotlib==3.2.1
    - msgpack==1.0.0
    - opencv-python==4.2.0.34
    - packaging==20.3
    - pandas==1.0.3
    - protobuf==3.11.3
    - pyparsing==2.4.7
    - python-box==4.2.3
    - python-dateutil==2.8.1
    - pytz==2020.1
    - pyyaml==5.3.1
    - ruamel-yaml==0.16.10
    - ruamel-yaml-clib==0.2.0
    - scipy==1.1.0
    - tensorboardx==2.0
    - toml==0.10.1
    - torchsummary==1.5.1
    - tqdm==4.48.0
    
    
## 数据准备
因数据集过于庞大，无法随代码一同上交。故运行时需重新下载数据集以进行训练和测试。

### COCO数据集
COCO数据一般来说，需在官网 (https://cocodataset.org/#download) 进行下载。尔后需要通过预处理，如分割基类和新类，隐藏部分图片注释标签等。这部分内容过于繁琐，因此提供一个预处理完的json文件。

* 60 类基类数据 (https://drive.google.com/file/d/10mXvdpgSjFYML_9J-zMDLPuBYrSrG2ub/view?usp=sharing)
* 20 类新类数据 (https://drive.google.com/file/d/1FZJhC-Ob-IXTKf5heNeNAN00V8OUJXi2/view?usp=sharing)

同样，需对每一类别抽取一部分数据作为支持集。这部分可以随机采样。这里同样给出已随机采样后的支持集：

支持集数据 (https://drive.google.com/file/d/1nl9-DEpBBJ5w6hxVdijY6hFxoQdz8aso/view)

### 布匹瑕疵数据集
布匹瑕疵数据集同样需在阿里天池大赛官网进行下载，这里给出一个下载路径。尔后需划分基类和新类。并将注释标签数据转化为COCO格式的json文件。

布匹瑕疵数据集 (https://pan.baidu.com/s/1XXjWey0x48Pnrxl9RyX0lQ)   密码：q9bq

## 预训练模型加载

### 骨干网络模型加载
可自主更换骨干网络，例如Res50, Vgg16等。本项目所使用的默认骨干网络为Res101。需自主下载。其中一个下载地址为

Res101模型 (https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

下载完成后将其放置在data/pretrained_model文件夹下。

### 本项目模型加载

本文模型加载：可在训练后进行替换，用来测试。这里的模型可以在自主训练后在models/文件夹下找到。

## Cuda环境依赖

运行前需运行cuda环境依赖，步骤：

$ cd lib

$ python setup.py build develop

## 训练和测试

从头训练：

$ python train.py --dataset coco_base --flip --net DAnA --lr 0.001 --lr_decay_step 12 --bs 1 --epochs 16 --disp_interval 20 --save_dir models/DAnA --way 2 --shot 10 --sup_dir /mnt/data2/Data/supports/all

（其中尝试不同的dataset 及其他参数意义和修改规则可以在utils.py中查看规则。支持集位置可自定义输入。而依据自定义的训练数据集的位置，可以在lib/datasets/coco.py 中的self._data_path进行修改。）

继续训练：

$ python train.py --dataset coco_base --flip --net DAnA --lr 0.001 --lr_decay_step 12 --bs 1 --epochs 16 --disp_interval 20 --save_dir models/DAnA --way 2 --shot 10 --r --load_dir models/AttentionFRCNN --checkepoch 5 --checkpoint 137873 --sup_dir /mnt/data2/Data/supports/all

（其中load_dir后可接你保存的模型位置和名称）

评估：

$ python inference.py --eval --dataset val2014_novel --net DAnA --r --load_dir models/DAnA_10 --checkepoch 5 --checkpoint 137873 --bs 1 --shot 10 --eval_dir dana_10 --sup_dir /mnt/data2/Data/supports/all

（其中dataset规则同上）

demo演示：

$ python demo.py --eval --dataset val2014_novel --net DAnA --r --load_dir models/DAnA_10 --checkepoch 5 --checkpoint 137873 --bs 1 --shot 10 --eval_dir dana_10 --sup_dir /mnt/data2/Data/supports/all --uimag /home/guest2/jtzhu/faster-rcnn.pytorch-pytorch-1.0/images/car.png

（其中--uimag后接想要检测的图片，图片最好是jpg和png格式的，且大小接近600*1000。检测结果图片会在同一目录下显示一张新的检测图片。



## 感谢
本项目在Nvidia 3090显卡上完成。具体参数设置可在论文中得知。感谢我的导师周毅的指导和帮助。
