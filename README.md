# 一、整体说明

## 待解决的问题
声呐图像的检测研究在工业、环境及军事等诸多领域均有着广泛的应用价值。声呐图像数据经常伴随水深探测及底层探测数据一并获取，从而能够使我们观测海底浅层结构。声呐图像目标检测，即从给定的水下侧扫声呐或前视声呐图像中检测由特殊地形地貌、人造物等构成的特征目标，标注目标区域的位置和范围。

## 整体思路 
借鉴一般图像目标检测的方法，通过迁移学习，对声呐图像进行目标检测。

## 项目重难点
声呐图像目标检测任务面临着数据集稀缺和噪声扰动两大挑战，容易导致模型过拟合。

## 主要创新点
1. 通过迁移学习，利用coco数据集的预训练模型，采用Cascade-RCNN对声呐图像进行检测。
2. 通过对声呐图像噪声的数学分析，引入斑点噪声数据增强，增加模型的鲁棒性。

# 二、数据、模型

## 数据处理方案
* 图像归一化
* 随机水平翻转
* 随机增加斑点噪声   
## 模型选择
* 检测算法：Cascade-RCNN+FPN
* backbone: ResNet-50+DCN, ResNet-101+DCN
* RCNN的IoU阈值修改为[0.55, 0.65, 0.75]
## 训练方案
* 2卡， 学习率0.005， 12个epoch， warm up 500 iters, 使用coco预训练。
* 采用多尺度训练
## 训练结果
* 采用多尺度测试，并做水平翻转的测试增强；
* 将epoch10、11与12进行权重平均处理；
* 最后使用bbox vote进行融合；
* 单模后处理：soft-nms， 模型融合后处理：nms
## 相关论文/项目链接
1. 基于[mmdetection](https://github.com/open-mmlab/mmdetection)
2. [Training with Noise Adversarial Network: A Generalization Method for Object Detection on Sonar Image](http://openaccess.thecvf.com/content_WACV_2020/papers/Ma_Training_with_Noise_Adversarial_Network_A_Generalization_Method_for_Object_WACV_2020_paper)



# 三、项目运行环境

## 项目所需的工具包/框架
* mmdetection 1.1.0
* pytorch 1.1.0
* torchvision 0.3.0
* mmcv 0.3.2
* icecream 2.0.0
* albumentations 0.4.5
* imagecorruptions 1.1.0

## 安装环境
* 创建并激活虚拟环境 conda create -n underwater python=3.7 -y conda activate underwater
* 安装 pytorch conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
* 安装其他依赖 pip install cython && pip --no-cache-dir install -r requirements.txt
* 编译cuda op等： python setup.py develop

## 项目运行的资源环境
模型由两位队员分别训练，具体资源如下：
* 1：2卡 Titan RTX
* 2：2卡 11G-RTX 2080ti

# 四、项目运行办法

## 项目的文件结构
```shell
├── README.md
├── test.sh
├── train.sh
├── download_pretrained_model.sh
└── mmdetection
    ├── checkpoints
    ├── mmdet
    └── undersonar
        ├── coco
        │   └── annotations
        ├── configs
        ├── data
        │   ├── image
        │   │   ├── all
        │   │   ├── ce
        │   │   └── qian
        │   └── train
        │       ├── ce
        │       ├── fu
        │       └── qian
        ├── json
        ├── src
        └── work_dirs
```

## 数据准备
```shell
1. 原始数据压缩包放在`./mmdetection/undersonar/data/`中, 运行`prepare_data.sh`即可解压并重命名相关文件夹。
2. 预训练模型放在`./mmdetection/checkpoints/`中，运行`download_pretrained_model.sh`即可下载至指定位置。
3. 我们队伍训练的模型放在`./mmdetection/undersonar/work_dirs/`，由于文件较大，存放到百度云中，可下载至指定位置，具体位置如下所示。
		└── work_dirs
				├── r101_epoch_ensemble.pth
				├── r50_db_epoch_ensemble.pth
				├── r50_gc_epoch_ensemble.pth
				└── r50_speckle_epoch_ensemble.pth
```


## 项目的运行步骤
### 训练：
```shell
1. 运行`prepare_data.sh`
2. 运行`train.sh`
3. 在`./mmdetection/undersonar/work_dirs/`文件夹中含有生成的结果
```
### 测试：

我们对最后训练生成的epoch10、11与12进行权重平均处理，即运行`mmdetection/undersonar/src/swa_db.py`，这些文件太大，我们直接生成了最后的结果，如数据准备部分第三点所示。

```shell
1. 若没有执行前面训练的部分，需运行`prepare_data.sh`准备数据
2. 运行`test.sh`
3. 在`./mmdetection/undersonar/json/`文件夹中含有生成测试结果的中间文件
4. 运行`sub.sh`生成最后csv文件。
```

**注：我们采用双卡测试 **

## 运行结果的位置

1. `./mmdetection/undersonar/submit.csv`
