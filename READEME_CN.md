[[日语(README.md)/英语]]

> **说明**
> <br>我创建了一个关键点分类的模型库仓库。
> <br>→ [Kazuhito00/手部关键点分类模型库](https://github.com/Kazuhito00/hand-keypoint-classification-model-zoo)

# 基于MediaPipe的手势识别
使用MediaPipe（Python版本）估算手部姿态。<br>
这是一个通过检测关键点，利用简易多层感知机（MLP）识别手语与手指动作的示例程序。
![动图演示](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

本仓库包含以下内容：
* 示例程序
* 手语识别模型（TFLite）
* 手指动作识别模型（TFLite）
* 手语识别训练数据及配套训练笔记
* 手指动作识别训练数据及配套训练笔记

# 环境依赖
* mediapipe 0.8.1
* OpenCV 3.4.2 及以上版本
* Tensorflow 2.3.0 及以上版本<br>tf-nightly 2.5.0开发版及以上（仅制作LSTM模型的TFLite文件时需要）
* scikit-learn 0.23.2 及以上（仅需展示混淆矩阵时使用）
* matplotlib 3.3.2 及以上（仅需展示混淆矩阵时使用）

# 运行演示
通过摄像头运行演示程序：
```bash
python app.py
```

通过Docker搭配摄像头运行演示：
```bash
docker build -t hand_gesture .

xhost +local: && \
docker run --rm -it \
--device /dev/video0:/dev/video0 \
-v `pwd`:/home/user/workdir \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
-e DISPLAY=$DISPLAY \
hand_gesture:latest

python app.py
```

运行演示时可配置以下参数：
* --device<br>指定摄像头设备编号（默认：0）
* --width<br>摄像头采集画面宽度（默认：960）
* --height<br>摄像头采集画面高度（默认：540）
* --use_static_image_mode<br>是否为MediaPipe推理启用静态图像模式（默认：不启用）
* --min_detection_confidence<br>
检测置信度阈值（默认：0.5）
* --min_tracking_confidence<br>
跟踪置信度阈值（默认：0.5）

# 目录结构
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│
└─utils
    └─cvfpscalc.py
</pre>

### app.py
推理主示例程序。<br>
同时可采集手语识别的关键点训练数据，以及手指动作识别的指尖坐标轨迹训练数据。

### keypoint_classification.ipynb
手语识别模型的训练脚本。

### point_history_classification.ipynb
手指动作识别模型的训练脚本。

### model/keypoint_classifier
存放手语识别相关文件：
* 训练数据（keypoint.csv）
* 训练完成的模型（keypoint_classifier.tflite）
* 标签文件（keypoint_classifier_label.csv）
* 推理调用模块（keypoint_classifier.py）

### model/point_history_classifier
存放手指动作识别相关文件：
* 训练数据（point_history.csv）
* 训练完成的模型（point_history_classifier.tflite）
* 标签文件（point_history_classifier_label.csv）
* 推理调用模块（point_history_classifier.py）

### utils/cvfpscalc.py
帧率（FPS）计算工具模块。

# 模型训练
可自行新增、修改训练数据，重新训练手语识别与手指动作识别模型。

## 手语识别训练
### 1. 采集训练数据
按下「k」键，进入关键点数据录制模式（界面提示：模式：录制关键点）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>

按下数字键「0~9」，即可将当前关键点数据存入 `model/keypoint_classifier/keypoint.csv`：<br>
首列：按下的数字（作为类别ID），后续列：关键点坐标<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>

关键点坐标已完成①至④步预处理：<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>

初始内置三类数据：张开手掌（类别ID：0）、握紧手掌（类别ID：1）、指尖指向（类别ID：2）。<br>
可按需新增3及以上类别，或删除CSV原有数据自定义数据集。<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

### 2. 执行模型训练
用Jupyter Notebook打开「keypoint_classification.ipynb」，自上而下逐格运行。<br>
如需修改类别总数，修改代码中 `NUM_CLASSES = 3`，并同步修改标签CSV文件。<br><br>

### 补充：模型结构
该训练脚本内置模型结构示意图：
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

## 手指动作识别训练
### 1. 采集训练数据
按下「h」键，进入指尖坐标轨迹录制模式（界面提示：模式：录制坐标轨迹）<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>

按下数字键「0~9」，即可将轨迹数据存入 `model/point_history_classifier/point_history.csv`：<br>
首列：按下的数字（类别ID），后续列：坐标轨迹序列<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>

坐标同样经过①至④步预处理：<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>

初始内置四类数据：静止不动（ID：0）、顺时针滑动（ID：1）、逆时针滑动（ID：2）、随意移动（ID：4）。<br>
可按需新增5及以上类别，或清空CSV自定义数据集。<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

### 2. 执行模型训练
用Jupyter Notebook打开「point_history_classification.ipynb」，自上而下逐格运行。<br>
修改类别数需调整 `NUM_CLASSES = 4`，并同步更新标签文件。<br><br>

### 补充：模型结构
常规模型示意图：
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>

LSTM时序模型示意图：<br>
如需启用LSTM，将代码 `use_lstm = False` 改为 `True`（需安装tf-nightly，适配2020年12月16日版本）<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# 应用案例
* [手势操控大疆特洛无人机](https://towardsdatascience.com/control-dji-tello-drone-with-hand-gestures-b76bd1d46447)
* [基于OAK-D设备识别美式手语字母](https://www.cortic.ca/post/classifying-american-sign-language-alphabets-on-the-oak-d)

# 参考资料
* [MediaPipe 官方网站](https://mediapipe.dev/)
* [Kazuhito00/MediaPipe Python示例仓库](https://github.com/Kazuhito00/mediapipe-python-sample)

# 作者
高桥一仁（Kazuhito Takahashi）(https://twitter.com/KzhtTkhs)

# 开源许可
本项目遵循 [Apache 2.0 开源协议](LICENSE)。