# 人脸检测与识别 (OpenCV + MediaPipe)

基于 OpenCV 与 MediaPipe 库实现人脸检测、特征点提取及识别全流程。

## 功能

从视频/图片/摄像头中定位人脸区域并绘制矩形或圆形标注；通过 MediaPipe Face Mesh 提取人脸特征点，完成特征点标记与连接线（网格）绘制；构建人脸数据集，基于 LBPH 算法训练识别模型，支持从图片中检测人脸并完成 1:1 身份校验，输出匹配标签、用户名及置信评分。

系统适配本地图片/视频文件及摄像头输入，模型训练结果可保存、可读取。

## 理论与工具

本示例串联了「人脸检测 → 关键点提取 → 人脸识别」三个经典计算机视觉任务，下面分别说明背后的原理与所用工具。

### 整体流程

```
输入(图片/视频/摄像头)
   │
   ├─[检测] Haar 级联 → 定位人脸框 (x, y, w, h)
   │
   ├─[关键点] MediaPipe Face Mesh → 468 个 3D 面部关键点
   │
   └─[识别] 裁剪灰度人脸 → LBPH 比对 → 身份标签 + 置信度
```

注意：**检测 (detection)** 只回答“画面里有没有脸、在哪里”，而 **识别 (recognition)** 进一步回答“这是谁”。本项目检测用 OpenCV Haar，识别用 OpenCV LBPH，关键点用 MediaPipe。

### 1. 人脸检测：Haar 级联 (Viola–Jones)

`detect.py` / `capture.py` / `recognize.py` 中的检测基于 OpenCV 自带的 Haar 级联分类器，即经典的 **Viola–Jones** 算法，核心思想有三点：

- **Haar-like 特征**：用一组黑白矩形模板在图像上滑动，计算“黑区像素和 − 白区像素和”。人脸的眼睛、鼻梁、嘴等区域恰好对应明暗变化，能被这些特征捕捉。
- **积分图 (Integral Image)**：预先计算前缀和，使任意矩形区域的像素和都能 O(1) 求出，让海量特征的计算变得极快。
- **AdaBoost + 级联 (Cascade)**：用 AdaBoost 从数万个特征里挑出少量强分类器，再串成多级级联——绝大多数“非人脸”窗口在前几级就被快速否决，只有疑似人脸才进入后续精细判别，因此能做到实时。

实践中我们在灰度图上检测，并先做 `equalizeHist` 直方图均衡以增强对比、缓解光照影响。Haar 简单、离线、无需额外模型文件，非常适合教学，但对大角度、遮挡、弱光照较敏感。

### 2. 人脸关键点：MediaPipe Face Mesh

`landmarks.py` 使用 Google 的 **MediaPipe Face Mesh**。它是基于深度学习的轻量级模型，可在一张脸上回归出 **468 个 3D 面部关键点**（含 `refine_landmarks` 时还会细化眼/唇/虹膜）。

- 工作流：先做人脸检测得到 ROI，再由关键点回归网络预测每个点的 `(x, y, z)` 归一化坐标。
- 与传统 dlib 的 68 点相比，Face Mesh 点更密、带深度信息、对姿态更鲁棒，且推理速度适合实时。
- 本示例用 `FACEMESH_TESSELATION`（三角网格）与 `FACEMESH_CONTOURS`（轮廓线）把关键点连成可视化的“面具网格”。

> README 标题里提到的“68 个特征点”是传统 dlib/300-W 标注体系的说法；MediaPipe 用的是更密集的 468 点方案，二者目标一致，只是关键点数量与定义不同。

### 3. 人脸识别：LBPH (Local Binary Patterns Histograms)

`train.py` / `recognize.py` 使用 OpenCV `cv2.face.LBPHFaceRecognizer`，原理分两步：

- **LBP（局部二值模式）**：对每个像素，将其与周围 8 邻域比较，邻域≥中心记为 1、否则记为 0，按顺序拼成一个 8 位二进制数（0–255）作为该像素的纹理编码。LBP 描述的是**局部纹理/梯度方向**，对单调的光照变化天然鲁棒。
- **H（直方图）**：把人脸图划分成若干网格，在每个网格内统计 LBP 值的直方图，再把所有网格的直方图拼接成一个长特征向量来代表这张脸。
- **识别**：预测时对待测脸提取同样的特征向量，与训练库中各样本做距离比较（默认卡方/直方图距离），取最近者作为身份；返回的 `confidence` 就是这个**距离**——**越小越像**。

LBPH 训练快、对小样本友好、可增量更新，适合本地、少量用户的 1:1 核验场景；但在大角度、跨光照域、跨年龄等差异较大时，识别率会明显下降，工业级系统通常改用深度人脸嵌入（如 FaceNet/ArcFace）。

### 4. 工具一览

| 工具 | 作用 | 备注 |
|------|------|------|
| **OpenCV** (`opencv-contrib-python`) | 读写图像/视频、Haar 检测、LBPH 识别 | `cv2.face` 仅在 contrib 版提供 |
| **MediaPipe** | 人脸关键点 (Face Mesh) | 深度学习模型，开箱即用 |
| **NumPy** | 数组/标签运算 | OpenCV 的底层数据结构 |
| **scikit-learn** | 提供 Olivetti/ORL 公开数据集 | 仅 `verify_olivetti.py` 用，`--extras verify` 安装 |
| **Poetry** | 依赖与虚拟环境管理 | 见下方“安装” |

## 安装

### 关于 Poetry

[Poetry](https://python-poetry.org/) 是 Python 的依赖管理与打包工具，相比 `pip + requirements.txt` 它能：

- **统一声明依赖**：所有依赖、版本约束、可选 extras 都写在 `pyproject.toml`，一目了然。
- **可复现安装**：`poetry.lock` 锁定每个包的精确版本，保证你我他装出来的环境完全一致（例如本项目就用它把 `mediapipe` 锁在仍带 `mp.solutions` 的 `0.10.21`）。
- **隔离虚拟环境**：Poetry 会为项目自动创建独立虚拟环境，不污染全局 Python，也无需手动 `python -m venv`。
- **匹配 Python 版本**：依据 `requires-python` 自动挑选合适的解释器（本项目要求 3.11–3.12）。

安装 Poetry（macOS / Linux，详见[官方文档](https://python-poetry.org/docs/#installation)）：

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry --version   # 验证安装
```

### 安装依赖

在本项目目录下执行：

```bash
poetry install                 # 安装核心依赖
poetry install --extras verify # 额外安装 verify_olivetti.py 所需的 scikit-learn
```

随后用 `poetry run python <脚本>.py ...` 运行脚本（无需手动激活环境）；
也可以执行 `poetry env activate`（旧版本为 `poetry shell`）进入虚拟环境后直接 `python <脚本>.py`。
本文后续命令统一使用 `poetry run` 形式。

> 重要：LBPH 识别器位于 `cv2.face` 中，仅在 **`opencv-contrib-python`** 中提供。
> 不要同时安装 `opencv-python`，否则二者会互相覆盖导致 `cv2.face` 不可用。
> 如已装过 `opencv-python`，先 `pip uninstall opencv-python`。
>
> 本项目限定 Python 3.11–3.12：上限受 MediaPipe wheel 限制（暂无 3.13+），下限受 `verify` 额外依赖 scikit-learn（需 ≥3.11）限制。

## 整体设计与主要流程

### 设计理念

- **小而专的脚本**：每个计算机视觉概念对应一个独立可运行脚本（检测、关键点、采集、训练、识别），便于单独学习与演示，而不是堆在一个大文件里。
- **共享底座 `common.py`**：把“输入源抽象 + Haar 检测 + 显示/退出”等公共逻辑集中起来，所有脚本复用同一套代码路径与 `--source` 约定，避免重复。
- **统一输入抽象**：`iter_frames(source)` 把图片、视频、摄像头统一成“逐帧产出”的迭代器，上层脚本无需关心输入到底是什么。
- **文件化的中间产物**：数据集 (`dataset/`) 和模型 (`model/`) 都落地为可检视的文件（PNG + JSON + YAML），训练与识别通过文件解耦，可分别独立运行、复现。
- **CLI 驱动**：每个脚本用 `argparse` 暴露参数，无 GUI/服务依赖，方便在课堂、终端、CI 中跑通。

### 模块协作

```
            ┌────────────┐
            │ common.py  │  输入源迭代 + Haar 检测 + 显示/退出
            └─────┬──────┘
       ┌──────────┼───────────┬───────────┬───────────┐
       ▼          ▼           ▼           ▼           ▼
   detect.py  landmarks.py  capture.py   train.py   recognize.py
   (检测可视化) (关键点可视化) (采集样本)   (训练模型)  (识别身份)
                               │            │            ▲
                               ▼            ▼            │
                           dataset/  ───►  model/  ──────┘
                        样本+labels.json   lbph.yml+labels.json
```

`detect.py` 与 `landmarks.py` 是**独立的可视化工具**，不参与训练/识别链路；真正的识别能力来自下面两条主流程。

### 主流程一：训练（离线，建库）

```
摄像头/图片  →  capture.py  →  检测人脸 → 裁剪 → 转灰度 → resize(200×200)
                              → 存为 dataset/<label>_<username>/NNN.png
                              → 更新 dataset/labels.json (label→用户名)
                                          │
                                          ▼
                          train.py  →  读取全部灰度样本 + 标签
                                    →  LBPHFaceRecognizer.train()
                                    →  写出 model/lbph.yml + model/labels.json
```

### 主流程二：识别（在线，1:1 核验）

```
摄像头/图片/视频  →  recognize.py  →  load model/lbph.yml + labels.json
                                   →  每帧: Haar 检测人脸框
                                   →  裁剪灰度 → resize(200×200)
                                   →  recognizer.predict() → (label, distance)
                                   →  distance ≤ threshold ?
                                        ├─ 是 → 绿框: 用户名 + 置信度
                                        └─ 否 → 红框: unknown
                                   →  终端打印 label/用户名/置信度
```

> 关键约定：采集、训练、识别三处都必须用**相同的灰度化与尺寸 (200×200)** 预处理，LBPH 才能正确比对。该逻辑集中在脚本中以保持一致。

## 目录结构

```
opencv-mediapipe-demo/
├── common.py          # 共享：输入源(图片/视频/摄像头) + Haar 人脸检测
├── detect.py          # 人脸检测
├── landmarks.py       # MediaPipe 人脸关键点 + 网格可视化
├── capture.py         # 采集灰度人脸样本到数据集
├── train.py           # 训练 LBPH 模型
├── recognize.py       # 加载模型并进行 1:1 身份识别
├── verify_olivetti.py # 用公开数据集 Olivetti/ORL 快速验证准确率
├── dataset/           # 采集的样本：<label>_<username>/NNN.png + labels.json
└── model/             # 训练产物：lbph.yml + labels.json
```

`--source` 参数统一约定：`0`（或其它数字）表示摄像头索引，`xxx.jpg/.png` 表示图片，`xxx.mp4` 等表示视频。流式输入按 `q` 或 `ESC` 退出。

## 使用

### 1. 人脸检测

```bash
poetry run python detect.py --source 0                  # 摄像头
poetry run python detect.py --source face.jpg           # 图片（按任意键关闭）
poetry run python detect.py --source clip.mp4 --shape circle
poetry run python detect.py --source face.jpg --save out.jpg
```

### 2. 人脸关键点

```bash
poetry run python landmarks.py --source 0
poetry run python landmarks.py --source face.jpg --save mesh.jpg
```

### 3. 采集数据集

```bash
poetry run python capture.py --username walter --label 1 --count 30
poetry run python capture.py --username alice  --label 2 --count 30
```

会把灰度人脸裁剪保存到 `dataset/1_walter/000.png ...`，并维护 `dataset/labels.json`。

### 4. 训练模型

```bash
poetry run python train.py
```

读取 `dataset/` 中的样本，训练后保存到 `model/lbph.yml`，标签映射保存到 `model/labels.json`。数据集为空时会报错且不生成模型。

### 5. 身份识别

```bash
poetry run python recognize.py --source 0
poetry run python recognize.py --source group.jpg --threshold 70
```

检测人脸后用 LBPH 预测，命中则标注用户名+置信度（绿框），否则标注 `unknown`（红框），同时在终端打印匹配标签、用户名、置信度。

### 快速验证（无需摄像头/隐私安全）

使用公开数据集 Olivetti/ORL（scikit-learn 自动下载，40 人 × 10 张 64×64 灰度）做端到端正确性检查：

```bash
poetry run python verify_olivetti.py --train-per-person 8
```

会按每人前 N 张训练、其余测试，并打印识别准确率。

## 关于置信度

LBPH 的 `predict` 返回的 **confidence 是“距离”，数值越小越相似**。`recognize.py` 用 `--threshold` 作为可接受的最大距离（默认 70），小于等于阈值才判定为匹配。可根据数据集和场景调整。

## 已知限制

- Haar 级联检测在弱光照、大角度、遮挡场景下准确率有限，仅适合教学演示。
- LBPH 适合小规模灰度样本，光照/姿态变化大时识别率下降；可用 Yale/Extended Yale B 数据集观察该现象。
- 仅做 1:1 身份核验，不含 1:N 大库检索。
