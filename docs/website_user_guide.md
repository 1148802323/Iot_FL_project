# AI4I Federated Predictive Maintenance Web Platform 使用说明书

## 1. 网站用途

本网站用于展示和运行 AI4I 预测性维护联邦学习实验。用户可以查看已有实验结果、登录后创建新的联邦学习实验、运行不同算法，并查看实验状态、最终指标和收敛曲线。

当前网站包含以下主要模块：

- Overview
- Results
- Experiments
- Live Prediction
- Factory Split
- Figures
- Reproduce
- Login

## 2. 启动方式

网站前端默认通过本地静态服务访问：

```bash
cd /Users/xuan/Desktop/Iot_FL_project_stage3-flower
python -m http.server 8080
```

浏览器打开：

```text
http://127.0.0.1:8080/site/
```

后端 API 默认地址是：

```text
http://127.0.0.1:8000
```

如果需要启动后端：

```bash
PYTHONPATH=src uvicorn iot_fl.backend.main:app --reload
```

## 3. 登录与注册

进入左侧导航栏的 **Login** 页面。

### 登录

输入：

- Username
- Password

点击 **Login**。

登录成功后，网站会把 JWT token 保存到浏览器本地存储中，后续访问实验接口时会自动携带该 token。

### 注册

注册表单包含：

- Username
- Email
- Password
- Role
- Factory ID

普通用户选择 `client`，并填写 Factory ID。管理员用户选择 `admin`，Factory ID 会自动禁用。

### 会话操作

登录后可以使用：

- **Me**：查看当前登录用户
- **Admin Check**：测试当前用户是否有管理员权限
- **Logout**：退出登录

如果 token 失效或后端返回 `401`，网站会自动清除登录状态，并提示重新登录。

## 4. Experiments 实验管理

进入左侧导航栏的 **Experiments** 页面。

实验功能需要先登录。如果未登录，页面会显示提示信息：

```text
Login to create experiments and view your experiment history.
```

### 上传数据集

Experiments 页面现在支持客户端上传 CSV 数据集。

上传区域包含：

- Dataset CSV
- Upload Dataset

点击 **Upload Dataset** 后，网站会调用：

```text
POST /api/datasets
```

后端会完成以下工作：

1. 保存原始 CSV 文件。
2. 校验是否包含 AI4I 实验所需字段。
3. 生成标准化后的 `processed.csv`。
4. 自动生成 `iid`、`moderate_non_iid`、`highly_non_iid` 三种 factory split。
5. 把数据集保存到当前登录用户账户下。

上传成功后，该数据集会出现在：

- Dataset 下拉框
- Available Datasets 列表

点击数据集列表中的某个数据集，可以把它选为下一次实验使用的数据集。

上传 CSV 至少需要包含这些 AI4I 字段：

- `UDI`
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`

如果包含故障模式字段，系统也会使用：

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

如果缺少故障模式字段，后端会把缺失字段按 0 处理。

### 创建实验

实验表单包含：

- Algorithm
- Distribution
- Global Rounds
- Local Epochs
- Learning Rate
- Dataset

Algorithm 选项来自后端接口：

```text
GET /api/algorithms
```

当前支持：

- FedAvg
- Failure-Aware FedAvg V1
- Failure-Aware FedAvg V2
- Dynamic Failure-Aware FedAvg

Distribution 与 Algorithm 是分开的选项，当前支持：

- IID
- Moderate Non-IID
- Highly Non-IID

填写配置后，点击 **Start Experiment**。

网站会先调用：

```text
POST /api/experiments
```

创建实验记录，然后调用：

```text
POST /api/experiments/{experiment_id}/run
```

执行实验。

如果 Dataset 选择为 `Default project dataset`，实验会使用项目自带 AI4I 数据。

如果 Dataset 选择为上传的数据集，实验会使用该上传数据集生成的 processed CSV 和 factory splits。

### 实验状态

实验状态面板会显示：

- Experiment ID
- Algorithm
- Distribution
- Status
- Created time
- Rounds
- Local epochs
- Learning rate

状态可能是：

- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`

只有后端算法执行成功后，实验才会显示为 `COMPLETED`。如果执行失败，状态会变为 `FAILED`，并显示后端保存的错误信息。

### 查看实验结果

实验完成后，Final Metrics 区域会显示：

- Accuracy
- Precision
- Recall
- F1
- Communication Cost
- Training Time

这些数据全部来自后端数据库，不是前端生成的假数据。

### 收敛曲线

Convergence 区域会根据后端返回的 `convergence_history` 绘制曲线。

优先显示：

- validation F1
- validation recall
- mean client loss

具体显示哪一种指标取决于算法返回的历史记录字段。

### 实验历史

页面底部会显示当前用户可见的历史实验：

- ID
- Algorithm
- Distribution
- Status
- Recall
- F1
- Training Time
- Created At

普通用户只能看到自己的实验。管理员可以看到全部实验。

点击某一行历史实验，可以重新打开该实验的详细状态、最终指标和收敛曲线。

## 5. Results 结果展示

进入 **Results** 页面可以查看项目已有的静态实验结果。

该页面从本地 `reports/*.csv` 文件加载数据，用于展示：

- Centralized baseline
- FedAvg baseline
- 不同分布下的 Precision / Recall / F1
- FedAvg validation F1 收敛趋势

这部分是已有研究结果展示，不会创建新的实验记录。

## 6. Live Prediction 实时预测

进入 **Live Prediction** 页面。

用户可以输入一条机器传感器记录：

- Product Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

选择模型后点击 **Generate Prediction**。

页面会显示：

- Failure probability
- Risk label
- Model threshold
- Temperature gap
- Power proxy
- Decision
- Main risk contributors

该模块使用本地保存的模型参数进行前端推理，不会调用后端实验训练接口。

## 7. Factory Split 工厂拆分

进入 **Factory Split** 页面。

可以切换：

- IID
- Moderate Non-IID
- Highly Non-IID

页面展示每个 factory client 的：

- 样本数量
- 故障率
- 主要故障模式

该模块用于解释联邦学习客户端数据分布。

## 8. Figures 图表

进入 **Figures** 页面可以查看项目生成的图像结果，例如：

- FedAvg final metric comparison
- FedAvg convergence curve
- Centralized baseline metrics
- Confusion matrix
- Highly Non-IID client distribution
- Feature correlation heatmap

点击图卡可以打开对应图片。

## 9. Reproduce 复现实验

进入 **Reproduce** 页面可以查看本地复现实验命令，包括：

- 数据准备
- Centralized baseline
- FedAvg baseline

这些命令用于在本地重新生成研究结果。

## 10. 常见问题

### 10.1 Experiments 页面提示需要登录

说明当前浏览器没有有效 JWT token。请进入 **Login** 页面登录或注册。

### 10.2 点击 Start Experiment 后报错

常见原因：

- 后端服务没有启动
- 输入参数不合法，例如 rounds 小于等于 0
- 算法执行时发生错误
- token 已过期

可以先检查后端是否正常：

```text
http://127.0.0.1:8000/api/health
```

正常返回：

```json
{"status":"ok"}
```

### 10.3 页面加载不到静态结果

请确认是通过本地 HTTP 服务访问网站，而不是直接双击打开 HTML 文件。

推荐访问方式：

```text
http://127.0.0.1:8080/site/
```

### 10.4 新实验运行时间较长

实验运行时间取决于：

- Global Rounds
- Local Epochs
- 所选算法
- 当前机器性能

如果只是测试功能，建议先使用：

- Global Rounds: `1` 或 `5`
- Local Epochs: `1`

## 11. 推荐演示流程

1. 启动后端 FastAPI。
2. 启动前端静态网站。
3. 打开 `http://127.0.0.1:8080/site/`。
4. 进入 **Login** 页面并登录。
5. 进入 **Experiments** 页面。
6. 选择算法，例如 FedAvg。
7. 选择分布，例如 Highly Non-IID。
8. 设置：
   - Global Rounds: `5`
   - Local Epochs: `1`
   - Learning Rate: `0.01`
9. 点击 **Start Experiment**。
10. 等待实验完成。
11. 查看状态、最终指标和收敛曲线。
12. 在实验历史表中点击该实验，确认可以重新打开结果。
