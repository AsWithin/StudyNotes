# StudyNotes

个人 Python 与机器学习学习笔记。

私有仓库：https://github.com/AsWithin/StudyNotes

## 项目结构

- `MachineLearning.ipynb`：机器学习学习笔记。
- `UsageExample.ipynb`：Notebook 使用示例。
- `Orthers/LinearRegression.py`：线性回归、梯度下降与成本曲线。
- `Orthers/LogisticRegression.py`：逻辑回归，支持静态图与动态回放。
- `Orthers/Overfitting.py`：可导入 Notebook 的过拟合实验，支持回归/分类、1~6 阶拟合、点击添加样本。
- `images/`：笔记配图，迁移时需要一起保留。
- `requirements.txt`：运行示例及 VS Code Notebook 的基本依赖。
- `.gitignore`：排除虚拟环境、缓存和本地密钥文件。

## 在另一台 Windows 电脑上开始学习

### 1. 准备软件与 GitHub 登录

安装 Git、Python、VS Code；在 VS Code 中安装 Microsoft 的 Python 和 Jupyter 扩展。
需要 AI 协助时，安装并登录 Codex，打开本项目所在的文件夹。

这是私有仓库，克隆和推送需要有权限的 GitHub 账号。首次 Git 操作可能弹出登录授权，请在官方登录界面完成；不要把密码或令牌粘贴进代码、README 或对话。

### 2. 克隆项目

在 PowerShell 中进入你希望存放项目的父目录，再执行：

```powershell
git clone https://github.com/AsWithin/StudyNotes.git
cd StudyNotes
```

路径可以自行选择，不必与原电脑的 `D:\PythonCode\StudyNotes` 相同。若目录里已有项目，先检查其状态，不要覆盖或再次嵌套克隆。

### 3. 创建项目独立环境并安装依赖

以下命令都在项目根目录运行：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import numpy, matplotlib, ipykernel; print('Environment OK')"
```

如果 `python` 命令不可用，但已安装 Windows Python Launcher，可用 `py -m venv .venv` 创建环境。
直接使用虚拟环境中的 Python 路径即可，无需激活脚本，也无需修改 PowerShell 执行策略。

依赖清单没有锁定版本，安装时会解析兼容版本；它用于快速建立环境，不保证两台电脑的软件版本完全一致。`.venv` 不上传 GitHub，每台电脑各自创建。

### 4. 选择 VS Code 解释器和笔记本内核

1. 用 VS Code 打开整个 `StudyNotes` 文件夹。
2. 执行 `Python: Select Interpreter`，选择 `.venv\Scripts\python.exe`。
3. 打开 `.ipynb`，在右上角的内核选择器中选择该 `.venv` 环境。
4. 按顺序运行单元格；上方单元格定义的变量和函数供后续单元格使用。

若中文图表字体显示为方框，检查系统是否有笔记中指定的中文字体，或将字体配置改为该电脑已安装的中文字体。

## 运行学习案例

```powershell
.\.venv\Scripts\python.exe Orthers\LinearRegression.py
.\.venv\Scripts\python.exe Orthers\LogisticRegression.py
```

逻辑回归脚本提示选择模式：

- 输入 `0`：最终决策边界、完整成本、最后 100 次训练成本三幅静态图。
- 输入 `1`：动态回放，支持暂停、重播和拖动训练次数。

静态绘图函数为 `plot_static(X_train, y_train, w_out, b_out, J_history)`，适合整理到笔记中。普通 Notebook 静态图不需要 `ipympl`。

如果以后要在 Notebook 中使用交互图，需要额外安装：

```powershell
.\.venv\Scripts\python.exe -m pip install ipympl
```

重启笔记本内核，在绘图前的代码单元运行 `%matplotlib widget`。这条魔法命令仅用于 Notebook，不要写入普通 `.py` 脚本。交互图需要内核运行，静态输出更适合直接保存和阅读。

### 在笔记中使用过拟合案例

在项目根目录的 Notebook（例如 `MachineLearning.ipynb`）中，首次先运行：

```python
%pip install ipympl
```

安装后重启内核，再运行下面的单元格。交互模式设置必须放在绘图前：

```python
%matplotlib widget
import matplotlib.pyplot as plt
from Orthers.Overfitting import overfit_example

plt.close("all")
ofit = overfit_example(False)  # False：分类；True：回归
```

不需要截图中的 `display(output)`，图表和控件由模块直接创建。保留 `ofit` 变量，避免控件回调随对象被回收。

- **Degree**：选择 1~6 阶，然后点击 **fit data**。更换阶数或添加数据后，旧拟合清除，需重新拟合。
- **Regression / Categorical**：切换回归和分类；每种模式分别保留添加的样本。
- **添加样本**：点击绘图区。分类默认添加红色 `y=1`；按住 Shift 点击添加蓝色 `y=0`，或通过 **New class** 选择标签。启用工具栏平移/缩放时不会添加数据。
- **Reset data**：恢复当前模式的初始样本。默认种子固定，重新运行可复现实验。
- 橙色点线 **ideal** 是生成数据的真实二次函数/分类边界；绿色线是拟合函数或预测概率为 0.5 的分类边界。

建议在**同一批数据**上分别尝试 1、2、6 阶：1 阶无法表示真实的二次关系，通常欠拟合；高阶模型可能追随噪声。观察训练误差下降时，独立测试误差是否升高，而不只是看曲线是否弯曲。回归显示均方误差 MSE；分类显示交叉熵和准确率。测试集由同一分布独立生成，不参与拟合；若反复参考这些分数选模型，它实际承担的是验证集的作用。不是所有随机种子或新增样本都会让 6 阶最差。

训练代码使用 NumPy：回归采用最小二乘，分类采用牛顿法和回溯线搜索，含极弱 L2（`1e-6`，截距不惩罚）以稳定数值。无需 scikit-learn 或 SciPy。图中文字使用英文，避免未安装中文字体时显示方框；代码注释为中文。

也可以用代码控制，并查看数值：

```python
ofit.set_mode("Regression")
ofit.set_degree(6)
metrics = ofit.fit()
print(metrics)
# ofit.add_point(0.5, 1.2)  # 添加样本后重新 fit()
# ofit.reset()
```

若只需静态图片，可在 `%matplotlib inline` 模式使用 `show=False`，再调用 `fit()` 和 `plt.show()`；静态图的按钮无法点击。桌面窗口也支持交互：`python Orthers/Overfitting.py`。交互图需要运行中的内核，重新打开笔记时应重新运行单元格。

交互后端说明：[Matplotlib ipympl 官方文档](https://matplotlib.org/ipympl/)。

更新 `Overfitting.py` 后，已运行的 Notebook 会继续使用内存里的旧版本。可重启内核后重新运行，或用以下单元格关闭旧图并重新加载模块：

```python
%matplotlib widget
import importlib
import matplotlib.pyplot as plt
import Orthers.Overfitting as overfitting

plt.close("all")
importlib.reload(overfitting)
ofit = overfitting.overfit_example(False)
```

交互已合并自动刷新请求，并复用散点、曲线和分类网格；按钮悬停不再刷新画布。Notebook 中的鼠标操作仍需要内核处理，如果其他单元格正在长时间运算，应等待其结束后再操作图表。

## 两台电脑之间同步

GitHub 不会自动上传保存的文件。需要先提交（commit）再推送（push），另一台电脑才能拉取（pull）。

### 开始学习前

```powershell
git status
git pull --ff-only
```

先确认工作区没有尚未处理的修改。如果拉取提示本地修改、分支分歧或冲突，先保留当前工作并检查原因，不要直接强制覆盖。`--ff-only` 会在分支分歧时停下。

### 学习结束后

先保存 VS Code 中的文件，检查修改内容：

```powershell
git status
git diff --stat
git diff
```

确认没有密钥、隐私内容或无关文件后再提交上传：

```powershell
git add .
git commit -m "更新学习笔记"
git push
```

如果没有修改，不需要创建提交。笔记本输出也是文件内容的一部分，提交前应一起检查。

首次提交若提示缺少作者身份，可仅为此仓库设置：

```powershell
git config user.name "AsWithin"
git config user.email "145670195+AsWithin@users.noreply.github.com"
```

上面的身份对应本仓库所有者，并使用 GitHub 隐私邮箱。提交身份与 GitHub 登录授权是两件事。

尽量避免两台电脑在未同步时同时编辑同一个 `.ipynb`。换电脑前完成推送，另一台电脑开始前先拉取。

## 发给新电脑上 Codex 的指令

可以复制下面这段话，把目标路径换成你实际希望存放项目的位置：

> 请帮我配置学习项目 https://github.com/AsWithin/StudyNotes，这是我的私有仓库。目标目录为 D:\PythonCode\StudyNotes。先检查目标目录及 Git、Python 和 GitHub 登录状态；如果目录不存在就克隆，如果已经存在就检查仓库和未提交修改，不要覆盖已有文件。阅读项目 README.md，在项目内创建 .venv，按 requirements.txt 安装依赖，验证 NumPy、Matplotlib 和 ipykernel 能正常导入，并检查学习示例的训练部分能否运行。请告诉我在 VS Code 中应选择哪个解释器和 Notebook 内核。遇到 GitHub 登录或系统权限要求时，明确告诉我需要完成哪一步，不要索要或记录明文密码、令牌。最后阅读机器学习笔记和回归示例，总结当前学习进度，继续用中文辅导我。

Codex 可以在获得相应工具和权限的情况下执行这些步骤；私有仓库登录、软件安装或系统审批可能需要你亲自完成。

项目文件随 Git 同步；Codex 对话记录不会由本仓库自动打包迁移。新对话可以通过读取项目和你补充的学习进度继续工作。
