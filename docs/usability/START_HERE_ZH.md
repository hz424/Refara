# 首次使用练习

用提供的小型合成数据，独立完成导入、运行和结果解释。本次练习记录已了解框架的项目成员首次操作，不能作为陌生用户测试。自己的研究数据可另开一次练习。

任务限时 45 分钟，环境准备完成后开始计时。安装、查找 Python 环境和阅读数据说明的时间单独记录。可以查文档、命令帮助和报错；先独立修改配置、解释结果。遇到问题时保留报错并记录停在哪一步；如向作者求助，记录帮助内容，并将本次标记为有帮助完成。

## 准备

在软件包根目录操作。将 `PY` 设为已安装此版本及 AnnData 依赖的 Python 路径，先运行：

```bash
PY=/完整路径/到/python
"$PY" -m reference_design --help
"$PY" -c 'import anndata, h5py; print(anndata.__version__, h5py.__version__)'
```

若这里失败，先记录安装问题和用时，再完成环境准备。开始任务前记录起始时间。请使用全新的 `pilot_attempt` 目录，保留自己的配置和输出，不覆盖材料。

```bash
mkdir pilot_attempt
cp -R docs/usability/materials/A pilot_attempt/A
cp -R docs/usability/materials/B pilot_attempt/B
cp -R docs/usability/materials/C pilot_attempt/C
```

## 数据说明

每组材料的 `cells.h5ad` 中，一行是一枚细胞。`obs` 的 `patient` 是独立供体，`treatment` 是实验状态，`compartment` 是细胞类型，`lane` 是需要等权的采集层。`vehicle` 是对照，`stim` 是处理。只比较 `tasks.tsv` 指定的供体、状态和细胞类型。以供体等权汇总；同供体内任务等权。

使用 `layers['measured']`，这些数值已做完预处理。`X` 是另一套数值，不适用于此次比较。基因用 `var_names` 匹配。处理组需要先在每个 lane 内求均值，再让各 lane 等权。每个对照 lane 有四枚可用细胞，每块要抽两枚，每次分配内不同块不得重复使用细胞。每组任务做两次分配，种子 19。

`first.h5ad` 与 `second.h5ad` 都是一任务一行，任务名在 `obs['task_name']`，预测数值在 `X`，基因在 `var_names`；行列顺序不保证相同。A、B 中两模型直接预测效应量，没有使用此次预测对照细胞。要比较的是对新观测效应的预测误差。C 的 first 模型原本预测处理状态，文件中保存的是减去了某个基线的数值；提供者没有交付该基线。C 的 second 仍是直接效应预测。模型来源说明必须保留。

## 三个任务

对 A、B、C 分别决定能否完成目标比较。编辑各目录中的 `manifest.template.json`，将 `请填写` 和块数 `0` 换成适用配置，另存为 `manifest.json`。可以检查输入文件；不修改原始表达值、模型来源说明或模型类型以绕过报错。

下面以 A 为例，B、C 使用相同命令结构；若认为某项资料不足，请记录原因和需要的资料：

```bash
"$PY" -m reference_design prepare pilot_attempt/A/manifest.json --output pilot_attempt/A/prepared
"$PY" -m reference_design run pilot_attempt/A/prepared/plan.json --output pilot_attempt/A/results
```

对能够运行的任务，请依据预览和结果报告回答：

1. 实际用了哪些细胞、表达层、对照块和汇总单位？
2. 两模型的总体误差各是多少，哪个较小？结果是否支持你最初的比较目标？
3. 哪些参考敏感性判断能从输出得到，哪些还缺资料？下一步是否需要补数据或改变比较？

对不能运行的任务，记录错误、原因，以及能否在不改变问题含义的前提下解决。45 分钟到时停止，保存已完成的步骤和答案。

## 交回记录

复制并填写 `record.csv`，或直接用文字回答：准备用了多久、任务开始和结束时间、各任务完成到哪里、关键结果/报错、查了哪些文档、是否获得帮助、最难理解的一步。保留 `pilot_attempt` 中的合成数据配置和输出，无需提交自己的研究数据。

若接着尝试自己的数据，另建目录，并单独记录数据来源、预处理、模型输出含义、对照来源及映射工作量。分别记录合成材料和自己数据的运行结果。
