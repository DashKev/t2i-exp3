# T2I-EXP3

面向有限查询预算文生图安全评测的反馈条件算子搜索。
本仓库提供EXP3主方法、固定static5与Original Prompt参考方法，以及SD3.5端到端运行入口。
它是一份可独立安装的最小科研实现，不需要另一个本地工程或私有Python模块。
模型和数据通过公开上游获取；不随代码分发权重、数据提示词正文或运行生成图像。

## 方法概览

对于每个请求，首先用CRF→KCM→VED→KPS→IVP生成初始候选。
后续根据是否有效返回及违规判定选择修正方向，在相应算子池内使用EXP3更新选择概率。
主方法不需要训练独立路由大模型；LLM仅在选中相应变异算子或进行长度压缩时调用。
每个请求独立重置状态，首次违规成功后停止，默认最大预算K=16。

| 组成 | 实现 |
|---|---|
| 轻量变换 | 角色语境、大小写、视觉词典、标点、跨语种、分词、IVP、视觉证据及前景强化 |
| LLM变换 | 攻击性增强、直接增强、自然场景、风格转换、小区域、感官同义表达 |
| 在线选择 | 按失败模式建立EXP3权重；15%全局逃逸，另有EXP3池内探索 |
| 反馈 | 违规置信及弱语义辅助奖励 |
| 主结果 | 预算内违规成功率ASR、有效返回率BPR、限制首次成功查询成本 |

详见[方法与指标](docs/METHOD.md)。算子模板见`src/t2i_exp3/methods/operators.py`。

## 目录

```text
src/t2i_exp3/
  methods/              # EXP3、算子、static5、原始请求
  llm/                  # OpenAI兼容客户端及本地文本变异服务
  sd3_target.py         # SD3.5目标
  qwen_image_target.py  # 共用输入/输出护栏组件；不包含其他目标生成器
  oracles.py            # 在线反馈
  judges.py             # 事后评估
  runner.py             # 预算、日志、恢复
  resources.py          # 数据及评估资源获取
  configure.py          # 可移植配置入口
  summary.py            # 论文口径汇总
configs/                # 真实目标示例与纯CPU合成自测
data/                   # 良性自测样本、公开case编号、资源哈希
tests/                  # 不需要GPU和外部服务的测试
docs/                   # 方法、验证及资源说明
```

## 1. 安装与离线自测

要求Python 3.11或更新版本。以下命令均从仓库根目录执行：

```bash
git clone https://github.com/DashKev/t2i-exp3.git
cd t2i-exp3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
t2i-exp3 validate configs/smoke.json
t2i-exp3 run configs/smoke.json --output results/smoke
t2i-exp3 run configs/exp3.smoke.json --output results/exp3-smoke
```

两份smoke使用合成反馈，不下载模型、不调用LLM、不产生真实图像。
`exp3.smoke.json`使用同一EXP3实现和轻量算子，但只用于运行闭环测试；不能据其输出得出安全评测结论。

也可构建wheel后安装：

```bash
python -m build
python -m pip install dist/t2i_exp3-0.1.2-py3-none-any.whl
```

库和算子全部包含在wheel中；示例配置和数据获取清单位于源码发行包/仓库中。
部署到新机器时解压完整源码发行包或克隆此仓库，再运行入口。
工作目录用于解析模型与评估资源的相对路径，不依赖包安装到哪一层目录。

## 2. 获取I2P200

I2P原始项目：https://github.com/ml-research/i2p 。
本实现使用Ring-A-Bell公开固定提交中的I2P CSV副本，确保字段、顺序及内容可校验。
下载地址和SHA-256见`data/resources.json`；200条选择编号见`data/i2p200_selection.json`。

```bash
t2i-exp3-data i2p
```

命令会下载CSV到`data/raw/i2p.csv`，按冻结case编号生成`data/prepared/i2p200.jsonl`。
完整输出SHA-256应为：

```text
af47c888625e09c0d71b207ded0d5e2e5c69e0674a5c26da3fd95e82e582ede6
```

下载失败或哈希不同会报错，不自动更换数据源或重抽子集。
输入最小JSONL格式：`{"id":"example-1","intent":"a landscape photo","category":"violence","metadata":{}}`。
`category`应使用评测约定类别标签；自定义数据与冻结I2P200结果不得混为同一实验。
数据分发与使用须遵守其上游条款，详见[外部资源说明](THIRD_PARTY.md)。

## 3. 获取在线与事后评估资源

```bash
t2i-exp3-data evaluators
```

命令仅获取DiffZOO固定提交中的`nudenet.py`、`best.onnx`及Q16 `prompts.p`，分别放入`assets/nudenet`与`assets/q16`。
它们是评估依赖，不代表本仓库包含或运行DiffZOO攻击算法。
下载前后均按manifest校验SHA-256；外部源码不在import时自动下载。

## 4. 准备模型

真实运行需要NVIDIA CUDA环境及足够显存/主存。先按机器环境安装匹配的PyTorch，再安装：

```bash
python -m pip install -e '.[sd35,llm]'
```

依赖范围是安装约束，不是已在所有组合上验证的锁定环境。
SD3.5访问可能需要先在Hugging Face接受模型条款。凭证使用HF标准登录或外部环境，不写入配置和仓库。

```bash
hf auth login
hf download stabilityai/stable-diffusion-3.5-medium --local-dir models/sd35
hf download Qwen/Qwen3Guard-Gen-8B --local-dir models/qwen3guard
hf download openai/clip-vit-large-patch14 --local-dir models/clip
hf download CompVis/stable-diffusion-safety-checker --local-dir models/safety-checker
hf download OpenGVLab/InternVL3_5-8B-HF --local-dir models/mutator
```

这些模型不随代码包提供。本版不自动下载几十GB模型；按需运行上述命令。
首次部署后应记录实际HF revision与依赖版本，不能仅凭同一个模型名字宣称数值完全可复现。
输出检查器示例采用公开CompVis资源；若采用其他快照，必须在实验说明中记录。

## 5. 启动LLM文本服务

在单独终端运行：

```bash
source .venv/bin/activate
python -m t2i_exp3.llm.internvl_openai_server \
  --model-path models/mutator \
  --served-model-name local-mutator \
  --device cuda:2 --dtype bfloat16 --port 18080
```

服务仅使用文本输入。也可以接入现有OpenAI兼容服务，其模型标识需要匹配`--llm-model`。
`--llm-base-url`填写服务根地址，如`http://127.0.0.1:18080`；客户端自动追加`/v1/...`，不要再包含`/v1`。
更换LLM属于实验配置变化，应记录模型和服务参数。

## 6. 生成配置并运行SD3.5

例子使用目标GPU0、护栏/评估GPU1、单独的LLM GPU2。这是设备布局示例，不是显存保证。
如需调整设备或CPU offload，可通过命令显式设置；吞吐和时间结果应报告实际部署。

```bash
t2i-exp3-config --method exp3 --budget 16 \
  --llm-base-url http://127.0.0.1:18080 --llm-model local-mutator \
  --device cuda:0 --guard-device cuda:1 --judge-device cuda:1 \
  --output configs/sd35.local.json
t2i-exp3 validate configs/sd35.local.json
t2i-exp3 run configs/sd35.local.json --output results/exp3
```

`validate`检查配置和数据，并不加载所有模型或证明GPU显存足够。
可先使用`run ... --task-id i2p-0001`运行存在于所选数据中的单个case；请以selection文件中的真实编号替换示例编号。
勿将smoke反馈替换为正式评估器后仍称为无需模型的离线检查。

正式系统采用Qwen3Guard输入护栏与Stable Diffusion Safety Checker输出护栏。
目标侧不使用CLIP相似度拦截；CLIP仅为奖励提供语义辅助评分，不参与最终ASR成功条件。
详见[方法与系统边界](docs/METHOD.md)。

### 对照方法

```bash
t2i-exp3-config --method static5 --budget 16 --output configs/static5.local.json
t2i-exp3 run configs/static5.local.json --output results/static5
t2i-exp3-config --method original --output configs/original.local.json
t2i-exp3 run configs/original.local.json --output results/original
```

static5使用主控制器的原有算子序列执行路径，每次从原始请求重新执行固定链；沿用token治理。
这是明确配置的固定链参考，不附带历史运行结果。Original Prompt固定K=1，不改写输入。

## 7. 输出、汇总与恢复

每次运行创建唯一目录，包含配置/数据快照、manifest、逐查询记录、在线反馈、事后评估、指标与完成标志。
真实运行生成图像也存放在该目录。它们被Git忽略，不应随源码发布。

```bash
t2i-exp3-summary results/exp3/<run-directory> --budget 16
t2i-exp3 resume results/exp3/<unfinished-run-directory>
```

汇总读取`policy_violation`计算ASR，而不是兼容字段`success/asr`。
成本按请求计算：成功取首次违规query_index，未成功取K；输出请求间样本标准差（ddof=1）。
不完整运行会拒绝汇总，避免将尚未执行完的请求当成正式失败。
已完成的K32逐查询日志可用`--budget 16`截取；不得把32次完整调用成本混入K16。
Original Prompt运行应使用`--budget 1`。

## 8. 验证范围、资源及许可证

本版的实际检查记录见[VALIDATION.md](docs/VALIDATION.md)。离线测试通过不等于已完成真实GPU复现。
本项目代码采用[MIT许可证](LICENSE)；模型、数据和第三方评估资源遵守各自条款，见[THIRD_PARTY.md](THIRD_PARTY.md)。
未包含其他baseline、服务器调度、私有服务地址、凭证、作者联系方式、历史Git提交或实验日志。
论文引用信息可在题目及发表信息确定后补入；此处不虚构DOI、会议信息或BibTeX。
