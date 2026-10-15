# 外部资源与权利说明

此仓库没有打包第三方模型、数据正文、NudeNet源码或Q16二进制。

| 资源 | 官方来源 | 获取方式 |
|---|---|---|
| I2P | https://github.com/ml-research/i2p | 本仓库使用Ring-A-Bell固定提交的数据副本，下载校验后按公开case编号重建200条子集 |
| I2P数据副本 | https://github.com/chiayi-hsu/Ring-A-Bell | `data/resources.json`包含固定提交URL与SHA-256 |
| NudeNet/Q16参考实现及资源 | https://github.com/CherryBlueberry/DiffZOO | 仅下载三项评估所需文件，固定提交与SHA-256；不运行其搜索算法 |
| SD3.5 Medium | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium | 需自行接受模型条款并获取访问权限 |
| Qwen3Guard | https://huggingface.co/Qwen/Qwen3Guard-Gen-8B | 单独下载至`models/qwen3guard` |
| CLIP | https://huggingface.co/openai/clip-vit-large-patch14 | 单独下载至`models/clip` |
| 输出安全检查器 | https://huggingface.co/CompVis/stable-diffusion-safety-checker | 单独下载至`models/safety-checker` |
| 文本变异服务 | https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF | 按模型说明部署；本仓库附文本接口服务 |

各资源分别适用其上游许可证和使用条款。下载不表示这些资源的许可证相同。
Q16的`prompts.p`使用pickle读取，只使用校验通过的固定资源，不加载不可信替代文件。

本项目代码采用MIT许可证，见LICENSE。第三方模型、数据和评估资源仍遵守其各自上游条款，不因本仓库采用MIT而改变许可证。
