# Challenge Track · OmniFaceRig 复现项目

> 形式：**个人挑战**（经询问后独自成组）· +20 分
> 报名截止：**2026-08-19 24:00**（组长联系 TA）——已询问，待最终确认
> 项目根目录：`challenge_omnifacerig/`

## 任务

复现论文 **OmniFaceRig**（Meta Reality Labs，SIGGRAPH Asia / TOG 2026，arXiv:2606.08043）：
上传一张图像 → 输出带**运动骨骼 + 面部表情动画**的角色（glb 文件）。

## 交付物（5 项，全部必须）

1. A100 服务器上的**完整可预览项目**（链接可跑通全流程）
2. 满足 omnifacerig.github.io 要求：图 → 带骨骼 + 表情的 glb
3. 面部表情**无破损**；牙齿与口腔内部**定位与动画正确**
4. 支持 **ARKit 52 blendshapes** 全集
5. **中、英文**音频口型同步，对齐正确

## 官方提供的资源

- A100 服务器
- 多套高精度 T-pose 角色 glb 模型 + 2D T-pose 角色图
- 现有方案的部分代码走读（walkthrough）

## 关键链接

- 项目页：https://omnifacerig.github.io/
- 论文：arXiv:2606.08043（已下载至 `paper/OmniFaceRig_arXiv_2606.08043.pdf`）
- 数据集：论文页标注 Coming Soon（留意开放）

## 目录结构

```
challenge_omnifacerig/
├── paper/      # 论文 PDF 与精读笔记
├── notes/      # 调研、实验记录、TODO
├── code/       # 复现代码
├── data/       # glb 模型、输入图、测试集
└── outputs/    # 生成的 glb 与演示视频
```

## 与 Deep Tech 课程其他任务的时间线

| 截止 | 事项 |
|---|---|
| 08-19 24:00 | 报名（组长联系 TA） |
| 08-23 23:59 | 中期个人作业（梦想职业游戏） |
| 08-26 23:59 | 团队 final（50 分） |
| 08-27 | 汇报日 + Formal Hall |
| challenge 截止 | 以 TA 通知为准（预计与课程同步或更早） |

## 进展记录

见 `进度.md`。
