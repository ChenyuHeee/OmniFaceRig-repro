# OmniFaceRig-repro

> 论文 **OmniFaceRig**(Meta Reality Labs,SIGGRAPH Asia / TOG 2026,arXiv:2606.08043)的学习与复现仓库。
> 任务:上传一张图像 → 输出带**运动骨骼 + 面部表情动画**的角色(glb 文件)。

- 项目页:https://omnifacerig.github.io/
- 论文:arXiv:2606.08043(PDF 已入库,在 `paper/`)
- 官方作业原文:见 `docs/assignment/`(Homework-Deeptech.pdf 第 9 页为交付物出处)
- 状态:课程 challenge(个人)· 私有仓库

## 交付物

1. A100 服务器上的**完整可预览项目**(链接可跑通全流程)
2. 满足 omnifacerig.github.io 要求:图 → 带骨骼 + 表情的 glb
3. 面部表情**无破损**;牙齿与口腔内部**定位与动画正确**
4. 支持 **ARKit 52 blendshapes** 全集
5. **中、英文**音频口型同步,对齐正确

## 仓库结构

```
OmniFaceRig-repro/
├── paper/           # 论文 PDF + 提取全文 + 精读笔记
├── notes/           # 进度、需求核实、组件调研(components/)
├── docs/assignment/ # 官方作业文档与 challenge 要求原文
├── code/            # 复现代码
├── data/            # glb 模型、输入图、测试集(gitignore)
└── outputs/         # 生成的 glb 与演示视频(gitignore)
```

## 时间线

| 截止 | 事项 |
|---|---|
| 08-19 24:00 | 报名(已确认 ✅) |
| 08-23 23:59 | 中期个人作业(梦想职业游戏) |
| 08-26 23:59 | 团队 final(50 分) |
| 08-27 | 汇报日 + Formal Hall |
| challenge 截止 | 以 TA 通知为准 |

## 进展

见 `notes/进度.md`。
