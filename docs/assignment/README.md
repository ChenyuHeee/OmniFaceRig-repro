# docs/assignment · 官方作业与 challenge 要求原文

> 存放课程官方发放的、与 challenge track 相关的原始文件,供接手 agent 直接查阅,无需再从其他目录找。

## 文件清单

| 文件 | 说明 |
|---|---|
| `Homework-Deeptech.pdf` | **官方作业文档**(Deep Tech Series · Lesson 1)。**Challenge track 的 5 项交付物原文在第 9 页**,包含:交付物 1-5、资源清单(A100 / T-pose glb + 2D 图 / 代码走读)、组队与报名规则(5–9 人,+20 分/人,08-19 24:00 截止)。这是需求的**权威出处**。 |
| `Lesson1-Cambridge-Phenomenon-slides.pptx` | Lesson 1 幻灯片(原名「DECIPHERING THE CAMBRIDGE PHENOMENON ZJU LCC 2026.pptx」)。其中 slide 30/38/63/64 含 challenge track / OmniFaceRig / blendshape / ARKit 相关内容。 |
| `challenge原始README.md` | 报名时在课程仓库 `challenge_omnifacerig/` 建的原始项目 README(任务、5 项交付物、官方资源、时间线)。 |

## 需求原文核对结论

- 5 项交付物已逐条与 Homework-Deeptech.pdf 核对,与本仓库 `notes/challenge需求核实.md` 中记录的完全一致。
- 需求要点(原文,第 9 页):
  1. A complete A100 server project, so anyone can preview and try the whole workflow from a link
  2. Meet the requirements of https://omnifacerig.github.io/: upload an image, then output a character with a motion skeleton + facial expression animation, as a glb file
  3. No broken facial expressions, and the teeth and mouth interior must be positioned and animated correctly
  4. Support the ARKit 52 blendshape set
  5. Lip-sync for both Chinese and English audio, correctly aligned
- 注意:官方写「form your own team of 5–9 people」,本人经询问后**独自成组**参赛(已报名确认)。
