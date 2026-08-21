# Challenge 需求核实(2026-08-19)

> 来源:课程官方作业文档 `Cambridge/818/Homework-Deeptech.pdf`(Deep Tech Series · Lesson 1,第 9 页),与项目页 https://omnifacerig.github.io/ 交叉核对。

## 官方原文(Homework-Deeptech.pdf)

> **Challenge track · +20 marks each · TEAM SIGN-UP CLOSES 19 AUG 2026, 24:00, TEAM LEADER CAN TALK TO TA**
> Intensive research-paper reproduction challenge(form your own team of 5–9 people)
>
> **Deliverables(five items):**
> 1. A complete A100 server project, so anyone can preview and try the whole workflow from a link
> 2. Meet the requirements of https://omnifacerig.github.io/: upload an image, then output a character with a motion skeleton + facial expression animation, as a glb file
> 3. No broken facial expressions, and the teeth and mouth interior must be positioned and animated correctly
> 4. Support the ARKit 52 blendshape set
> 5. Lip-sync for both Chinese and English audio, correctly aligned
>
> **Resources provided:**
> 1. A100 server
> 2. Several high-res T-pose avatar glb models, plus 2D T-pose avatar images
> 3. A walkthrough of parts of the existing solution's code

## 项目页佐证(omnifacerig.github.io)

Showcase 卡片每张展示三列:**input body image → original surface-only GLB → rigged GLB with FACS animation**,与交付物 2 的「上传图 → 输出带运动骨骼+表情动画的 glb」流程一致。

## 结论

- 本地 README(`challenge_omnifacerig/README.md`)记录的 5 项交付物与官方作业文档**逐条一致**,无出入。
- 注意:官方写的是「5–9 人组队」,本人经询问后独自成组(+20 分/人),报名已于 08-19 确认。
- 论文(输入 = 静态 3D mesh)与 challenge(输入 = 图像)的 gap 依然成立,详见 `notes/论文精读.md` §7。
