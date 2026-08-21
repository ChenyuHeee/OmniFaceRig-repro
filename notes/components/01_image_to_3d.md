# 单图 → 3D 角色网格(带贴图)前端候选方案调研

> 核验日期:2026-08-21。核验方式:GitHub REST API(`api.github.com/repos/...` / `search/repositories`)、HuggingFace API(`huggingface.co/api/models/...`、`/tree/main?recursive=true`)、`curl` 直连官方页面与文档、`web_search` 交叉核验。
> 所有 stars / license / 下载量 / 权重大小均为现场 API 返回值,未做凭记忆估计;核验不到的字段明确标注「未确认」。

---

## 1. SF3D(Stable Fast 3D,Stability AI)

- 【结论:**部分可用**】代码与权重均开源,但权重在 HF 上 **gated**(需登录 HF 并勾选同意条款、填联系信息才能下载),且 license 带商业限制;仓库已停更约一年半。
- 仓库:`https://github.com/Stability-AI/stable-fast-3d`(⭐ 1788,`license: other`/NOASSERTION,`pushed_at 2025-01-22`)
- 权重:HuggingFace [`stabilityai/stable-fast-3d`](https://huggingface.co/stabilityai/stable-fast-3d) — `gated: auto`(需同意),`downloads ≈ 83.6k`,最后更新 `2025-04-08`。核心文件 `model.safetensors` **约 4.02 GB**。
- license:**Stability AI Community License**(HF `license_name: stabilityai-ai-community`)。条款要点(据其 [LICENSE.md](https://raw.githubusercontent.com/Stability-AI/stable-fast-3d/main/LICENSE.md)):研究/非商业免费;商业使用免费但要求实体**年收入 < $1,000,000**,且商用须在 `stability.ai/community-license` 注册;超过阈值须单独向 Stability 申请 enterprise 授权。
- 维护状态:代码 `pushed_at 2025-01-22`,基本停更;Stability 已转向后继模型 **SPAR3D**(HF [`stabilityai/stable-point-aware-3d`](https://huggingface.co/stabilityai/stable-point-aware-3d),同样 gated + 社区 license)。
- 硬件:README 明确单图默认设置约 **6 GB VRAM**(也支持 CPU / Apple MPS)。
- 输出:**GLB**,带 **PBR 贴图**(albedo/base color + metallic + roughness + normal)与 UV 展开(illumination disentanglement);支持 `--remesh_option` 三角/四边重网格化。
- 适用性点评:输出形态正好契合我们需求(PBR GLB),轻量、快;但 **权重 gated + 社区 license 商业限制**是摩擦点,且代码停更。作为课程 research 用途可用,商用/分发需注意。

---

## 2. TRELLIS / TRELLIS.2(Microsoft)

- 【结论:**可用**(v1 与 v2 均完全开源、MIT、无 gating,质量当前最强)】
- v1 仓库:`https://github.com/microsoft/TRELLIS`(⭐ 13472,MIT,`pushed_at 2026-06-26`,仍在更新)
- v2 仓库:[`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2)(⭐ 10735,MIT,`pushed_at 2026-07-10`,活跃) — "Native and Compact Structured Latents for 3D Generation"
- 权重:
  - v1:[`microsoft/TRELLIS-image-large`](https://huggingface.co/microsoft/TRELLIS-image-large)(原 `JeffreyXiang/TRELLIS-image-large` 已重定向到 microsoft org)— MIT、ungated、`downloads ≈ 194 万`、总计 **约 3.3 GB**。
  - v2:[`microsoft/TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B)— MIT、ungated、`downloads ≈ 149 万`、总计 **约 16.24 GB**(4B 参数,多组 flow/Vae checkpoint)。
- license:**MIT**(模型 + 绝大部分代码;README 注明子模块 `diffoctreerast`、`FlexiCubes` 有各自 license,不影响使用)。
- 维护状态:非常活跃(v1 `2026-06-26`、v2 `2026-07-10`),v2 为主推当前版本。
- 硬件:v1 README 明确「**至少 16 GB** NVIDIA GPU(A100/A6000 验证)」;v2 明确「**至少 24 GB**(A100/H100 验证)」。
- 输出:**GLB** 带贴图网格;v2 直接输出 **PBR-ready** 材质(`texture_size` 最高 4096,支持 `extension_webp`),还支持 Gaussian/Radiance field 等格式。
- 适用性点评:**首选候选**——license 最宽松(MIT,可商用)、无 gating、活跃维护、直接给 PBR GLB、质量 SOTA。风险:(a) 是通用物体生成模型,非角色/人脸专用,面部与口腔细节(OmniFaceRig 依赖 inner-mouth)可能欠佳,或需配合重贴图/角色专用后处理;(b) v2 需 24 GB 显存,显存不足退而用 v1(16 GB)。

---

## 3. TripoSR(Stability / Tripo)

- 【结论:**可用**(最轻量、MIT,但质量与纹理能力有限)】
- 仓库:`https://github.com/VAST-AI-Research/TripoSR`(⭐ 6867,MIT,`pushed_at 2026-06-04`)
- 权重:[`stabilityai/TripoSR`](https://huggingface.co/stabilityai/TripoSR)(MIT、ungated、`downloads ≈ 18.6 万`)— `model.ckpt` **约 1.68 GB**。
- license:**MIT**。
- 维护状态:社区较活跃(代码 `2026-06-04`),但模型本身是 2024 年产物。
- 硬件:README 明确单图默认约 **6 GB VRAM**(A100 上 <0.5s)。
- 输出:**OBJ/GLB** 网格;默认 **顶点色**,`--bake-texture` 才输出贴图(仅 diffuse,无 PBR、无 normal/metallic)。
- 适用性点评:轻量、MIT、低显存,适合 pipeline 冒烟测试;但**无 PBR、纹理简单、几何质量低**,不足以作为最终前端。

---

## 4. Hunyuan3D-2 / 2.1(腾讯)

- 【结论:**部分可用**(开源、质量高,但 **non-commercial license** 是硬约束)】
- 仓库:[`Tencent-Hunyuan/Hunyuan3D-2`](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)(⭐ 14531,`license: other`/NOASSERTION,`pushed_at 2025-10-28`);2.1:[`Tencent-Hunyuan/Hunyuan3D-2.1`](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)(⭐ 3871,含 PBR 模型与训练代码)
- 权重:[`tencent/Hunyuan3D-2`](https://huggingface.co/tencent/Hunyuan3D-2)(ungated、`downloads ≈ 12.3 万`,全仓库含多版本约 **74.89 GB**,实际取 shape + texture 两套 checkpoint 约 5+7 GB);2.1:[`tencent/Hunyuan3D-2.1`](https://huggingface.co/tencent/Hunyuan3D-2.1)(ungated,约 **14.91 GB**,含 `hunyuan3d-paintpbr-v2-1` PBR 纹理模型)
- license:**Tencent Hunyuan Community License**(`tencent-hunyuan-community`)= **非商用**(non-commercial)。2.1 虽 "fully open-sourced + PBR + 训练代码",license 仍是同一社区非商用条款(已核验 HF `license_name: tencent-hunyuan-community`)。
- 维护状态:代码 `2025-10` 后基本停更(2.0),2.1 于 `2025-06` 发布。
- 硬件:README 明确 **6 GB(仅形状)/ 16 GB(形状+纹理)** VRAM。
- 输出:**glb/obj** + 纹理;2.1 输出 **PBR**(base color/roughness/metallic/normal)。
- 适用性点评:角色与纹理/PBR 质量好、显存适中(16 GB);但 **non-commercial license** 意味着只能用于课程/学术研究,不能商用或公开分发——若 challenge 无商用要求则可用作质量导向备选。

---

## 5. InstantMesh(TencentARC,可选)

- 【结论:**可用**(较老,Apache-2.0)】
- 仓库:`https://github.com/TencentARC/InstantMesh`(⭐ 4502,**Apache-2.0**,`pushed_at 2025-01-03`)
- 权重:[`TencentARC/InstantMesh`](https://huggingface.co/TencentARC/InstantMesh)(Apache-2.0,ungated;提供 4 个稀疏视图重建变体 + Zero123++ UNet)
- license:**Apache-2.0**。
- 维护状态:代码 `2025-01` 后停更,较老。
- 硬件:README 未给出单卡明确数值(「未确认」);支持双卡跑 gradio 省显存。
- 输出:**OBJ**(默认顶点色),`--export_texmap` 才出纹理贴图(UV 展开较慢),无 PBR。
- 适用性点评:可作备选;质量与纹理能力弱于 TRELLIS/Hunyuan,且停更,优先级低。

---

## 6. 商业 API:Meshy / Tripo

### 6.1 Meshy API
- 【结论:**商业 API**(可用,有免费额度,自带 auto-rig/骨骼/动画)】
- 页面:定价 [`meshy.ai/pricing`](https://www.meshy.ai/pricing),文档 [`docs.meshy.ai`](https://docs.meshy.ai)
- 免费额度:有 Free 计划,约 **200 credits/月**(官方定价页为 JS 渲染,数值未能逐字抓取;多个 2026 第三方来源一致报 200,标注「约」)。付费 Pro/Studio/Enterprise 起价约 $20/月(第三方来源,未逐字核验)。
- 输出:支持 **GLB / FBX / OBJ / STL / USDZ / BLEND / 3MF**;image-to-3d 直接给**带贴图(含 PBR)的 glb**,并可 remesh / retexture / unwrap UV。
- 自动骨骼:文档明确有 **「Rigging & Animation API」**(`Auto-rig character`)与 **Animation(自动骨骼+动画)** 能力 —— 能直接产出带骨骼、可动的角色,是 challenge「带运动骨骼的 glb」的潜在捷径(但**面部表情动画仍需 OmniFaceRig 复现**)。

### 6.2 Tripo API
- 【结论:**商业 API**(可用,有免费额度但 **Free 限非商用**,有 auto-rig)】
- 页面:定价 [`tripo3d.ai/pricing`](https://www.tripo3d.ai/pricing)(已逐字抓到),文档 [`developers.tripo3d.ai`](https://developers.tripo3d.ai)
- 免费额度:**Free $0/月,200 credits ≈ 13 模型**,但明确标注 **Non-Commercial Use + Public Models + Standard 质量**(不可私有、不可商用、非高清)。Pro $19.90/月(≈200 模型,可商用、私有、HD)。
- 输出:可拿**带纹理 glb**;有 **auto-rig** 端点(文档 [`animations-rig`](https://developers.tripo3d.ai/docs/animations-rig))。
- 适用性点评(两者共同):如果目标是「先跑通单图→带纹理/带骨骼 glb」而非全开源复现,商业 API 是最快路径,且自带 auto-rig 可作对照;但都**不能替代 OmniFaceRig 的面部表情 rigging**,且 Free 档有「非商用」限制(课程研究用途通常可接受)。

---

## 7. OmniFaceRig 官方代码与 Dataset 核验

- 【结论:**代码未公开;Dataset 宣称已 release 但无实际下载入口**】

### 7.1 官方代码
- GitHub 搜索 `OmniFaceRig`(按 stars 排序)仅返回 **1 个**仓库:[`omnifacerig/omnifacerig.github.io`](https://github.com/omnifacerig/omnifacerig.github.io)(⭐ 4,无 license,`pushed_at 2026-07-19`,即项目官网源码)。
- **未发现任何官方实现代码仓库**(无模型推理/训练代码)。HuggingFace 上搜索 `OmniFaceRig` 的 model/space/dataset 均为空。
- 判定:**官方代码未发布**;是否会在未来公开为「未确认」。

### 7.2 Dataset(Omni-Bench)
- 官网 [`omnifacerig.github.io`](https://omnifacerig.github.io/) 有「Dataset」章节,描述其数据集 **Omni-Bench**:1,000 个 biped 3D 角色(500 人类/类人 + 500 动物),带 FACS blendshapes(最高 155 个,含牙/牙龈/舌)与 inner-mouth 几何,并含完整生成管线元数据(text prompt + 中间 2D 参考图 + 最终 mesh)。
- 页面文字宣称「**we release Omni-Bench, ... first open-source benchmark**」,**但页面 HTML 中没有任何外部下载链接**(经抓取全部 `href`:仅 `#dataset` 锚点、`arxiv.org/abs/2606.08043`、作者主页;无 HuggingFace / Google Drive / GitHub 数据仓库链接)。
- 判定:**Dataset 尚无可用下载入口**(未实际可下载,接近「Coming Soon / 待发布」状态)。
- ⚠️ 注意区分:HF 上存在多个名为 `Omni-Bench`/`OmniBench` 的**无关**数据集(如 [`ModalityDance/Omni-Bench`](https://huggingface.co/datasets/ModalityDance/Omni-Bench) 是多模态 VQA benchmark,`arxiv:2601.09536`,仅 43 下载量),**与 OmniFaceRig 无关,勿混淆**。
- 对复现的影响:无法取得作者基准角色,故须**自备「单图→角色 mesh」前端生成输入**(这正是本调研的动机),也无法用官方数据做对齐评测。

---

## 推荐组合

**首选前端:TRELLIS / TRELLIS.2(Microsoft)**
- 理由:MIT(完全宽松、可商用)、权重 ungated、活跃维护(v2 最近 commit `2026-07-10`)、直接输出 **PBR 贴图 GLB**、质量当前最强;与「静态 3D mesh 输入」的 OmniFaceRig 无缝衔接。
- 风险:通用物体生成模型,**角色/人脸与口腔细节**不是其专长(而 OmniFaceRig 依赖 inner-mouth 几何),可能需要二次重贴图或角色专用微调;v2 需 **24 GB** 显存,显存不足用 v1(16 GB,权重仅 3.3 GB)。

**质量导向备选:Hunyuan3D-2.1(腾讯)**
- 角色与 PBR 纹理质量好、16 GB 显存;但 **non-commercial license**,仅当 challenge 允许研究用途且不商用/不分发时选用。

**轻量兜底:TripoSR**
- MIT、6 GB、亚秒级,只用于 pipeline 冒烟测试,质量不足以做最终前端。

**商业 API 捷径/对照:Meshy API(优先)或 Tripo API**
- 若目标是快速跑通「单图→带纹理/带骨骼 glb」而非全开源复现:Meshy 免费额度 + 自带 **auto-rig/骨骼/动画** + glb 输出,是最省事路径;Tripo Free 200 credits(非商用)亦可。二者都不能替代 OmniFaceRig 的面部表情 rigging,只作为前端或输出对照。

> 核心提醒:OmniFaceRig 的**代码与数据集均未发布**,因此「自备角色 mesh 前端」是复现的必要前置;最终 glb 的「面部表情动画」仍需依赖 OmniFaceRig 方法本身,前端只需提供**干净、带贴图(尽量 PBR)、尽量 T-pose 的角色 mesh**。
