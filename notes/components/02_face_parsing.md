# 02 · 人脸 landmark / 分割 / 解析组件实地核验报告

> 核验日期：2026-08（据 web_search + GitHub API + HuggingFace API + PyPI 实时数据）
> 目标：复现 OmniFaceRig（arXiv:2606.08043）Stage 1 —— 正视图渲染 → 2D 关键点/分割 → 反投影 3D。
> challenge 硬性需求：图像 → 带骨骼 + 表情动画的 glb，需 **ARKit 52 blendshapes** + 中英口型同步。

---

## 1. face-alignment（pip 包，1adrianb/face-alignment）

- **【结论：可用】**
- **URL**：<https://github.com/1adrianb/face-alignment>
- **landmark 点数**：**68 点**（`TWO_D` 68×2、`TWO_HALF_D`、`THREE_D` 68×3，基于 FAN 网络）。源码 `face_alignment/utils.py` 中 `NUM_LANDMARKS = 68`，`api.py` 的 `LandmarksType` 枚举只有 `TWO_D / TWO_HALF_D / THREE_D` 三种。
  - **98 点（WFLW）不在此包内**——98 点模型在原作者的 Lua 版仓库 `1adrianb/2D-and-3D-face-alignment` 中，pip 包并未提供。核验时「98 点」字段标 **未提供**。
- **license 精确条款**：**BSD-3-Clause**（README badge + GitHub API `license.spdx_id = BSD-3-Clause`）。
- **维护状态**：活跃。`pushed_at = 2026-04-06`，未 archived，7.5k stars。已内置多种人脸检测后端（SFD/BlazeFace/YuNet/RetinaFace/SCRFD）。
- **pip/安装难度**：低。`pip install face-alignment`（PyPI 1.5.0，`requires_python >=3`），依赖 torch + scikit-image。
- **硬件要求**：CPU 可跑；FAN 是轻量 CNN，单张 450×450 上检测+对齐在 M2 上约百毫秒级。可选 CUDA/MPS。
- **权重**：随 pip 包自动下载，模型很小（几十 MB 级）。
- **适用性点评**：FAN 只训过真人脸（300W/LS3D-W），**对动物/风格化角色会失效或漂移**；且 68 点只有轮廓/五官坐标，**没有牙齿/唇等语义细分**。适合做真人类别的「68 点」基准对齐，但不能独立满足口型同步。

---

## 2. MediaPipe Face Landmarker

- **【结论：可用】**
- **URL**：<https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>；源码 <https://github.com/google-ai-edge/mediapipe>
- **API 形态**：**新 Tasks API**（`mediapipe.tasks.vision.FaceLandmarker`）。旧 `mp.solutions.face_mesh` legacy solution 已在 mediapipe 1.0.x 移除/弃用。
- **输出**：**478 个 3D landmark**（FaceMesh-V2）+ **52 个 blendshape 分数**（文档原文「predicts 52 blendshape scores」）。**这 52 个 blendshape 与 ARKit 52 blendshapes 直接对口**，是 challenge「ARKit 52 blendshapes」需求的最省事来源。
- **license 精确条款**：代码 + 预训练模型（face_landmark 等 `.tflite`）均为 **Apache-2.0**；官方文档内容 CC BY 4.0。GitHub issue [#2595](https://github.com/google-ai-edge/mediapipe/issues/2595) 中 Google 成员确认 face mesh 模型按 Apache 2.0 可用于商业用途。
- **⚠️ 3D face mesh 限制（需注意的差异）**：landmark（468/478 点）+ blendshape 本身是 Apache 2.0；但用于 **3D 姿态/网格渲染的 `face_geometry` 模块**（canonical face model / ARCore face mesh、`geometry_pipeline_metadata.binarypb`）**不是 Apache 2.0**，其 canonical 网格与 ARCore 条款绑定、且 metadata 需另行下载。**只取 478 点坐标 + 52 blendshape 时不受此限制；要做 3D 网格投影时需单独评估 ARCore 条款。**
- **pip/安装难度**：低。`pip install mediapipe`（PyPI 1.0.1）。
- **硬件要求**：CPU 实时、移动端优化，无需 GPU。
- **适用性点评**：只训过真人脸，**对动物/风格化角色失效**；但「478 点 + 52 blendshape」是真人/写实类角色表情驱动的最强开源入口。口型同步（中英）需自备 phoneme→blendshape 映射，模型本身只给系数。

---

## 3. RetinaFace / insightface

- **【结论：可用，但 license 不清晰（风险项）】**
- **URL**：insightface <https://github.com/deepinsight/insightface>；serengil 版 retina-face <https://github.com/serengil/retinaface>
- **功能**：人脸检测 + **5 点 landmark**（双眼、鼻尖、两嘴角）。无 68 点、无牙齿/唇细分。
- **license 精确条款**：
  - **insightface 仓库无 LICENSE 文件**（GitHub API `license = null`，`/LICENSE` 返回 404），PyPI `insightface==1.0.1` 的 license 字段也为空。**商用条款不明确 → 标「未确认 / 有风险」**。
  - serengil/retinaface（`pip install retina-face==0.0.18`）为第三方封装，仓库为 **MIT**（PyPI 字段未填，以仓库为准）。
- **维护状态**：insightface 活跃（`pushed_at = 2026-07-27`，29.5k stars）；serengil/retinaface 亦在维护。
- **pip/安装难度**：中。`pip install insightface` 可用，但 RetinaFace 的 ONNX 权重（`det_10g.onnx` ~104MB）需从 Google Drive 手动下载（仓库 `det` 目录流程）。
- **硬件要求**：CPU 可跑（RetinaFace 检测在 M2 上 ~25ms）。
- **适用性点评**：只训过真人脸，**动物/风格化角色失效**；5 点太粗，不能作为 OmniFaceRig 的 68 点来源，只适合做「人脸框检测」的备选（face-alignment 已内置 RetinaFace 后端，可跳过本项）。

---

## 4. Sapiens（Meta，Khirodkar et al.）

- **【结论：可用（但 CC-BY-NC 非商用，课程 challenge 可接受）】**
- **URL**：代码 <https://github.com/facebookresearch/sapiens>；权重 <https://huggingface.co/facebook/sapiens>
- **开源 checkpoint 现状**：seg 头有 **0.3B / 0.6B / 1B**；**无 2B seg**（2B 只有 pretrain）。HF 上为 `facebook/sapiens-seg-{0.3b,0.6b,1b}` 及其 `-bfloat16`（.pt2）变体。
- **任务头**：seg（body-part + face）、pose、depth、normal、pretrain。**没有独立的 19/20 类人脸解析头**——「face」含在 28 类 body-part 分割里。
- **人脸分割类别表（Goliath，28 类）**：`Background, Apparel, Face_Neck, Hair, Left/Right Foot, Hand, Lower_Arm, Lower_Leg, Shoe, Sock, Upper_Arm, Upper_Leg, Lower_Clothing, Torso, Upper_Clothing, Lower_Lip, Upper_Lip, Lower_Teeth, Upper_Teeth, Tongue`（原始 34 类去掉 Eyeglass_Frame/Eyeglass_Lenses/Visible_Badge/Chair/Lower_Spandex/Headset）。
  - ✅ **有上下唇（Lower_Lip / Upper_Lip）、上下牙（Lower_Teeth / Upper_Teeth）、舌头（Tongue）细分** —— 这是少数开源提供**口腔细分类**的模型，对口型同步极有价值。
  - ❌ **无 eye / brow 细分**：眼/眉不单独标注，整脸只给 `Face_Neck`。
- **license 精确条款**：**CC-BY-NC-4.0**（代码与权重均为 CC-BY-NC-4.0，仓库根 `LICENSE` 即 CC BY-NC 4.0 全文）。**非商用许可**；课程/challenge 非商用场景可用，商用需替换。
- **权重下载方式与大小**：HF 直接下载（无 gating）。`sapiens-seg-0.3b` 的 `.pth` 约 **1.36 GB**（fp32），`-bfloat16` `.pt2` 版本用于 Sapiens-Lite（torchscript/bfloat16，快约 4×）。
- **维护状态**：代码库活跃（`pushed_at = 2026-05-26`，5.4k stars）。
- **pip/安装难度**：高。无 PyPI 包，需 `git clone` + conda 装 mmcv/mmseg 全家桶（或用轻量的 `sapiens_lite`：仅 pytorch>=2.2 + opencv + tqdm + json-tricks）。
- **硬件要求**：0.3B（bfloat16 ~1.4GB 权重）推理约需 **2–4 GB 显存**，可上消费级 GPU；1B 需更大（~8 GB+）。官方未给精确显存数字 → 标「近似，未官方确认」。
- **适用性点评**：训练数据为真人（Goliath），**动物/风格化角色会失效**；但**牙齿/唇/舌细分独一份**，是口型同步的首选语义来源。缺眼/眉细分，需与 BiSeNet 或自训模型互补。

---

## 5. SAM 2（Meta）

- **【结论：可用】**
- **URL**：<https://github.com/facebookresearch/sam2>
- **开源现状**：代码 + 权重 + 训练代码全开源，无需 gating。
- **license 精确条款**：**Apache-2.0**（README 明确「checkpoints、demo code、training code 均为 Apache 2.0」；GitHub API `license = Apache-2.0`）。
- **权重**：`sam2.1_hiera_tiny/small/base_plus/large` 四档，`.pt` 从 ~39MB（tiny）到 ~897MB（large），HF 直接下载。
- **pip/安装难度**：中。无官方 PyPI 包，需 `git clone` + `pip install -e .`。（PyPI 上的 `segment-anything-2==0.0.1` 是第三方 MIT 封装，非官方。）
- **点/框 prompt 用法**：`Sam2ImagePredictor` → `set_image()` → `set_point_prompt()` / `add_box()`（`predictor.predict(point_coords=..., point_labels=..., box=...)`）。
- **硬件要求**：tiny 可 CPU/低显存；large 建议 GPU。
- **适用性点评**：**通用分割兜底**，对动物/风格化角色也能框出对象，但**零语义**——不能直接给「嘴唇/牙齿」标签，只能当「前景掩码 / 物体级兜底」。适合 OmniFaceRig 里「分割模型兜底」一环，不能替代人脸解析。

---

## 6. SAM 3（Meta，arXiv:2511.16719）

- **【结论：可用（已开源，但 gated + 重依赖）】**
- **URL**：论文 <https://arxiv.org/abs/2511.16719>；代码 <https://github.com/facebookresearch/sam3>
- **开源现状（截至 2026-08）**：**已开源代码 + 权重**。GitHub `facebookresearch/sam3`（11.4k stars，`pushed_at = 2026-08-14`）；权重在 HF `facebook/sam3`，**gated = manual（需申请访问并通过）**。另 **SAM 3.1（Object Multiplex）已于 2026-03-27 发布**，权重在 `facebook/sam3.1`（同样 gated）。**不是「仅有论文」。**
- **license 精确条款**：自定义 **「SAM License」**（Last Updated 2025-11-19，GitHub API `license = NOASSERTION`）。要点：非独占、全球、**免版税、可商用**，可分发/修改/做衍生；限制为禁止逆向工程、须遵守出口管制/制裁/ITAR、禁止军事/核/间谍用途、加州法管辖。**非 OSI 标准许可证，但允许商用。**
- **权重下载方式与大小**：HF gated 下载。`facebook/sam3`：`model.safetensors` **3.44 GB** + `sam3.pt` **3.45 GB**。
- **维护状态**：活跃（2026-08 仍有提交）。
- **pip/安装难度**：高。需 `git clone` + `pip install -e .`；硬性依赖 **Python 3.12+、PyTorch 2.7+、CUDA 12.6+ GPU**，可选 flash-attn-3。
- **硬件要求**：需 CUDA GPU；~3.4GB 权重 → 至少 **8–12 GB 显存**（近似）。
- **适用性点评**：支持**文本（概念）/点/框/掩码 prompt**，可用 `"teeth"`、`"upper lip"` 这类文本概念直接分割，理论上对风格化角色比纯点 prompt 更泛化；但非人脸专用、对精细口腔边缘可能不如 Sapiens 稳定，且 gated + 重依赖是使用门槛。课程场景建议「有卡再用」。

---

## 7. BiSeNet face parsing（可选，快速核实）

- **【结论：可用（可选）】**
- **URL**：zllrunning/face-parsing.PyTorch <https://github.com/zllrunning/face-parsing.PyTorch>（经典实现）；yakhyo/face-parsing <https://github.com/yakhyo/face-parsing>（新封装，`pushed_at=2026-04-14`）。
- **license 精确条款**：**MIT**（两者均 MIT）。
- **类别**：19 类（CelebAMask-HQ）：`skin, nose, eye_glass, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip, l_lip, hair, hat, ear_r, neck_l, neck, cloth, bg`。
  - ✅ 有 **左/右眼（l_eye/r_eye）、眉、上下唇（u_lip/l_lip）、嘴（mouth）**。
  - ❌ **无牙齿细分**。
- **权重**：`79999_iter.pth` ~330MB，Google Drive 手动下载；无官方 pip。
- **维护状态**：zllrunning 版 2023-05 后停更；yakhyo 版仍在维护。
- **硬件要求**：CPU 可跑（ResNet18 backbone）。
- **适用性点评**：只训真人脸（CelebAMask-HQ），**动物/风格化失效**；与 Sapiens 互补——Sapiens 有牙无眼，BiSeNet 有眼无牙，二者都不是「眼+牙+唇全细分」的单一方案。

---

## 8. 加分项：stylized / anime 角色 landmark

- **【结论：部分可用（无官方代码）】**
- **StylizedFacePoint（Cheng et al., ACM MM 2024）**：论文公开（<https://openreview.net/pdf?id=J3mF5Ea5JG>；<https://dl.acm.org/doi/10.1145/3664647.3680984>），但**未检索到公开代码/权重 → 标「未开源，仅有论文」**。
- 社区替代：<https://github.com/ayutaz/anime-face-detector>（基于 mmdet + mmpose 的动漫人脸检测/landmark，非官方、质量未核验）。
- **适用性点评**：风格化/动画角色的 landmark 目前无成熟官方开源；最现实路径是**自训一个小 landmark/解析模型**（用 AnimeFace 类数据 + 伪标注蒸馏）。

---

## 推荐组合（仅开源组件）

OmniFaceRig 论文原组合 = **68 点 landmark + Sapiens + SAM3 + 微调 Sapiens** 的四模型 ensemble。在只有开源组件、且要兼顾 challenge（ARKit 52 blendshape + 口型）的约束下，建议：

1. **2D landmark**：**MediaPipe Face Landmarker（478 点 + 52 blendshape）为主**（直接满足 ARKit 52 blendshape），叠加 **face-alignment 68 点**作为论文同款基准与反投影锚点。
2. **人脸解析（口腔/唇）**：**Sapiens seg-0.3b（有唇/牙/舌）** —— 唯一开源口腔细分来源，做口型同步语义；**注意 CC-BY-NC**（课程非商用 OK）。
3. **眼/眉细分补充**：**BiSeNet（MIT，19 类）**，补 Sapiens 缺的眼睛/眉毛。
4. **通用兜底**：**SAM 2（Apache 2.0，免 gating、轻量）** 做前景/物体级掩码兜底；有 GPU 且需要文本概念分割时升级到 **SAM 3（gated）**。

**最弱一环**：**「风格化/动物角色的细粒度人脸解析」**。所有 landmark 与解析模型（FAN/MediaPipe/Sapiens/BiSeNet）都只训过真人脸，对风格化/动物角色必然失效；且**没有任何单一开源模型同时提供「眼 + 牙 + 唇」细分**（Sapiens 有牙无眼、BiSeNet 有眼无牙）。因此**需要自训一个小解析模型**（如 DeepLabV3+/轻量 BiSeNet/UNet，类别表= skin/eye/eyebrow/upper_lip/lower_lip/upper_teeth/lower_teeth/tongue/…，用 Sapiens+BiSeNet 伪标注蒸馏 + 风格化数据微调）。**口型同步的 phoneme→blendshape 映射也需自备**（中英各一套），现有模型只输出系数不输出音素。

### 风险/未确认清单
- insightface 无 LICENSE → 商用条款**未确认**。
- Sapiens 显存数字为近似，官方未给精确值 → **未确认**。
- face-alignment 的 98 点模型 → **此 pip 包未提供**（在 Lua 原仓库）。
- StylizedFacePoint 官方代码 → **未开源**。
