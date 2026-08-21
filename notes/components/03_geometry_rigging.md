# OmniFaceRig 复现 — Stage 2 几何/绑定组件实地核验报告

> 调研日期：2026-08-21（今天是 2026 年 8 月，本报告基于当日实测）
> 方法：`web_search` + `curl` GitHub API / 官方文档 / PyPI / conda-forge / Anaconda API。
> 说明：所有 license、URL、版本号均为当次 API 实测值；无法核实处标「未确认」，不编造。

---

## 0. 结论速览（TL;DR）

| 组件 | 结论 | License | 是否影响 A100 部署 |
|---|---|---|---|
| libigl（ARAP / SDF / boolean / convex hull） | 部分可用 | 核心 MPL-2.0；`copyleft/`（boolean、convex_hull、CGAL-SDF）为 GPL-3.0 | ⚠️ boolean/hull 会引入 GPL |
| trimesh | 可用 | MIT（boolean 后端 manifold3d=Apache-2.0） | ✅ 安全 |
| xatlas / xatlas-python | 可用 | MIT | ✅ 安全 |
| CGAL | 不可用（license 风险 + 安装重） | GPL-3.0+/LGPL-3.0+；swig bindings GPL-3.0 | ❌ 避免 |
| Deformation Transfer 实现 | 可用（自写最稳） | 各仓 MIT/Apache/无 license | ✅ 选 MIT/Apache 的即可 |
| Delta Mush / Point Deform | 需自写 | 无独立 Python 库 | ✅ 自写 ~50 行 |
| FLAME 模板 | 需申请（免费注册，CC-BY 许可） | CC-BY 4.0（署名） | ✅ 可商用（需署名+引用） |
| ICT FaceKit | 可用 | MIT（含牙齿/舌头/眼球） | ✅ 安全 |
| ARKit 52 blendshapes | 可用（硬编码名单） | 官方文档公开 | ✅ 无风险 |
| glTF 导出（morph+skinning） | 可用 | pygltflib=MIT；bpy=GPL | ⚠️ bpy 是 GPL，pygltflib 更干净 |
| RigAnyFace | 未公开代码 | 论文 NeurIPS 2025 | 未确认 |

**核心 license 风险点**：布尔（mesh_boolean）与凸包（convex_hull）这两个 Stage 2 必需操作，在 libigl 里位于 `copyleft/cgal/`，是 **GPL-3.0**；`pip install libigl` 的官方 Python 包也整体是 **GPL-3.0**。CGAL 的 Python 绑定同样是 GPL。因此推荐用 **trimesh + manifold3d（MIT/Apache-2.0）** 做布尔/凸包/SDF，**避开 libigl 的 copyleft 部分和 CGAL**。

---

## 1. libigl

- **结论：部分可用**（ARAP、SDF 在核心层可用且为 MPL-2.0；但布尔/凸包只在 `copyleft` 层，为 GPL-3.0）
- **URL**：https://github.com/libigl/libigl
- **License（精确）**：双许可证。核心库（`include/igl/**`）每个文件头声明 **MPL-2.0**（实测 `signed_distance.h` 头文件即 "Mozilla Public License v. 2.0"）；`include/igl/copyleft/` 子目录（`cgal/`、`tetgen/`、`opengl2/`）为 **GPL-3.0**（仓库根有 `LICENSE.MPL2` 与 `LICENSE.GPL` 两份文件；GitHub API 据此上报主 license 为 GPL-3.0）。
- **功能核实（当日 git tree 实测）**：
  - ARAP：`include/igl/arap.h`、`arap_linear_block` 等 → **核心层，MPL-2.0** ✅
  - SDF：`include/igl/signed_distance.h/.cpp` → **核心层，MPL-2.0**（支持 pseudo-normal / winding number / fast winding number 三种符号化）✅
  - 布尔：`include/igl/copyleft/cgal/mesh_boolean.{h,cpp}` → **copyleft/cgal，GPL-3.0** ⚠️（核心层无布尔）
  - 凸包：`include/igl/copyleft/cgal/convex_hull.{h,cpp}`（另有 `outer_hull`）→ **copyleft/cgal，GPL-3.0** ⚠️（核心层无凸包）
  - 碰撞/点-实体符号距离：`copyleft/cgal/point_solid_signed_squared_distance` → GPL-3.0 ⚠️
- **Python 安装方式**：
  - 官方绑定仓库 `libigl/libigl-python-bindings`（nanobind 实现，371★，active），`pip install libigl`（PyPI 实测 v2.6.2）。⚠️ 该绑定仓库 license 为 **GPL-3.0**（因为它把含 GPL 的 copyleft 部分也一起打包编译）。
  - 第三方 `compas_libigl`（COMPAS 生态，文档 https://compas.dev/compas_libigl/latest/）。
- **维护状态**：活跃（pushed_at 2026-08-20，5073★）。
- **适用性点评**：ARAP 与 SDF 可以直接用 MPL-2.0 核心；但布尔/凸包走 libigl 会踩 GPL，对我们 A100 服务器内部部署是传染性风险，建议这两个操作改用 trimesh/manifold3d。若一定要用 libigl，应只自编译 MPL-2.0 子集并用自写 pybind11/nanobind 包装，避开 `copyleft` 头文件。

---

## 2. trimesh

- **结论：可用**（几何操作全覆盖 + glb 静态导入导出；但 **morph targets / skinning 导出不支持**）
- **URL**：https://github.com/mikedh/trimesh
- **License**：**MIT**（实测 PyPI 元数据）。布尔后端 manifold3d 为 **Apache-2.0**（`elalish/manifold`，2240★，active）；凸包用 scipy（BSD）。
- **pip 可用性**：`pip install trimesh`（PyPI 实测 v5.0.0，requires Python ≥3.10）。
- **功能核实**：
  - 布尔：`trimesh.boolean`，后端可选 `manifold` 或 `blender`（实测 `boolean.py`，默认走 manifold3d）✅
  - 凸包：`trimesh/convex.py`，基于 `scipy.spatial.ConvexHull` ✅
  - SDF：`trimesh/proximity.py` 有 `signed_distance(mesh, points)`（scipy cKDTree + 符号判定）✅
  - 最近点焊接：`trimesh.base.merge_vertices()` ✅
  - glb 导入：`load_glb/load_gltf`（仅 glTF 2.0，源码明示 "only GLTF 2 is supported"）✅
  - glb 导出：`export_glb/export_gltf` ✅，但当日 grep 整个 `exchange/gltf/` 包，**无任何 morph/target/skin/joint/weight 处理** → **不支持 morph targets 与 skinning**（只能导静态几何 + 材质 + UV）❌
- **维护状态**：活跃（pushed_at 2026-08-20，3655★）。
- **适用性点评**：Stage 2 的布尔/凸包/SDF/焊接/静态网格清洗全用 trimesh 一把梭（MIT 安全）；最终带 morph+skinning 的 glb 导出**不要**交给 trimesh，改用 pygltflib 或 bpy。

---

## 3. xatlas（UV 重打包）

- **结论：可用**（独立 MIT 库 + 现成 Python 绑定）
- **URL**：C++ 库 https://github.com/jpcy/xatlas ；Python 绑定 https://github.com/mworchel/xatlas-python
- **License**：xatlas 本体 **MIT**（README badge 明示）；xatlas-python **MIT**。
- **能力**：xatlas 是 thekla_atlas 的独立 fork，C++11、无外部依赖，生成适合烘焙/绘制的 UV 坐标（chart 分片 + 打包）。xatlas-python 提供 `xatlas.parametrize()` 单网格、`xatlas.Atlas()` 多网格合并图集（`ChartOptions`/`PackOptions`）。
- **Python 安装方式**：`pip install xatlas`（xatlas-python，209★，2025-07 更新）。
- **维护状态**：xatlas 本体 last push 2024-06、无正式 release（成熟/冻结，2545★）；xatlas-python 较新（2025-07）。
- **备选注意**：nvdiffrast 也内嵌 xatlas，但 nvdiffrast 是 **NVIDIA Source Code License（非商用/研究评估）**，实测其 `LICENSE.txt` 3.3 条款禁止商用 → 我们若用它拿 UV 会被非商用条款卡住，**改用 MIT 的 xatlas-python 即可绕开**。
- **适用性点评**：UV 重打包直接 `pip install xatlas`，MIT，最省事。

---

## 4. CGAL

- **结论：不可用**（GPL/LGPL 传染 + Python 绑定同为 GPL + 安装依赖重）
- **URL**：https://github.com/CGAL/cgal ；Python 绑定 https://github.com/CGAL/cgal-swig-bindings
- **License（精确）**：CGAL 按文件双许可证 **GPL-3.0-or-later / LGPL-3.0-or-later**（实测 `Installation/LICENSE`：核心在 `include/CGAL`，逐文件声明 GPL 或 LGPL；另有 Boost/BSL、CORE/ImageIO=LGPL 等第三方）。可向 GeometryFactory 购买商业许可。
- **Python bindings**：`CGAL/cgal-swig-bindings` 仓库只有 `LICENSE.GPL` 与 `LICENSE.COMMERCIAL` → **绑定为 GPL-3.0**（395★，2025-12 更新）。conda-forge 有 `cgal` 6.0.1（Anaconda API 实测 license=GPL-3.0-or-later）。pip 源构建需 gmp/mpfr/boost/tbb（实测 setup.py 依赖列表）。
- **维护状态**：CGAL 本体活跃（pushed_at 2026-08-21，6015★）。
- **适用性点评**：LGPL 核心虽可链接（需满足再链接义务），但布尔/凸包等功能所在包 + Python 绑定为 GPL，对 A100 服务器内部工具链是传染风险，且编译安装成本高。**结论：全程避开 CGAL**，用 trimesh/manifold3d 替代。

---

## 5. 变形迁移（Deformation Transfer, Sumner & Popović 2004）开源实现

- **结论：可用**（有多个实现，但最稳的是自写最小二乘；最贴合需求的 `vasiliskatr` 直接产出 ARKit blendshapes）
- 候选清单（GitHub API 实测）：
  1. **vasiliskatr/deformation_transfer_ARkit_blendshapes** — MIT，69★，last push 2022-01（停更但任务完全对口：DT + 为任意人脸生成全部 ARKit blendshapes）。https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes
  2. **brianlaiii/VectorDisplacementTransfer** — **Apache-2.0**，31★，2025-03（较新），"blendShape delta 迁移，保留局部顶点旋转"，Maya 脚本（含 `tangentSpaceBlendShape.py` 黑盒版）。https://github.com/brianlaiii/VectorDisplacementTransfer
  3. **hendrikp/Deformation-Transfer-for-Triangle-Meshes** — MIT，3★，2022，纯 Python + 浏览器可视化，教学向。https://github.com/hendrikp/Deformation-Transfer-for-Triangle-Meshes
  4. **prashantdomadiya/Guided-Deformation-Transfer** — 无 license（❌ 不可直接商用），12★，2024-10。https://github.com/prashantdomadiya/Guided-Deformation-Transfer
  5. **ThibaultGROUEIX/3D-CODED** — 无 license（❌），328★，2021，PyTorch；注意其本质是"学习稠密对应"（deep correspondence），**不是**经典 DT，作者提到它是"3D-CODED + Learning Elementary Structure"。https://github.com/ThibaultGROUEIX/3D-CODED
- **质量评价**：`vasiliskatr` 是唯一"DT→ARKit blendshape"的现成轮子，MIT，可作为参考实现/起点；但 2022 年后未维护，依赖环境可能需小改。`VectorDisplacementTransfer` 是 Maya 向、Apache-2.0、较新，思路更工程化（保留局部旋转）。
- **适用性点评**：经典 DT 本质是稀疏最小二乘（约束 + Laplacian 平滑项，解一个 `A^T A x = A^T b`），用 scipy 自写约 100–200 行即可，license 最干净、最可控；建议以 `vasiliskatr` 为对照验证自写结果。

---

## 6. Delta Mush / Point Deform

- **结论：需自写**（无独立成熟的 Python 库；Maya 插件与 Blender 内建可作参照）
- **开源现状（GitHub API 实测）**：
  - `duncanskertchly/dm2skin`（Unlicense，62★，2017，Maya delta-mush→skin）
  - `ShaderManager/DeltaMush`（无 license，36★，2014，Maya C++）
  - `2TallTim/direct-delta-mush`（无 license，57★，2019）
  - `scorza/jsDelta*`（MIT，GPU 实现，2016）
  - 这些几乎都是 **Maya 插件**，无 pip 可装的独立 Python 库。
  - **Blender 内建 "Corrective Smooth" 修改器**（本质即 Delta Mush：平滑 + 细节回贴），可作为免开发参照。
- **适用性点评**：Delta Mush 数学很轻（对每顶点 delta = v − smooth(v)，用余切 Laplacian 平滑 delta，再 LBS + 加回平滑后 delta），用 numpy/scipy 自写约 50–80 行；不建议为它引入 Maya 插件依赖。

---

## 7. 3D 人脸模板：FLAME 与 ICT FaceKit

### 7.1 FLAME
- **结论：需申请**（免费注册 + 同意许可；许可为 CC-BY 署名，非"非商用"）
- **获取方式**：在 https://flame.is.tue.mpg.de/download.php 注册（免费）并同意模型许可后下载（FLAME-Universe README 实测确认流程；也有 `Rubikplayer/flame-fitting/fetch_FLAME.sh` 脚本，但仍需先注册取得凭证）。
- **License（精确）**：随模型分发的 `Readme.pdf`（实测文本）明确写 **"FLAME is available under Creative Commons Attribution license"**（即 CC-BY，署名，非 NC）；并强制要求论文引用。官方 `model_license` 页面为 JS 渲染无法 curl（标：正文以捆绑 Readme.pdf 为准）。
- **几何内容**：FLAME 含线性 identity 空间 + **articulated neck / jaw / eyeballs** + 姿态相关 corrective blendshapes + 全局表情 blendshapes。→ **含眼球，但不含牙齿/舌头/口腔内部**（嘴部是开洞）。
- **适用性点评**：challenge 已提供 T-pose glb，模板可自建，FLAME 非必需；若想用标准拓扑/表情先验可申请（CC-BY 可用），但需额外补牙齿/舌头/内口腔资产。

### 7.2 ICT FaceKit
- **结论：可用**（MIT，且自带牙齿/舌头/眼球/泪液/睫毛全量内口腔几何，极适合本 challenge）
- **URL**：https://github.com/USC-ICT/ICT-FaceKit（754★，last push 2020-12，稳定）
- **License（精确）**：**MIT**（实测 `LICENSE` 文件，Copyright (c) 2020 USC Institute for Creative Technologies）。
- **几何内容（README 实测）**：ICT Face Model **Light** 含完整头部 26719 顶点 / 26384 面，分 17 组：Face、Head and Neck、Mouth socket、Eye socket L/R、**Gums and tongue**、**Teeth**、**Eyeball L/R**、Lacrimal fluid L/R、Eye blend L/R、Eye occlusion L/R、Eyelashes L/R。
- **适用性点评**：MIT + 自带牙齿/舌头/眼球/泪液/睫毛，正好补齐中英口型同步所需的"口腔内部 + 眼球"资产，是比 FLAME 更适合本 challenge 的模板补充来源。注意：repo 提供的是"Light"版模型；ICT 完整扫描模型需另行与 USC 协商（未确认细节），Light 版按 MIT 即可。

---

## 8. ARKit 52 blendshapes

- **结论：可用**（官方文档公开，52 个名称稳定，硬编码即可）
- **官方文档**：Apple `ARFaceAnchor.BlendShapeLocation` — https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
- **52 个名称（经 Unity `ARKitBlendShapeLocation` 枚举 + Apple 文档交叉核验，52 项）**：
  `browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight, cheekPuff, cheekSquintLeft, cheekSquintRight, eyeBlinkLeft, eyeBlinkRight, eyeLookDownLeft, eyeLookDownRight, eyeLookInLeft, eyeLookInRight, eyeLookOutLeft, eyeLookOutRight, eyeLookUpLeft, eyeLookUpRight, eyeSquintLeft, eyeSquintRight, eyeWideLeft, eyeWideRight, jawForward, jawLeft, jawOpen, jawRight, mouthClose, mouthDimpleLeft, mouthDimpleRight, mouthFrownLeft, mouthFrownRight, mouthFunnel, mouthLeft, mouthLowerDownLeft, mouthLowerDownRight, mouthPressLeft, mouthPressRight, mouthPucker, mouthRight, mouthRollLower, mouthRollUpper, mouthShrugLower, mouthShrugUpper, mouthSmileLeft, mouthSmileRight, mouthStretchLeft, mouthStretchRight, mouthUpperUpLeft, mouthUpperUpRight, noseSneerLeft, noseSneerRight, tongueOut`
- **对照表/工具包**：
  - 中文对照：见 CSDN《ARKit 52个表情名称、顺序》（https://blog.csdn.net/cool_da/article/details/122064627）、Convai 映射参考（https://docs.convai.com/...mappings-reference）等社区资料（非官方）。
  - Python 工具：无单一权威 pip 包；`montybot/FACSHuman`（LGPL-3.0，55★，2026-04 更新）是 MakeHuman 的 FACS 插件，**不是**即插即用的 ARKit blendshape 库。LiveLinkFace 是 Epic 的商用方案（驱动 MetaHuman）。→ **建议直接硬编码上述 52 名 + 自建名称→blendshape 映射表**。
- **适用性点评**：名单公开稳定，无 license 风险；直接内置到 glb 的 `mesh.extras.targetNames` 与 `morph target` 命名即可。

---

## 9. glTF 导出（带 morph targets + skinning）

- **结论：可用**（三选一，按"省事 vs license 干净"取舍）
- **pygltflib**：
  - **URL**：GitLab https://gitlab.com/dodgyville/pygltflib（注意：主仓在 GitLab，不在 GitHub）
  - **License**：**MIT**（PyPI 实测 MIT）
  - **安装**：`pip install pygltflib`（PyPI v1.16.5，2025-07-24 上传，纯 Python，requires ≥3.6）
  - **能力**：完整 glTF 2.0 dataclass；实测源码 `Primitive` 含 `targets: List[Attributes]`（morph targets）、`Mesh.weights`、`Skin`、`Node`（skin/mesh/weights）、`Animation/AnimationChannel/AnimationSampler`（`WEIGHTS` 通道）→ **可手写 morph targets + skinning + 动画**，但属底层手工拼装（无自动蒙皮/自动权重）。
- **trimesh**：MIT，但 glb 导出**不支持 morph targets / skinning**（见第 2 节），只能做静态 glb。
- **Blender headless（bpy）**：
  - **License**：**GPL-2.0+**（Blender 本体）
  - **能力**：`blender -b --python export_script.py` 导出带 shape keys（morph targets）+ armature（skinning）+ 动画的 glb，最省事、最稳。
  - **License 注意**：Blender 是 GPL，但**导出产物（.glb）不受 GPL 传染**（GPL 约束 Blender 程序本身，不约束其输出数据）；真正引入 GPL 的是"直接 `import bpy` 进我们 Python 进程"的脚本——若用 subprocess 调 `blender -b` 只是把 Blender 当工具用，通常视为安全。保守起见，若想全程 MIT，选 pygltflib。
- **适用性点评**：**最省事=bpy headless**（正确性最好）；**license 最干净=pygltflib**（MIT，但需自己组装 buffer/accessor/skin 矩阵）。推荐：管线主链路用 pygltflib（MIT），必要时用 bpy headless 做一次"黄金样例"交叉校验。

---

## 10. 加分项：近期自动 face rigging 开源项目（RigAnyFace 等）

- **RigAnyFace**：
  - **结论：未公开代码**（论文公开，代码仓库未发现）
  - 论文：NeurIPS 2025，《RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data》，arXiv:2511.18601；项目页 https://wenchao-m.github.io/RigAnyFace.github.io/（可下载论文/视频）。
  - GitHub API 搜 `RigAnyFace` **无任何代码仓库**（未确认，截至 2026-08-21）。→ 标「未确认/未公开」，不要等它。
- 其他近期项（GitHub API 实测，均非完整自动 rig 一体机）：
  - `yfeng95/DECA`（NOASSERTION/自定义，2513★，FLAME 表情重建）、`soubhiksanyal/FLAME_PyTorch`（MIT，810★）、`Rubikplayer/flame-fitting`（无 license，820★）等，均为"重建/拟合"而非"全自动绑定输出 glb"。
  - 真·自动绑定近期主要是商业/闭源（如 MetaHuman、RigAnyFace 未开源）。→ 对本 challenge，按 Stage 2 手工几何管线（DT + ARAP + blendshape 输出）是最可控路径。

---

## 11. 推荐组合（最小技术栈）

### Stage 2 几何管线最小栈（纯 Python + 少量 C/C++ wheel）

| 用途 | 工具 | License | 安装 |
|---|---|---|---|
| 网格 I/O、布尔、凸包、SDF、最近点焊接、清洗 | **trimesh**（布尔后端 manifold3d） | MIT / Apache-2.0 | `pip install trimesh` |
| UV 重打包（xatlas 类） | **xatlas-python** | MIT | `pip install xatlas` |
| ARAP 形变 | 自写（scipy 稀疏 Cholesky + 局部旋转 SVD），或 libigl 仅编译 MPL-2.0 子集 | MIT（自写） | `pip install scipy` |
| SDF 碰撞处理 | **trimesh.proximity.signed_distance**（或自写 KD 树 + winding number） | MIT | `pip install trimesh` |
| 凸包减法布尔 | **trimesh.boolean / manifold3d** | MIT/Apache-2.0 | 随 trimesh |
| 变形迁移（Sumner & Popović） | 自写稀疏最小二乘；对照 `vasiliskatr`（MIT） | MIT | `pip install scipy` |
| Delta Mush 平滑 | 自写（余切 Laplacian） | MIT | `pip install scipy` |
| FACS/ARKit 52 blendshape 输出 + glb 导出（morph+skinning） | **pygltflib**（主），bpy headless（校验） | MIT / GPL(仅作工具) | `pip install pygltflib` |
| 数值/稀疏 | numpy + scipy | BSD | `pip install numpy scipy` |

> 一句话：**trimesh + manifold3d（布尔/凸包/SDF/焊接）+ xatlas-python（UV）+ scipy（ARAP/DT/Delta Mush 自写）+ pygltflib（glb morph+skinning）**，全部 MIT/Apache-2.0/BSD，无 GPL，A100 服务器部署无传染风险。

### FLAME / 模板选择建议

- **首选**：直接用 challenge 官方提供的 T-pose glb 作基底模板（最省事、语义对齐）。
- **口腔内部 + 眼球补齐**：用 **ICT-FaceKit（MIT）** 的 Teeth / Gums-and-tongue / Eyeball / Lacrimal / Eyelashes 组，或从该模板自建牙齿/舌头。
- **FLAME**：仅在需要标准表情先验/眼-颌-颈关节参数化时申请（免费注册，CC-BY 署名），注意它**不含牙齿/舌头**，需自行补内口腔。
- **ARKit 52 blendshapes**：直接硬编码官方名单，经变形迁移映射到自建模板顶点。

---

## 附：字段速查索引

- 【结论】各组件首行。
- License 传染性提醒：**GPL-3.0 出现在** libigl `copyleft/`（boolean/convex_hull/CGAL-SDF）、libigl Python 绑定包、CGAL 及其 swig bindings、Blender（bpy）。**nvdiffrast 为非商用**。其余推荐项均为 MIT/Apache-2.0/BSD/CC-BY。
- 「未确认」项：RigAnyFace 代码仓库（未公开）、FLAME 官方 `model_license` 页面正文（JS 渲染，以捆绑 Readme.pdf 的 CC-BY 为准）、ICT 完整（非 Light）模型授权条款。
