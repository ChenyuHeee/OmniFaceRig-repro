# 04 · 音频 → 3D 面部 blendshape 权重驱动方案调研

> 目标：为 OmniFaceRig 复现 challenge 交付物 4（输出 ARKit 52 blendshapes）+ 交付物 5（**中/英文双语音唇形同步、正确对齐**）找一个「音频（中文+英文）→ 3D 面部 blendshape 权重（最好直接 ARKit 52 命名）」的驱动方案。
>
> 背景：OmniFaceRig 论文本身**没有音频驱动模块**，本环节完全独立选型。
>
> 调研方式：`web_search` + `curl GitHub API / raw README / 官方文档 / NVIDIA 页面` 实地核验（2026-08），非记忆复述。查不到的标注「未确认」。

---

## 0. 一句话总结

- **英文**：有现成方案，两条路都成立 —— ① 官方级质量：NVIDIA Audio2Face-3D（开源权重 + SDK，**直接输出 ARKit blendshape**，需 NVIDIA GPU）；② 轻量 CPU：Rhubarb Lip Sync（音素→嘴型时间线）或 TalkingHead（词→Oculus viseme→**ARKit 权重表**，MIT）。
- **中文**：**没有现成的「中文音频→ARKit 52 权重」开源模型**。最接近的是经典 Omniverse Audio2Face（官方确认中英文训练），但新版开源 Audio2Face-3D 的中文支持**未官方声明**。中文必须走「自映射」路线：TTS 前端拿声韵母/音素序列 → 查 viseme 表 → 映射 ARKit → 平滑插值。
- **最大风险**：中文路径是自建映射 + 对齐，效果上限于映射表质量与音素时间对齐精度；交付物 5 的「中英双语对齐」是最大工程量点。

---

## 1. Rhubarb Lip Sync

- **结论：可用（英文）/ 部分支持（中文）**
- URL：https://github.com/DanielSWolf/rhubarb-lip-sync
- License：**MIT**（代码本身）；README 明确「对音频跑出的口型数据归用户所有，生产用途甚至无需关心 MIT」。第三方依赖均为 permissive（MIT/BSD）。见 [LICENSE.md](https://github.com/DanielSWolf/rhubarb-lip-sync/blob/master/LICENSE.md)。
- 输入/输出：输入 WAV/OGG（另支持扩展名解码），输出**音素→嘴型时间线**（TSV/XML/JSON）。嘴型为 Hanna-Barbera 9 嘴型体系：`A B C D E F`（基本）+ `G H X`（扩展）。**不是 ARKit 权重**。
- **中文支持（重点）**：**部分支持**。Rhubarb 有两个识别器（`--recognizer`）：
  - `pocketSphinx`（默认）：**仅英文**。README 原话「PocketSphinx only recognizes English dialog」。
  - `phonetic`：**语言无关**。README 原话「recognize individual sounds and syllables … language-independent … use it if your recordings are not in English」，但「results are usually less precise」。
  - 作者在 [issue #5「Languages」](https://github.com/DanielSWolf/rhubarb-lip-sync/issues/5) 明确：不打算近期官方支持非英文，难点在声学模型/语言模型/G2P 规则（其音素映射按美式英语写的，超过 200 条 soundchange 规则）。所以 `phonetic` 模式对中文**能跑但精度没有保证**（声学模型仍是英文音素库，普通话声调/声母归并会不准）。
- 维护状态：**活跃**。最新 release [v1.14.0（2025-04-03）](https://github.com/DanielSWolf/rhubarb-lip-sync/releases/latest)，仓库最后 push 2026-06-16。
- 硬件：**纯 CPU**，命令行工具，极轻量，秒级处理。
- 是否直接输出 ARKit 52：**否**（输出 9 嘴型），需自己映射到 ARKit（A→mouthClose、D→jawOpen/mouthOpen、F→mouthFunnel/mouthPucker 等）。
- 对我们 challenge 的适用性：英文离线兜底方案最佳选择（时序对齐好、CPU 可跑）；中文只能靠 `phonetic` 模式勉强用，建议中文另走自映射路线。

---

## 2. Wav2Lip

- **结论：不可用（对本交付物）—— 是 2D 视频方案，非 3D blendshape**
- URL：https://github.com/Rudrabha/Wav2Lip
- License：**无正式 OSS license**（GitHub `license: None`）。README 明确「can only be used for personal/research/non-commercial purposes」——因为模型训练自 LRS2 数据集，商业使用严格禁止；商业版已转给 Sync Labs（[README §License and Citation](https://github.com/Rudrabha/Wav2Lip)）。
- 输入/输出：音频 + 单张人脸/视频 → **2D 下半脸视频帧**（GAN 合成嘴唇区域贴回画面）。**完全不涉及 3D blendshape**。
- **中文支持**：**差/部分**。原模型英文训练（LRS2），中文效果差；社区有 [zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip](https://github.com/zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip)（用 CMLR 中文数据重训）。
- 维护状态：基本停更（最后 push 2025-06-22，仅 README 指向商业版）。
- 硬件：**需 GPU**（人脸检测 + 生成器推理）。
- 对我们 challenge 的适用性：只能作为 2D 效果对比/可视化参考，**不满足交付物 4 的 ARKit 52 blendshape**，直接排除。

---

## 3. MeshTalk / FaceFormer / CodeTalker（3D 语音驱动）

三者共同点：**输出 = 3D 网格「顶点位移」（BIWI / VOCA / FLAME 拓扑），不是 blendshape**，且均英文训练。要把它们接入 ARKit，需要额外做「顶点位移 → ARKit 52 权重」的 retarget（工程量不小）。

### 3.1 MeshTalk
- **结论：不可用（非商业 + 已归档 + 非 blendshape）**
- URL：https://github.com/facebookresearch/meshtalk
- License：**CC BY-NC 4.0**（非商业，见 [LICENSE](https://github.com/facebookresearch/meshtalk/blob/main/LICENSE)）。
- 输出：FLAME/BIWI 拓扑顶点位移。英文（wav2vec 特征）。
- 维护状态：**archived=True**（2022-10 后停更）。
- 硬件：需 GPU。

### 3.2 FaceFormer
- **结论：部分可用（仅英文 + 顶点位移，需 retarget）**
- URL：https://github.com/EvelynFan/FaceFormer
- License：**MIT**（GitHub API；模型/数据另见其各自许可）。CVPR 2022，wav2vec2 encoder + Transformer 自回归。
- 输出：**顶点位移**（BIWI topology，5023 顶点 / VOCA FLAME 系），非 blendshape。
- 中文：**否**（英文 wav2vec2 + BIWI/VOCA 英文数据）。
- 维护状态：一般（最后 push 2023-08）。
- 硬件：推理可小 GPU（模型不大）。

### 3.3 CodeTalker
- **结论：部分可用（仅英文 + 顶点位移，需 retarget）**
- URL：https://github.com/Doubiiu/CodeTalker
- License：**MIT**。CVPR 2023，离散 motion code（VQ）+ 说话人条件。
- 输出：**顶点位移**（BIWI topology），非 blendshape。
- 中文：**否**（英文数据）。
- 维护状态：一般（最后 push 2023-09）。
- 硬件：推理可小 GPU。

> 三者对 challenge 适用性：**都不直接满足 ARKit 52 输出**；只有「英文顶点位移」，还得额外做 retarget。仅在「自训一个数据驱动模型」时才值得参考（FaceFormer/CodeTalker 是自训中文变体最常用的 backbone）。

---

## 4. NVIDIA Audio2Face-3D（开源版）+ Audio2Face NIM API

- **结论：可用（英文）/ 部分可用-未确认（中文）—— 唯一官方「直接输出 ARKit blendshape」的开源方案**
- 开源仓库（collection）：https://github.com/NVIDIA/Audio2Face-3D
  - **Audio2Face-3D SDK（Audio2X SDK）**：**MIT**，C++/CUDA/TensorRT。https://github.com/NVIDIA/Audio2Face-3D-SDK
  - **Training Framework**：Apache。https://github.com/NVIDIA/Audio2Face-3D-Training-Framework
  - Maya ACE 插件 / UE5 插件：MIT。
  - 预训练模型（Hugging Face，**NVIDIA Open Model License**，可商用）：[Audio2Face-3D-v3.0](https://huggingface.co/nvidia/Audio2Face-3D-v3.0)（diffusion，HuBERT encoder，180M 参数）、v2.3.x（regression，Mark/Claire/James）。
  - 论文：[arXiv:2508.16401](https://arxiv.org/abs/2508.16401)（2025-09，开源了 networks/SDK/training framework/example dataset）。
- **是否直接输出 ARKit 52 权重：是**。官方 [Audio2Face-3D-Samples README](https://github.com/NVIDIA/Audio2Face-3D-Samples) 原话：「converts speech into facial animation in the form of **ARKit Blendshapes**」；[NIM 文档](https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/index.html) 同样写「Convert audio input into lifelike facial animations **using ARKit blendshapes**」。输出含 jaw/tongue/eye/skin 等。
- **NIM API 可用性/免费额度**：
  - 自托管：NGC container + Helm chart（[catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/audio2face-3d)），**NVIDIA Software License Agreement（商业容器，非开源）**。
  - 托管 API：https://build.nvidia.com/nvidia/audio2face-3d（NVIDIA Build，`nvapi-` key）。
  - 免费额度：NVIDIA Build 对多数 NIM 模型有试用额度/免费 tier（2026 年社区仍报道「免费 API、不绑卡」），但 **A2F-3D NIM 的精确免费额度「未确认」**，需注册 build.nvidia.com 后按页面为准（用量通常按音频秒数/推理时间计）。
- **中文支持（重点）**：
  - 经典 Omniverse Audio2Face：NVIDIA 官方论坛（2023-05）明确「**Audio2Face is trained with English and Chinese languages for now**」—— 即老版**中英文都训练过**。见 [forums.developer.nvidia.com「Arabic language support」回帖](https://forums.developer.nvidia.com/t/arabic-language-support/251398/2)。
  - 新版开源 **Audio2Face-3D（2025）**：模型卡**未明确列出支持语言**（只写「Deployment Geography: Global」、音频 16kHz、HuBERT encoder、训练数据 <10,000 小时）。**中文支持「未确认」**——大概率英文为主、中文能跑但效果未保证，需实测。
- 硬件：**必须 NVIDIA GPU**（SDK 需 CUDA 12.8 + TensorRT，不支持纯 CPU）；Windows/Linux。NIM 可云端调用或自托管 Docker（也需 GPU）。
- 对我们 challenge 的适用性：**英文最佳方案**（官方、直出 ARKit 52、可商用 MIT 权重/SDK）；中文可用性不确定，且依赖 NVIDIA GPU/云端。

---

## 5. uTalk / VSFA / Modular Talking Head（其他 viseme 方案）

- **uTalk**：**未确认/未找到**对应的语音驱动口型开源项目。同名论文 [arXiv:2310.02739「uTalk: Bridging the Gap Between Humans and AI」](https://arxiv.org/abs/2310.02739) 是通用 AI 助手/HCI 方向，与口型无关。GitHub 上无 NVIDIA 官方 uTalk 口型仓库（搜索 2026-08 无结果）。很可能记混了（疑似 UniTalker / EmoTalk / MODA 之类）。
- **VSFA**：**未确认/非口型项目**。GitHub 上 [lidq92/VSFA](https://github.com/lidq92/VSFA)（MIT）= 「Quality Assessment of In-the-Wild Videos」（ACM MM 2019，视频质量评价），与唇形无关。未找到叫 VSFA 的语音驱动 3D 口型工作。
- **Modular Talking Head**：**未找到确定的开源项目**（无同名公开仓库/论文命中）。可能指代的相关工作（均可查、但输出顶点位移、英文为主）：EmoTalk（ICCV 2023，语音情感 3D 面部动画）、UniTalker（[X-niper/UniTalker](https://github.com/X-niper/UniTalker)，统一说话头）。均**不直接输出 ARKit blendshape**。
- 对本 challenge 适用性：这一组基本**可忽略**，把精力放在 Rhubarb/TalkingHead + Audio2Face 上。

---

## 6. 中文口型：公开实现 / 数据集

- **CMLR（Chinese Mandarin Lip Reading）**：中科院 VIPL 的普通话唇读数据集（2D 视频）。社区有 [zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip](https://github.com/zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip)（CMLR 预处理版，用于中文 Wav2Lip）。组织主页 [VIPL-Audio-Visual-Speech-Understanding](https://github.com/VIPL-Audio-Visual-Speech-Understanding)。
- **LRW-1000**：中文 in-the-wild 唇读数据集（2D 视频，1000 词），Nature 分布。同上，**均为 2D 视频/唇读数据，非 3D blendshape 权重**。
- **PaddleSpeech（百度）**：https://github.com/PaddlePaddle/PaddleSpeech —— 中文 TTS 前端可输出**拼音/音素序列**（`Chinese.phoneticize` / zh text frontend），适合做「中文文本→声母/韵母/音素」前端。**本身不输出 viseme 或 3D blendshape**。
- **中文「语音→ARKit 52 权重」对齐数据集**：**未找到公开大数据集**（基本空白）。商业 capture 数据（Tencent FaceGood 等）不公开。
- 结论：中文有「2D 视频唇读数据（CMLR/LRW-1000）」+「音素前端（PaddleSpeech 等）」，但**缺公开的 3D blendshape 对齐数据** → 中文必须自映射。

---

## 7. ARKit 52 blendshape 的 viseme 映射表（公开可直接抄）

- **结论：可用**。最干净、可直接抄的公开映射在 **TalkingHead** 项目里：

  [met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead)（MIT，1486★，2026-06 仍活跃，纯前端 JS/CPU 实时口型）。

  其 [blender/build-visemes-from-arkit.py](https://github.com/met4citizen/TalkingHead/blob/main/blender/build-visemes-from-arkit.py) 给出 **Oculus 15-viseme → ARKit blendshape 权重**的完整表（这是 Meta Oculus LipSync 的 de-facto 标准 viseme 集，`sil, PP, FF, TH, DD, kk, CH, SS, nn, RR, aa, E, I, O, U`）：

  | Viseme | ARKit 权重（其余 mouth/jaw 项为 0） |
  |---|---|
  | `aa` | jawOpen=0.6 |
  | `E` | mouthPressL/R=0.8, mouthDimpleL/R=1.0, jawOpen=0.3 |
  | `I` | mouthPressL/R=0.6, mouthDimpleL/R=0.6, jawOpen=0.2 |
  | `O` | mouthPucker=1.0, jawForward=0.6, jawOpen=0.2 |
  | `U` | mouthFunnel=1.0 |
  | `PP` | mouthRollLower=0.8, mouthRollUpper=0.8, mouthUpperUpL/R=0.3 |
  | `FF` | mouthPucker=1.0, mouthShrugUpper=1.0, mouthLowerDownL/R=0.2, mouthDimpleL/R=1.0, mouthRollLower=1.0 |
  | `DD` | mouthPressL/R=0.8, mouthFunnel=0.5, jawOpen=0.2 |
  | `SS` | mouthPressL/R=0.8, mouthLowerDownL/R=0.5, jawOpen=0.1 |
  | `TH` | mouthRollUpper=0.6, jawOpen=0.2, tongueOut=0.4 |
  | `CH` | mouthPucker=0.5, jawOpen=0.2 |
  | `RR` | mouthPucker=0.5, jawOpen=0.2 |
  | `kk` | mouthLowerDownL/R=0.4, mouthDimpleL/R=0.3, mouthFunnel=0.3, mouthPucker=0.3, jawOpen=0.15 |
  | `nn` | 同 `kk` + tongueOut=0.2 |
  | `sil` | （空） |

  TalkingHead 同时内置 word→viseme 的语言规则（英/德/芬/法/立陶宛语，`modules/lipsync-en.mjs` 等），并含 52 个 ARKit blendshape 的完整命名参考。

- 其他来源：Rhubarb 输出的是 9 嘴型（非 viseme/ARKit），需自己建 A–H–X → ARKit 映射。ARKit 52 标准命名（jawOpen/mouthFunnel/mouthPucker/…）见 Apple 文档与上述项目。**没有**单独的「官方音素→ARKit 52」表，社区映射即上面这张 Oculus viseme 表 + 自建「音素→viseme」字典。

---

## 8. 英文现成 speech→ARKit 开源桥接库

- **TalkingHead**（[met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead)）：MIT，JS 类库，实时把「词/文本 → Oculus viseme → **ARKit blendshape 权重**」，纯 CPU，支持 Ready Player Me / VRM / CC4 等 avatar。**最接近「现成英文 speech→ARKit」的开源桥接**（自带 viseme→ARKit 表，见 §7）。
- **Rhubarb + 自建映射**：Rhubarb 输出嘴型时间线 → 映射 ARKit（需自己写 A–H–X → ARKit 表）。
- **NVIDIA Audio2Face-3D SDK**：官方直出 ARKit，但需 NVIDIA GPU（§4）。
- 其它：Oculus LipSync（Meta，Unity 插件，闭源商业授权）等，不在开源首选。

---

## 9. 推荐组合

### 英文路径（推荐）
1. **首选（对齐/质量最好）**：NVIDIA **Audio2Face-3D** 开源模型 + SDK（MIT 权重 + MIT SDK，**直接输出 ARKit 52 权重**，含 jaw/tongue/eye）。前提：有 NVIDIA GPU（CUDA 12.8/TensorRT）或调用 build.nvidia.com NIM API。
2. **轻量 CPU 兜底**：**Rhubarb**（`--recognizer pocketSphinx`）拿嘴型时间线 → 自建 A–H–X → ARKit 映射；或直接用 **TalkingHead**（词→Oculus viseme→ARKit 表，纯 JS/CPU）。

### 中文路径（推荐，需自映射）
没有现成开源「中文音频→ARKit 52 权重」模型。最小可行路线：

1. **拿音素序列 + 时间对齐**：用中文 TTS 前端（**PaddleSpeech** Chinese frontend / WeNet / g2pW）把文本转成**声母+韵母（音素）**；若只有音频，用 ASR（Whisper/ParaFormer）出文本 + 强制对齐（或 VAD + 能量）得到每个音素的起止时间。
2. **建立「中文声韵母 → viseme → ARKit」查表**：复用 §7 的 Oculus viseme→ARKit 权重表，再补一层普通话声韵母→viseme 映射（例如：开口呼 a/o/e→`aa`/`O`/`E`，i→`I`，u/ü→`U`，双唇 b/p/m→`PP`，唇齿 f→`FF`，舌尖前 z/c/s→`SS`，舌尖后 zh/ch/sh/r→`CH`/`RR`，舌面 j/q/x→`I`+`CH` 混合，舌根 g/k/h→`kk`，鼻音 n/ng→`nn`）。
3. **平滑插值 + 输出 ARKit 52**：按音素时长做关键帧 + 指数/缓动平滑，其余 52 项里非口型项（眉眼等）置 0 或接 OmniFaceRig 的 idle，输出带时间戳的 ARKit 52 权重序列。
4. **（可选，数据驱动）**：用 Audio2Face-3D **Training Framework**（Apache）自训中文模型——但需要自己采集中文「音频+3D blendshape」对齐数据（公开数据基本空白），成本高，不建议在 challenge 周期内做。

### 最大风险
**中文「音频→ARKit 52 权重」无现成开源方案**：交付物 5 的中文对齐只能靠自建「音素→viseme→ARKit」映射 + 时间对齐，效果上限于映射表质量与对齐精度；英文可用 Audio2Face-3D（要 GPU）或 Rhubarb/TalkingHead（CPU）。其次是 **Audio2Face-3D 新版中文支持未官方确认**、且 SDK 强依赖 NVIDIA GPU。

---

## 附：关键链接索引

- Rhubarb：https://github.com/DanielSWolf/rhubarb-lip-sync （MIT；issue #5 语言说明 https://github.com/DanielSWolf/rhubarb-lip-sync/issues/5）
- Wav2Lip：https://github.com/Rudrabha/Wav2Lip （研究/非商业）
- MeshTalk：https://github.com/facebookresearch/meshtalk （CC BY-NC 4.0，archived）
- FaceFormer：https://github.com/EvelynFan/FaceFormer （MIT）
- CodeTalker：https://github.com/Doubiiu/CodeTalker （MIT）
- Audio2Face-3D：https://github.com/NVIDIA/Audio2Face-3D ｜ SDK https://github.com/NVIDIA/Audio2Face-3D-SDK ｜ Samples https://github.com/NVIDIA/Audio2Face-3D-Samples ｜ 模型 https://huggingface.co/nvidia/Audio2Face-3D-v3.0 ｜ 论文 https://arxiv.org/abs/2508.16401 ｜ NIM 文档 https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/index.html ｜ API https://build.nvidia.com/nvidia/audio2face-3d
- TalkingHead（viseme→ARKit 表）：https://github.com/met4citizen/TalkingHead ｜ 映射脚本 https://github.com/met4citizen/TalkingHead/blob/main/blender/build-visemes-from-arkit.py
- 中文数据：CMLR 预处理 https://github.com/zzj1111/Preprocessed-CMLR-Dataset-For-Wav2Lip ｜ VIPL https://github.com/VIPL-Audio-Visual-Speech-Understanding
- PaddleSpeech（中文前端）：https://github.com/PaddlePaddle/PaddleSpeech
- NVIDIA 论坛（经典 A2F 中英文训练声明）：https://forums.developer.nvidia.com/t/arabic-language-support/251398/2
