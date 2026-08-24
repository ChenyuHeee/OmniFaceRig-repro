# TRELLIS 图→mesh 前端 — 部署实测记录(issue #3)

> 核验日期:2026-08-25。执行者:agent trellis-front(分支 `agent/trellis-front`)。
> 全部仓库/权重信息均通过 gh API / HF(hf-mirror)API 现场核实,下载速率为本机实测。

## 1. 核实结论(仓库 / 版本 / 安装方式)

| 项 | 值 | 核实方式 |
|---|---|---|
| 代码仓库 | `microsoft/TRELLIS`,⭐ 13493,**MIT**,`pushed_at 2026-06-26` | gh api |
| 备选(v2) | `microsoft/TRELLIS.2`,⭐ 10813,MIT,24GB 显存 — 本任务用 v1(16GB) | gh api |
| pip 包 | **无官方 `trellis` pip 包**(`pip index versions trellis` → Not Found);须源码安装 | pip |
| 安装方式 | clone + `. ./setup.sh --new-env --basic --xformers [--flash-attn] --diffoctreerast --spconv`;或复用服务器 `torch2.4_cuda12.1` env 后 `pip install -e .` | README |
| 硬件 | README 明确 **≥16GB VRAM**(A100/A6000 验证);服务器 A100-40GB 满足 | README |
| 推理 API | `TrellisImageTo3DPipeline.from_pretrained(path)` → `pipeline.run(image, seed=1, formats=[...])` → `postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95, texture_size=1024)` → `glb.export(...)` | README + 源码 |

## 2. 权重清单(`microsoft/TRELLIS-image-large`,合计 ≈ 3.3 GB)

推理必需 6 个 checkpoint(与 `pipeline.json` 的 models 值一一对应,文件名即 basename):

| 文件 | 字节 |
|---|---|
| `ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors` | 1,130,770,840 |
| `ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors` | 1,203,755,136 |
| `ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 147,591,972 |
| `ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors` | 171,450,952 |
| `ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors` | 171,450,488 |
| `ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors` | 181,903,412 |

训练才用的编码器(非推理必需):`ss_enc_conv3d_16l8_fp16`(119,068,016)、
`slat_enc_swin8_B_64l8_fp16`(173,242,816)。另有每模型 `.json` 配置 + `pipeline.json`。

⚠️ 命名陷阱:HF 文件名是 `slat_dec_gs_*`(不是 pipeline 模型键 `slat_decoder_gs_*`),
`from_pretrained` 按 basename 找 `{name}.json` + `{name}.safetensors`。

**附带权重**(推理还需要,容易漏):
- 图像编码器 `dinov2_vitl14_reg`(DINOv2 ViT-L/14 reg,~1.2GB):TRELLIS 用
  `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)`,
  官方从 `dl.fbaipublicfiles.com` 拉取(**本机 403 被墙**)。方案:从
  `hf-mirror.com/showstarpro/dinov2_vitl14_reg4_pretrain` 下载原生
  `dinov2_vitl14_reg4_pretrain.pth`(1,217,607,321 字节,已核验内含原生 state-dict
  键:cls_token / pos_embed / register_tokens / blocks.*,无前缀),
  改名放到 `$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth`,
  torch.hub 按 URL basename 命中缓存即不再联网。
- 抠图模型 `rembg u2net.onnx`(~176MB):TRELLIS `preprocess_image` 用
  `rembg.new_session('u2net')`,放 `~/.u2net/u2net.onnx` 即免下载。
  github release 直连慢/被墙(~137KB/s);已从 `hf-mirror.com/Gulraiz00/u2net`
  取到同尺寸文件(构建略异,冒烟够用)。

## 3. 网络实测(本机,2026-08-25)

| 源 | 实测 | 结论 |
|---|---|---|
| `huggingface.co` 直连 | 连接超时(0 B/s) | ✗ 不可用 |
| **`hf-mirror.com`** | **~8-11 MB/s**(141MB 用 18s;1.2GB 用 114s;全量 3.3GB 一次成功) | ✓ **主通道** |
| `codeload.github.com`(源码 tarball) | ~8 MB/s(44MB 含子模块) | ✓ 源码通道 |
| `dl.fbaipublicfiles.com`(dinov2) | 403 | ✗ 走镜像 |
| `github.com` 直连(rembg release) | 超时/137KB/s 波动 | △ 走镜像或跳过 |
| gh API 通道 | 正常 | ✓ 兜底(tarball 中转) |

**关键结论:此前记录的「hf-mirror 37KB/s → 3.3GB 需 ~25h」已过时/不准确;
hf-mirror 实测 ~10MB/s,全量权重 ~6 分钟可下完。权重不再构成 blocker。**

本机已完整下载并核验:`~/.cache/trellis/TRELLIS-image-large`(19 个文件,
8 个 safetensors 尺寸逐字节匹配 HF API 清单、safetensors 头可解析)、
`~/.cache/trellis/dinov2/dinov2_vitl14_reg4_pretrain.pth`、
`~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth`(重命名缓存)、
`~/.u2net/u2net.onnx`。

## 4. 交付物

- `code/scripts/trellis_front.py`:`image_to_mesh(image_path, out_glb_path, device="cuda") -> out_glb_path`;
  `--mock` 占位几何(椭圆头 + 程序化纹理);权重缺失抛 `TrellisFrontError`(带修复指引)。
- `code/scripts/deploy_trellis.sh`:`--weights / --source / --vram / --all`;
  HF_ENDPOINT 可换源、curl `-C -` 断点续传 + 尺寸校验、gh API tarball 兜底说明、VRAM 检查。
- `tests/test_trellis_front.py`:3 个契约测试(mock→stage1 可消费、缺权重报错清晰、缺输入报错)。
- 本地验证:31 个 pytest 全过(含新增 3 个);mock glb(V=1058, F=2112, TEXCOORD+material+image)
  经 `stage1_real.load_mesh` 读取成功。

## 5. 剩余工作(网络恢复/服务器部署后)

1. 服务器上跑 `deploy_trellis.sh --all`(conda env + 源码 + 权重 + dinov2 + u2net);
2. 编译 CUDA 子模块(diffoctreerast / FlexiCubes,需 CUDA toolkit 11.8/12.2,与
   服务器 `torch2.4_cuda12.1` 匹配);
3. 真实推理冒烟:`python scripts/trellis_front.py --image <人物图> --out /tmp/x.glb`
   (A100-40GB,预计单张 ~1-2 分钟);
4. 接 `stage1_real.py --glb /tmp/x.glb` 走完整 rig 链路;
5. 若仍需提速,可只下推理必需 6 个 checkpoint(~2.8GB)跳过两个编码器。

## 6. 风险与备注

- TRELLIS 是通用物体生成模型,人脸/口腔细节非其专长(与调研 01_image_to_3d.md 一致);
  inner-mouth 几何仍由 inner-mouth agent 负责。
- `--mock` 产物满足 stage1 输入契约但几何简单,仅用于链路测试。
- gulraiz00 u2net.onnx 与官方 rembg u2net.onnx 同尺寸、构建略有差异;正式部署建议
  用官方文件(或接受冒烟级抠图)。
