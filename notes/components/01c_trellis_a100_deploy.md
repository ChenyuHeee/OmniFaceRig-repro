# TRELLIS A100 真实部署实录(2026-08-25,全部真实跑通)

> 服务器:A100-SXM4-40GB,conda env `torch2.4_cuda12.1`(torch 2.4.1+cu121)
> 结果:官方 2D 图 → TRELLIS mesh(16K 顶点,带纹理)→ stage1 rig → 52 morph + 53 关节 glb,~2 分钟/图

## 部署步骤(踩坑全记录)

1. **权重**:hf-mirror.com 实测服务器直连 0.2-5MB/s 波动;`curl -C -` 逐文件 + 尺寸校验
   (`dl_trellis2.sh` 模式,期望尺寸从本机缓存获取)。**多次重启 curl 会损坏文件**
   (两个 curl 同写一文件)→ 单进程 + rm 重下 + size 校验。
2. **源码**:codeload.github.com tarball(本机下载→sftp 上传);**git 子模块 FlexiCubes**
   需单独下载(MaxtirError/FlexiCubes)→ 解压到 trellis/representations/mesh/flexicubes/。
3. **依赖**(清华镜像装,快 2 倍):rembg onnxruntime xatlas easydict igraph pyvista
   pymeshfix timm transformers;**open3d**(447MB,仅文本管线用)可跳过;
   **utils3d** 从 git 装(本机下载 tarball 上传)。
4. **kaolin**:pypi 的 kaolin 是占位包 → flexicubes.py 打补丁
   (`check_tensor` 本地实现)。
5. **flash_attn 缺失**:TRELLIS 日志显示 "Backend: flash_attn" 但没装包。
   顶层 attention 支持 `ATTN_BACKEND=sdpa`,但 **sparse attention 不支持 sdpa** →
   打补丁(3 个文件加 sdpa 分支:windowed/full/serialized;block-diagonal mask 实现)。
6. **spconv**:`pip install spconv-cu120`(清华镜像有)。
7. **dinov2**(torch.hub):预置 `~/.cache/torch/hub/facebookresearch_dinov2_main`(dinov2 源码)
   + `~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth`(hf-mirror showstarpro 镜像)。
   **关键**:`torch.hub.load` 会先连 github.com 检查(被墙→SYN-SENT 卡死)→
   trellis_front.py patch 改为**本地目录 + source='local'**。
8. **diff-gaussian-rasterization**:用 graphdeco-inria main 版报 `kernel_size` 参数错误 →
   必须用 **mip-splatting 的 submodule 版**(autonomousvision/mip-splatting tarball 内含,带 kernel_size);
   `CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST='8.0' pip install --no-build-isolation .`。
9. **运行环境**(webapp 子进程必须注入):
   ```
   export TRELLIS_MODEL_PATH=$HOME/work/models/TRELLIS-image-large
   export TRELLIS_DINOV2_PTH=$HOME/.cache/torch/hub/checkpoints/dinov2_vitl14_reg_pretrain.pth
   export ATTN_BACKEND=sdpa
   export PYTHONPATH=$HOME/work/src/TRELLIS-main
   ```

## 运行

```
python scripts/trellis_front.py --image imgs/elon.png --out outputs/mesh.glb   # ~2min
python scripts/stage1_real.py --glb outputs/mesh.glb --inner-mouth --out outputs/rigged.glb
python scripts/animate_audio.py --glb outputs/rigged.glb --out outputs/talk.glb \
    --text "Hello world" --lang en     # 真实音频口型(piper+whisper)
```

Web 端到端:`POST /api/rig` + `image_to_mesh=1` + 图片 → 全链路自动完成。
