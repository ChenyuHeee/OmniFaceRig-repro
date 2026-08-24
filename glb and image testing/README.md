# glb and image testing · TA 提供的官方测试资源(2026-08-22 收到)

## 内容清单

| 路径 | 说明 |
|---|---|
| `glb/` | **23 个 T-pose 角色 glb 模型**(ai3d_01 ~ ai3d_23,每个约 56–60MB),官方提供的「高精度 T-pose 角色 glb 模型」,是复现 OmniFaceRig 的**主要输入资源** |
| `glb/manifest.csv` | glb 模型清单:序号、文件名、SHA256、字节数、创建时间 |
| `2d image/` | 12 张 2D T-pose 角色图(png/jpg):Hulk、Elon、Loki、Superman、Jobs、关羽、Mbappé、monk、ai-image、cyberpunk_tpose 等 —— 交付物 2「上传一张图」的**标准输入样例** |
| `FINAL_WORK_DEMO.glb.gz` | 官方「现有方案」的最终演示产物(glTF 2.0 binary,原文件 117MB 超 GitHub 100MB 单文件上限,已 gzip 压缩为 78MB)。**使用前解压**:`gunzip FINAL_WORK_DEMO.glb.gz` |
| `FINAL DEMO.mov` | 19MB 演示视频(现有方案的最终效果) |
| `a100_server_EN.docx` | **A100 服务器访问信息**(实例 ubuntu-11901、公网 IP、SSH 端口 22217、账号密码、端口映射 32170→8000 / 32171→8001) |

## ⚠️ 安全提醒

`a100_server_EN.docx` 含**明文服务器账号密码**。本仓库为私有仓库,已按用户要求入库;建议:
- 不要转公开仓库;若未来要公开,先删除此文件并轮换服务器密码
- 服务器密码建议 challenge 结束后轮换,或改用 SSH key 登录

## 对复现的意义

- 官方资源 3 项齐了:A100 服务器 ✅、T-pose glb + 2D 图 ✅、现有方案代码走读(待 TA 安排)
- `glb/` + `2d image/` 构成天然的输入-输出对,可用于:
  1. 验证「图 → mesh」前端(TRELLIS 等)与官方 glb 的差异
  2. 用官方 glb 直接测试 Stage 1/Stage 2 几何管线(跳过 image-to-3D 环节)
  3. `FINAL_WORK_DEMO.glb.gz` 作为「现有方案」输出标杆,对照验收口径
- 注意:manifest.csv 里的 SHA256 可用于校验文件完整性
