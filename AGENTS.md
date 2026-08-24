# AGENTS.md — 多代理协作约定(2026-08-25)

本仓库由 **1 个协调者 + 4 个并行 subagent** 协作推进。所有 agent 必须遵守本文件,
防止抢活/冲突。

## 团队

| 代号 | 分支 | 物理目录 | 任务(issue) | 负责范围 | 状态 |
|---|---|---|---|---|---|
| **rig-ops** | `agent/rig-ops` | `/opt/dsh/DSH/Cam-ChallengeTrack/agents/rig-ops` | #1 批量生产与验收 | 服务器 `~/work/outputs` 验收、`code/scripts/batch_rig.py`、`deliverables_check.py` | done-2026-08-25 |
| **inner-mouth** | `agent/inner-mouth` | `/opt/dsh/DSH/Cam-ChallengeTrack/agents/inner-mouth` | #2 口腔内部资产 | `code/omnifacerig_repro/inner_mouth.py`、ICT-FaceKit 获取、口腔几何 | active |
| **trellis-front** | `agent/trellis-front` | `/opt/dsh/DSH/Cam-ChallengeTrack/agents/trellis-front` | #3 图→mesh 前端 | `code/scripts/trellis_front.py`、部署脚本、权重下载方案 | done-2026-08-25 (PR #6,权重 blocker 已解除) |
| **web-ops** | `agent/web-ops` | `/opt/dsh/DSH/Cam-ChallengeTrack/agents/web-ops` | #4 Web 服务运维 | 服务器 webapp、systemd、前端页面 | done-2026-08-25 (PR #5) |
| 协调者 | `main` | 主仓库 | — | review/merge PR、分配任务、更新本文件 | — |

## 硬性规则(必须遵守)

1. **只在自己的物理目录和分支上工作**。永远不要 `git checkout` 别人的分支,
   不要直接 push 到 `main`。交付 = 推自己的分支 + `gh pr create --base main`。
2. **文件所有权**:核心库 `code/omnifacerig_repro/` 与 `code/scripts/` 是共享区,
   但你**只改自己任务清单里点名的文件**;确实需要动别人的文件时,先在 issue 里
   声明"我要改 X",等协调者确认。
3. **服务器权限**:
   - `rig-ops`:`~/work/outputs` 只读验收 + 运行 pipeline 生成验收产物。
   - `web-ops`:`~/work/webapp`、systemd、webapp 进程。
   - `inner-mouth` / `trellis-front`:**不碰服务器**(下载与代码只在本地)。
4. **你还有 3 个 coworkers 同时在干活**。防止抢活的具体做法:
   - 开工前 `git pull --rebase origin main`(或 merge main 进自己分支)获取最新。
   - 任务有交集时(比如都要动 `stage1_real.py`),先在对应 issue 留言认领文件。
   - 你看到的其他分支的中间状态不是最终状态,不要依赖、不要评论批评。
5. **协作通道**:任务追踪用 GitHub issue(`gh issue view <n>`),交付用 PR
   (`gh pr create`),进度更新用 `AGENTS.md` 状态行 + issue comment。
6. **完成/阻塞都要汇报**:PR 描述里写清楚:做了什么、验证结果(命令+输出摘要)、
   blocker(精确到命令与报错)。PR 标题格式:`[agent-代号] 一句话摘要 (closes #N)`。

## 关键事实(避免重复踩坑)

- 仓库 git 直连 github.com:443 被墙;push/pull 必须走 SSH:
  `git config core.sshCommand "ssh -i /opt/dsh/DSH/Cam-ChallengeTrack/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/opt/dsh/DSH/Cam-ChallengeTrack/.ssh/known_hosts -p 443"`
  (各 agent 目录已配好;`gh` CLI 可用且已认证,API 通道不受影响)。
- 服务器:SSH `linux@175.155.64.171:22217`,密码 `M5@cn`
  (用 `python3 /opt/dsh/DSH/Cam-ChallengeTrack/ssh_helper.py run "<cmd>"`)。
  Web 预览:`http://175.155.64.171:32170/`。torch env:`conda activate torch2.4_cuda12.1`。
- **网络限制**:服务器访问 github 22B/s、HF 20-40KB/s;本机 HF 0B/s、hf-mirror 37KB/s。
  大模型权重下载基本不可行 — 需要权重的工作先给方案再尝试,不要死磕。
- 管线现状:官方 glb → rigged glb 已跑通(~30s/角色,52 ARKit 稀疏 morph + 53 Mixamo
  关节 + 中英口型动画);验收 D2/D3/D4 通过(翻转面积 0.036%);web 端到端验证通过。
- 测试:`cd code && python -m pytest ../tests`(28 个,必须在改完代码后全跑)。
- 禁止把 CARV3D 身份数据(vasiliskatr data/)入库。

## 状态更新格式

完成一项 → 在 AGENTS.md 团队表把自己的状态行改为 `done-<日期>` 或注明 blocker,
并在 issue 里留 comment。协调者合并 PR 后会把 main 的最新状态同步到各分支。
