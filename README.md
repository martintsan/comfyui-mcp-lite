# comfyui-mcp-lite

给本地 ComfyUI 套一层 MCP（Model Context Protocol）server，让任何会说 MCP 的 agent —— pi、Claude Code、Codex、Cursor……——直接在你的机器上出图。

**特点**

- 只用 Python 标准库打 ComfyUI 的 HTTP API，**不依赖 `comfy-cli`**，不接管 ComfyUI 的启停，也不碰 ComfyUI 自己的 Python 环境（ROCm / CUDA venv 都不会被动到）
- 单文件实现，一个环境变量 `COMFY_URL` 就够
- 出图工具内置常用采样参数，agent 不用懂 sampler/scheduler
- 失败时把 ComfyUI 的节点级报错原样返回（哪个节点、哪条异常），agent 能自己纠偏

## 工具（11 个）

| 工具 | 作用 |
|---|---|
| `comfy_server_info` | 版本 / 空闲显存 / 内存 / 队列长度 |
| `comfy_generate` | **主力**：一句 prompt 出图。`preset=zimage\|flux_schnell\|sdxl`；`wait=false` 立即返回 `prompt_id` |
| `comfy_run_workflow` | 提交任意 ComfyUI API 格式工作流（UI 导出里的 `prompt` 段可直接粘） |
| `comfy_wait` | 等一个已提交任务 |
| `comfy_queue` | 队列 + 最近任务（带预览文件名与耗时） |
| `comfy_cancel` | 取消任务，不传 id 就打断当前这张 |
| `comfy_list_models` | 列模型文件夹内容 |
| `comfy_search_nodes` | 按关键字搜已安装节点（含自定义节点） |
| `comfy_upload_image` | 上传本地图给 `LoadImage` 用 |
| `comfy_unload_models` | 卸载显存里缓存的模型 |
| `comfy_list_outputs` | 列输出目录最新文件 |

生成成功返回本地绝对路径 + `url` + `public_url`。

## 安装

需要 Python ≥ 3.10 和 [uv](https://docs.astral.sh/uv/)（`pip install uv`）。ComfyUI 得先跑着。

```bash
uvx --from git+https://github.com/martintsan/comfyui-mcp-lite comfyui-mcp-lite --version
```

## 接入

### pi

最简形式（`COMFY_URL` 默认就是 `http://127.0.0.1:8188`，多数情况下不需要 env）：

```bash
pi mcp add comfy -- uvx --from git+https://github.com/martintsan/comfyui-mcp-lite comfyui-mcp-lite
```

需要传环境变量时加 `--env`（可重复）：

```bash
pi mcp add comfy --env COMFY_URL=http://127.0.0.1:8188 -- \
  uvx --from git+https://github.com/martintsan/comfyui-mcp-lite comfyui-mcp-lite
```

> `Error: Unknown option: --env` —— 你的 pi 版本偏旧（`pi mcp add` 的 `--env`/`--cwd` 是较近版本才有的，1.6.0 可用）。
> 三个办法：① 直接省略 `--env`，用上面的最简形式；② 升级 pi（`pi update`，或 `npm install -g --ignore-scripts @earendil-works/pi-coding-agent@latest`）；
> ③ 不折腾 CLI，手改下面这段 JSON。用 `pi mcp add --help` 看你本机支持哪些选项。

**建议补 `timeout`**（`pi mcp add` 不写这个字段，手动编辑 `~/.pi/agent/mcp.json`）：默认 60s，而冷启动第一次出图常常超过 60s 会被掐。

```json
{"mcpServers": {"comfy": {
  "command": "uvx",
  "args": ["--from", "git+https://github.com/martintsan/comfyui-mcp-lite", "comfyui-mcp-lite"],
  "env": {"COMFY_URL": "http://127.0.0.1:8188"},
  "timeout": 900,
  "description": "本地 ComfyUI 出图"
}}}
```

`pi mcp list` 应该显示 `comfy: connected, 11 tools`。

### Claude Code

```bash
claude mcp add comfy --scope user -e COMFY_URL=http://127.0.0.1:8188 \
  -- uvx --from git+https://github.com/martintsan/comfyui-mcp-lite comfyui-mcp-lite
```

### Cursor / 其它 GUI 客户端

`uvx` 可能不在 GUI 进程的 PATH 里 —— 用绝对路径：

```json
{"mcpServers": {"comfy": {
  "command": "/home/you/.local/bin/uvx",
  "args": ["--from", "git+https://github.com/martintsan/comfyui-mcp-lite", "comfyui-mcp-lite"],
  "env": {"COMFY_URL": "http://127.0.0.1:8188"}
}}}
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | 目标 ComfyUI |
| `COMFY_PUBLIC_URL` | 同 `COMFY_URL` | 返回给 agent 的分享链接前缀（例如你的反向代理 / tailscale serve 地址） |
| `COMFY_OUTPUT_DIR` | `$COMFY_DIR/output` → `~/ComfyUI/output` | 用来把产出解析成本地绝对路径 |
| `COMFY_ZIMAGE_MODEL` / `COMFY_ZIMAGE_CLIP` / `COMFY_ZIMAGE_VAE` | 官方模板同名文件 | 改 zimage 预设指向 |
| `COMFY_FLUX_MODEL` / `COMFY_SDXL_MODEL` | `flux1-schnell-fp8.safetensors` / `sd_xl_base_1.0.safetensors` | 改 checkpoint 预设指向 |

预设里写死的模型文件名只是**默认值**：提交前会查 `/models/<folder>`，找不到就直接报错并列出本机可用的文件，不会把工作流丢进队列再失败。

## 加自己的预设

`src/comfy_mcp_lite/server.py` 里 `PRESETS` 加一条，签名固定：

```python
def _wf_mine(prompt, negative, w, h, steps, cfg, sampler, scheduler, seed, prefix, batch):
    return {...}                      # 直接返回 ComfyUI API 格式工作流

PRESETS["mine"] = (_wf_mine, 20, 3.5, "euler", "simple", "我的默认档")
```

不想改代码就用 `comfy_run_workflow` 传完整工作流。

## 排坑

| 现象 | 处理 |
|---|---|
| `Unknown option: --env` | pi 版本旧，省略 `--env` 或升级 pi 或直接写 JSON（见上） |
| `command not found: uvx` | 先 `pip install uv`（或 `brew install uv`），确认 `which uvx` 有输出；GUI 客户端用绝对路径 |
| 工具返回 `连不上 ComfyUI` | ComfyUI 没跑，或 `COMFY_URL` 不对；`curl -s <COMFY_URL>/system_stats` 先自测 |
| 冷启动那张图报超时 | 把客户端的 MCP 请求超时调大（pi 在 `mcp.json` 里 `"timeout": 900`） |
| 出图失败 | 返回的是 `{"error": "..."}` 而不是抛异常，里面带着 ComfyUI 的节点名和异常文本，对着它改 prompt / 分辨率 / preset |

## 注意

- **跑工作流等于在本机执行任意 custom node 代码**。只把它接给你信任的 agent；不要把 ComfyUI 绑到 `0.0.0.0`（保持在 `127.0.0.1`，需要远程就在前面挂 tailscale serve 或反代）
- stdio server 不能跨机。远端 agent 要么直接打 ComfyUI 的 HTTP API，要么把这份改成 `streamable-http` 传输再自行加鉴权
- 与官方 [`comfy-mcp`](https://github.com/Comfy-Org/comfy-mcp) 的关系：官方那套功能全得多（42 工具，还能管 ComfyUI 启停、装节点），代价是要装 `comfy-cli` 并让它接管你的 ComfyUI 工作区。如果 ComfyUI 是手写脚本或 systemd 托管、且你不想让第二个工具碰它的环境，这个轻量版更省事

## 本地开发

```bash
uv venv .venv && uv pip install -e .
python -c "import comfy_mcp_lite, sys; print(comfy_mcp_lite.__version__)"
comfyui-mcp-lite --version
```

## License

MIT
