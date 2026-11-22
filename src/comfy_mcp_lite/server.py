"""Thin MCP server for a local ComfyUI.

标准库打 ComfyUI 的 HTTP API，不依赖 comfy-cli，不去管 ComfyUI 进程的启停，
也不会碰 ComfyUI 自己的 Python 环境。

环境变量
  COMFY_URL         ComfyUI 地址，默认 http://127.0.0.1:8188
  COMFY_PUBLIC_URL  返回给 agent 的分享链接前缀，默认同 COMFY_URL
  COMFY_OUTPUT_DIR  输出目录；不填则依次尝试 $COMFY_DIR/output 和 ~/ComfyUI/output
  COMFY_*_MODEL     覆盖预设里的模型文件名，见 PRESETS
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

__version__ = "0.1.0"

BASE = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
PUBLIC = os.environ.get("COMFY_PUBLIC_URL", BASE).rstrip("/")


def _default_output_dir() -> str:
    for cand in (
        os.environ.get("COMFY_OUTPUT_DIR"),
        os.path.join(os.environ["COMFY_DIR"], "output") if os.environ.get("COMFY_DIR") else None,
        os.path.expanduser("~/ComfyUI/output"),
    ):
        if cand and os.path.isdir(cand):
            return cand
    return os.environ.get("COMFY_OUTPUT_DIR") or os.path.expanduser("~/ComfyUI/output")


OUT_DIR = _default_output_dir()


mcp = MCPServer(
    "comfyui",
    instructions=(
        "本地 ComfyUI。出图首选 comfy_generate（预设已调好采样参数）；"
        "复杂图用 comfy_run_workflow 传 ComfyUI API 格式工作流；"
        "先调 comfy_server_info 确认服务在跑。显存不足时先 comfy_unload_models。"
    ),
)


# ------------------------------------------------------------------ HTTP 客户端
def _req(method: str, path: str, body: Any = None, timeout: float = 60.0) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ComfyUI 返回 {e.code} {path}: {e.read()[:400]!r}") from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"连不上 ComfyUI ({BASE}): {e}. 请先启动 ComfyUI，或用 COMFY_URL 指定正确地址"
        ) from None
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", "replace")


def _upload(path: str) -> dict:
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        blob = f.read()
    parts = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{os.path.basename(path)}"\r\nContent-Type: application/octet-stream\r\n\r\n'
    ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/upload/image", data=parts,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _obj(x: Any) -> Any:
    """工具统一返回对象，交给 SDK 序列化，避免字符串再套一层引号。"""
    return json.loads(x) if isinstance(x, str) else x


# ------------------------------------------------------------------ 工作流预设
def _check_model(kind: str, fname: str, folder: str) -> None:
    """模型不在盘上就早点报错，别把工作流丢进队列再失败。"""
    try:
        have = _req("GET", "/models/" + folder)
    except RuntimeError:
        return
    if fname not in have:
        raise RuntimeError(
            f"预设需要 {kind} 模型 {fname!r}，但本机 {folder}/ 里没有。可用: {have}。"
            f" 下载后重试，或用 comfy_run_workflow 自带模型名，或用环境变量改预设指向。")


def _wf_zimage(prompt, negative, w, h, steps, cfg, sampler, scheduler, seed, prefix, batch):
    unet = os.environ.get("COMFY_ZIMAGE_MODEL", "z_image_turbo_bf16.safetensors")
    clip = os.environ.get("COMFY_ZIMAGE_CLIP", "qwen_3_4b_fp8_mixed.safetensors")
    vae = os.environ.get("COMFY_ZIMAGE_VAE", "ae.safetensors")
    _check_model("diffusion", unet, "diffusion_models")
    _check_model("text encoder", clip, "text_encoders")
    _check_model("vae", vae, "vae")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.0, "model": ["1", 0]}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "lumina2"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["3", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
                "model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["7", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["9", 0]}},
    }


def _ckpt_wf(ckpt_default: str, env: str):
    """单文件 checkpoint 类预设（Flux schnell / SDXL）的公共骨架。"""
    def build(prompt, negative, w, h, steps, cfg, sampler, scheduler, seed, prefix, batch):
        ckpt = os.environ.get(env, ckpt_default)
        _check_model("checkpoint", ckpt, "checkpoints")
        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
            "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": batch}},
            "5": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
                    "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["6", 0]}},
        }
    return build


# 名称: (构造器, steps, cfg, sampler, scheduler, 说明)
PRESETS: dict[str, tuple] = {
    "zimage": (_wf_zimage, 8, 1.0, "res_multistep", "simple",
               "Z-Image Turbo：中文长描述效果好，cfg 固定 1，8 步"),
    "flux_schnell": (_ckpt_wf("flux1-schnell-fp8.safetensors", "COMFY_FLUX_MODEL"),
                     4, 1.0, "euler", "simple", "Flux schnell，4 步 cfg 1"),
    "sdxl": (_ckpt_wf("sd_xl_base_1.0.safetensors", "COMFY_SDXL_MODEL"),
             25, 6.0, "euler", "normal", "SDXL base，配 lcm-lora 时可降到 6-8 步 cfg 1"),
}


# ------------------------------------------------------------------ 任务与产出
def _submit(wf: dict) -> str:
    r = _req("POST", "/prompt", {"prompt": wf, "client_id": "mcp"})
    if "prompt_id" not in r:
        raise RuntimeError(f"提交失败: {r}")
    return r["prompt_id"]


def _wait(prompt_id: str, timeout: float) -> dict:
    t0 = time.time()
    while True:
        h = _req("GET", "/history/" + prompt_id)
        if prompt_id in h:
            return h[prompt_id]
        if time.time() - t0 > timeout:
            raise RuntimeError(f"等待超时（{timeout:.0f}s），任务 {prompt_id} 仍在队列里，可用 comfy_wait 继续等")
        time.sleep(1.0)


def _finish(entry: dict, secs: float) -> dict:
    st = entry.get("status", {})
    out: dict[str, Any] = {"status": st.get("status_str", "unknown"), "seconds": round(secs, 1)}
    if out["status"] != "success":
        details = []
        for msg in st.get("messages", []):
            if isinstance(msg, list) and len(msg) > 1 and isinstance(msg[1], dict):
                d = msg[1]
                details.append({"node": d.get("node_type"), "error": str(d.get("exception_message", ""))[:400]})
        out["failure"] = details or [st.get("status_str")]
        return out
    files = []
    for _nid, o in entry.get("outputs", {}).items():
        for key in ("images", "gifs", "audio"):
            for item in o.get(key, []) or []:
                sub, name = item.get("subfolder", ""), item.get("filename", "")
                local = os.path.join(OUT_DIR, sub, name)
                q = urllib.parse.quote
                files.append({
                    "name": name,
                    "path": local if os.path.exists(local) else None,
                    "bytes": os.path.getsize(local) if os.path.exists(local) else 0,
                    "url": f"{BASE}/view?filename={q(name)}&subfolder={q(sub)}&type=output",
                    "public_url": f"{PUBLIC}/view?filename={q(name)}&subfolder={q(sub)}&type=output",
                })
    out["files"] = files
    return out


async def _call(fn: Callable[[], Any]) -> Any:
    """在线程里跑阻塞 HTTP，异常转成可读的 error 字段而不是裸抛。"""
    try:
        return _obj(await asyncio.to_thread(fn))
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ 工具
@mcp.tool(description="ComfyUI 健康与荷载：版本、空闲显存(GB)、内存、队列长度。先调它确认服务活着。")
async def comfy_server_info() -> Any:
    def go():
        s = _req("GET", "/system_stats")
        q = _req("GET", "/queue")
        dev = s["devices"][0]
        free = dev["vram_free"] / 2**30
        info = {
            "comfyui_version": s["system"]["comfyui_version"],
            "device": dev["name"],
            "vram_free_gb": round(free, 1),
            "vram_total_gb": round(dev["vram_total"] / 2**30, 1),
            "ram_free_gb": round(s["system"]["ram_free"] / 2**30, 1),
            "queue_running": len(q.get("queue_running", [])),
            "queue_pending": len(q.get("queue_pending", [])),
        }
        if free < 40:
            info["note"] = "空闲显存偏小，换模型前先 comfy_unload_models，或降低分辨率/用 tiled VAE"
        return info
    return await _call(go)


@mcp.tool(description="列出某模型文件夹里已有的文件。folder: checkpoints|diffusion_models|text_encoders|vae|loras|clip|upscale_models|... 不传则列全部文件夹。")
async def comfy_list_models(folder: str | None = None) -> Any:
    return await _call(lambda: _req("GET", "/models/" + folder) if folder else _req("GET", "/models"))


@mcp.tool(description="按关键字搜索当前 ComfyUI 可用的节点类型（含自定义节点）。用于自己拼工作流前查节点名。")
async def comfy_search_nodes(query: str, limit: int = 25) -> Any:
    def go():
        info = _req("GET", "/object_info", timeout=90)
        q = query.lower()
        hits = []
        for name, spec in info.items():
            desc = spec.get("description") or ""
            if q in name.lower() or q in desc.lower():
                inputs = ",".join((spec.get("input", {}).get("required", {}) or {}).keys())
                hits.append({"node": name, "category": spec.get("category", ""),
                             "required_inputs": inputs[:200]})
            if len(hits) >= limit:
                break
        return {"total_nodes": len(info), "matched": len(hits), "query": query, "hits": hits}
    return await _call(go)


@mcp.tool(description=(
    "出图（推荐入口）。preset: zimage(默认) | flux_schnell | sdxl，参数已按各自官方模板调好。"
    "wait=false 时立刻返回 prompt_id，之后用 comfy_wait 收结果。"
    "分辨率过高导致 VAE 解码爆显存时，改用 comfy_run_workflow + VAEDecodeTiled。"))
async def comfy_generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int | None = None,
    cfg: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    negative: str = "",
    seed: int | None = None,
    batch: int = 1,
    preset: str = "zimage",
    filename_prefix: str = "mcp",
    wait: bool = True,
    timeout: float = 600,
) -> Any:
    if preset not in PRESETS:
        return {"error": f"未知 preset {preset}", "可用": list(PRESETS)}
    build, d_steps, d_cfg, d_samp, d_sched, note = PRESETS[preset]
    wf = build(
        prompt, negative, width, height,
        steps or d_steps, d_cfg if cfg is None else cfg,
        sampler or d_samp, scheduler or d_sched,
        seed if seed is not None else int.from_bytes(os.urandom(4), "big"),
        filename_prefix, batch,
    )

    def go():
        pid = _submit(wf)
        if not wait:
            return {"prompt_id": pid, "preset": preset, "note": "用 comfy_wait 取结果"}
        t0 = time.time()
        return _finish(_wait(pid, timeout), time.time() - t0) | {"prompt_id": pid, "preset": preset, "preset_note": note}
    return await _call(go)


@mcp.tool(description="提交任意 ComfyUI API 格式工作流（/prompt 接口那个 JSON，dict 或字符串都行），并等结果。")
async def comfy_run_workflow(workflow: str | dict, wait: bool = True, timeout: float = 600) -> Any:
    def go():
        wf = json.loads(workflow) if isinstance(workflow, str) else workflow
        if "prompt" in wf and len(wf) == 1:      # 允许直接粘 UI 导出里的 prompt 段
            wf = wf["prompt"]
        pid = _submit(wf)
        if not wait:
            return {"prompt_id": pid}
        t0 = time.time()
        return _finish(_wait(pid, timeout), time.time() - t0) | {"prompt_id": pid}
    return await _call(go)


@mcp.tool(description="等一个已提交任务完成（配合 comfy_generate 的 wait=false，或超时后接着等）。")
async def comfy_wait(prompt_id: str, timeout: float = 600) -> Any:
    def go():
        t0 = time.time()
        return _finish(_wait(prompt_id, timeout), time.time() - t0) | {"prompt_id": prompt_id}
    return await _call(go)


@mcp.tool(description="队列与最近任务状态（含每张已产出图的预览文件名）。")
async def comfy_queue(limit: int = 10) -> Any:
    def go():
        q = _req("GET", "/queue")
        try:
            jobs = _req("GET", f"/api/jobs?limit={int(limit)}").get("jobs", [])
        except RuntimeError:      # 旧版 ComfyUI 没有 /api/jobs
            jobs = []
        return {
            "running": len(q.get("queue_running", [])),
            "pending": len(q.get("queue_pending", [])),
            "recent": [{"id": j["id"][:8], "status": j.get("status"),
                        "preview": (j.get("preview_output") or {}).get("filename"),
                        "seconds": round((j.get("execution_end_time", 0) - j.get("execution_start_time", 0)) / 1000, 1)}
                       for j in jobs[:limit]],
        }
    return await _call(go)


@mcp.tool(description="取消任务。prompt_id 留空 = 打断当前正在跑的那张。")
async def comfy_cancel(prompt_id: str | None = None) -> Any:
    def go():
        if prompt_id:
            try:
                _req("POST", f"/api/jobs/{prompt_id}/cancel")
                return {"cancelled": prompt_id}
            except Exception as e:  # noqa: BLE001
                _req("POST", "/queue", {"delete": [prompt_id]})
                return {"cancelled": prompt_id, "via": "queue delete", "note": str(e)[:200]}
        _req("POST", "/interrupt")
        return {"cancelled": "当前任务已打断"}
    return await _call(go)


@mcp.tool(description="卸载显存里缓存的模型（换大模型前用）。")
async def comfy_unload_models() -> Any:
    def go():
        _req("POST", "/free", {})
        return {"vram_free_gb_after": round(_req("GET", "/system_stats")["devices"][0]["vram_free"] / 2**30, 1)}
    return await _call(go)


@mcp.tool(description="上传本地图片给工作流用（图生图/参考图），返回 LoadImage 需要的 name。")
async def comfy_upload_image(path: str) -> Any:
    def go():
        if not os.path.exists(path):
            return {"error": f"文件不存在: {path}"}
        r = _upload(path)
        r["load_image_name"] = r.get("name")
        return r
    return await _call(go)


@mcp.tool(description="列出输出目录里最新的图（按时间倒序）。output_dir 默认取 COMFY_OUTPUT_DIR / ~/ComfyUI/output。")
async def comfy_list_outputs(limit: int = 20, prefix: str | None = None, output_dir: str | None = None) -> Any:
    def go():
        root_dir = output_dir or OUT_DIR
        if not os.path.isdir(root_dir):
            return {"error": f"输出目录不存在: {root_dir}", "hint": "传 output_dir 或设 COMFY_OUTPUT_DIR"}
        items = []
        for root, _dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif")):
                    if prefix and not f.startswith(prefix):
                        continue
                    p = os.path.join(root, f)
                    items.append({"path": p, "bytes": os.path.getsize(p),
                                  "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))})
        items.sort(key=lambda x: x["mtime"], reverse=True)
        return {"dir": root_dir, "count": len(items), "latest": items[:limit]}
    return await _call(go)


def main() -> None:
    ap = argparse.ArgumentParser(prog="comfyui-mcp-lite", description="Thin MCP server for a local ComfyUI (stdio).")
    ap.add_argument("--version", action="version", version=f"comfyui-mcp-lite {__version__}")
    ap.add_argument("--comfy-url", help="覆盖 COMFY_URL")
    args = ap.parse_args()
    global BASE, PUBLIC
    if args.comfy_url:
        BASE = args.comfy_url.rstrip("/")
        PUBLIC = os.environ.get("COMFY_PUBLIC_URL", BASE).rstrip("/")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
