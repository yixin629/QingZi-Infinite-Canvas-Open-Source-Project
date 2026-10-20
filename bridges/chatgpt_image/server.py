"""ChatGPT 网页出图桥接（本机自用）。

对外：OpenAI 兼容的 POST /v1/images/generations（返回 b64_json）。
对内：Playwright 驱动一个独立的真实 Chrome 窗口（持久化用户目录），
复用你手动登录过的 ChatGPT 会话，在网页里出图后把图片取回。

不做任何隐身 / 反检测：窗口可见，遇到登录或人机验证由你在窗口里手动处理。
一次只跑一张（串行锁），适合低频自用。

启动：python -m bridges.chatgpt_image.server   （或双击 start-chatgpt-bridge.bat）
画布：MyApi 配置 → 图片 → 添加模型
      Base URL  http://127.0.0.1:7960   模型名 chatgpt-web   Key 随便填
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from playwright.async_api import BrowserContext, Page, async_playwright

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = Path(os.environ.get("CHATGPT_BRIDGE_PROFILE", ROOT / "data" / "chatgpt_bridge_profile"))
HOST = "127.0.0.1"
PORT = int(os.environ.get("CHATGPT_BRIDGE_PORT", "7960"))
CHAT_URL = os.environ.get("CHATGPT_BRIDGE_URL", "https://chatgpt.com/")
GEN_TIMEOUT_S = int(os.environ.get("CHATGPT_BRIDGE_TIMEOUT", "360"))
READY_TIMEOUT_S = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(message)s")
log = logging.getLogger("chatgpt_bridge")

# 1x1 透明 PNG：画布「测试连接」发 prompt=ping 时直接返回，不真的出图。
_PING_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# 生成图片在网页里的地址特征（上传的参考图在用户消息里，会被单独排除）。
_GEN_IMG_JS = r"""
(before) => {
  const out = [];
  for (const img of document.querySelectorAll('main img')) {
    const src = img.currentSrc || img.src || '';
    if (!src || before.includes(src)) continue;
    if (!/oaiusercontent|estuary|backend-api\/files/.test(src)) continue;
    if (img.closest('[data-message-author-role="user"]')) continue;
    if (!img.complete || img.naturalWidth < 256) continue;
    out.push({src, w: img.naturalWidth, h: img.naturalHeight});
  }
  return out;
}
"""

_ALL_IMG_SRCS_JS = "() => Array.from(document.querySelectorAll('main img')).map(i => i.currentSrc || i.src || '')"

_LAST_ASSISTANT_TEXT_JS = r"""
() => {
  const els = document.querySelectorAll('[data-message-author-role="assistant"]');
  return els.length ? (els[els.length - 1].innerText || '').trim() : '';
}
"""


class BridgeError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class ChatGPTWeb:
    def __init__(self) -> None:
        self._pw = None
        self._ctx: BrowserContext | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self._pw = await async_playwright().start()
        self._ctx = await self._pw.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            no_viewport=True,
            accept_downloads=True,
        )
        page = self._ctx.pages[0] if self._ctx.pages else await self._ctx.new_page()
        await page.goto(CHAT_URL, wait_until="domcontentloaded")
        log.info("Chrome 已打开 %s —— 第一次请在这个窗口里手动登录 ChatGPT", CHAT_URL)

    async def stop(self) -> None:
        if self._ctx:
            await self._ctx.close()
        if self._pw:
            await self._pw.stop()

    async def generate(self, prompt: str, ratio: str, ref_files: list[Path]) -> bytes:
        if self._ctx is None:
            raise BridgeError("浏览器未启动", 503)
        async with self._lock:
            page = await self._ctx.new_page()
            try:
                return await self._generate_on(page, prompt, ratio, ref_files)
            finally:
                await page.close()

    async def _generate_on(self, page: Page, prompt: str, ratio: str, ref_files: list[Path]) -> bytes:
        await page.goto(CHAT_URL, wait_until="domcontentloaded")
        composer = page.locator("#prompt-textarea")
        try:
            await composer.wait_for(state="visible", timeout=READY_TIMEOUT_S * 1000)
        except Exception as exc:
            raise BridgeError(
                "ChatGPT 输入框没出现：请到桥接打开的 Chrome 窗口里登录或完成验证后重试", 503
            ) from exc

        if ref_files:
            await self._attach(page, ref_files)

        before = await page.evaluate(_ALL_IMG_SRCS_JS)
        await composer.click()
        await page.keyboard.insert_text(_wrap_prompt(prompt, ratio, bool(ref_files)))
        await self._send(page)
        log.info("已发送，等待出图（最长 %ss）", GEN_TIMEOUT_S)

        src = await self._wait_image(page, before)
        return await self._download(page, src)

    async def _attach(self, page: Page, files: list[Path]) -> None:
        inputs = page.locator('input[type="file"]')
        count = await inputs.count()
        target = None
        for i in range(count):
            accept = (await inputs.nth(i).get_attribute("accept")) or ""
            if not accept or "image" in accept:
                target = inputs.nth(i)
                break
        if target is None:
            raise BridgeError("找不到 ChatGPT 的上传入口，网页可能改版了")
        await target.set_input_files([str(p) for p in files])
        # 等上传完成：发送按钮可点
        send = page.locator('[data-testid="send-button"]')
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if await send.count() and await send.is_enabled():
                return
            await asyncio.sleep(1)
        raise BridgeError("参考图上传超时")

    async def _send(self, page: Page) -> None:
        send = page.locator('[data-testid="send-button"]')
        try:
            await send.click(timeout=10_000)
        except Exception:
            await page.keyboard.press("Enter")

    async def _wait_image(self, page: Page, before: list[str]) -> str:
        deadline = time.monotonic() + GEN_TIMEOUT_S
        stop_btn = page.locator('[data-testid="stop-button"]')
        last_pick = ""
        stable = 0
        idle_since: float | None = None
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            streaming = await stop_btn.count() > 0
            found = await page.evaluate(_GEN_IMG_JS, before)
            if found and not streaming:
                pick = max(found, key=lambda x: x["w"] * x["h"])["src"]
                stable = stable + 1 if pick == last_pick else 0
                last_pick = pick
                if stable >= 1:  # 连续两次一致，认为渐进渲染已结束
                    return pick
                continue
            if not streaming:
                idle_since = idle_since or time.monotonic()
                # 回复已结束但没有图：多半是拒绝 / 额度用完 / 它在提问
                if time.monotonic() - idle_since > 25:
                    text = await page.evaluate(_LAST_ASSISTANT_TEXT_JS)
                    raise BridgeError(f"ChatGPT 没有出图，回复：{text[:300] or '（空）'}")
            else:
                idle_since = None
        raise BridgeError("出图超时", 504)

    async def _download(self, page: Page, src: str) -> bytes:
        if src.startswith("data:"):
            return base64.b64decode(src.split(",", 1)[1])
        resp = await page.context.request.get(src)
        if resp.ok:
            return await resp.body()
        log.warning("直接下载失败 HTTP %s，改为截图元素", resp.status)
        img = page.locator(f'main img[src="{src}"]').first
        return await img.screenshot(type="png")


def _wrap_prompt(prompt: str, ratio: str, has_refs: bool) -> str:
    lines = ["请直接调用图像生成画一张图，不要提问、不要解释、不要给多个方案。"]
    if ratio:
        lines.append(f"画面比例 {ratio}。")
    if has_refs:
        lines.append("严格参考我上传的图片里的人物长相、服装和画风，保持一致。")
    return "\n".join(lines) + "\n\n" + prompt.strip()


def _size_to_ratio(size: str | None) -> str:
    m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", size or "")
    if not m:
        return ""
    w, h = int(m.group(1)), int(m.group(2))
    r = w / h
    for label, val in (("1:1", 1), ("16:9", 16 / 9), ("9:16", 9 / 16), ("4:3", 4 / 3), ("3:4", 3 / 4),
                       ("3:2", 3 / 2), ("2:3", 2 / 3), ("21:9", 21 / 9)):
        if abs(r - val) < 0.06:
            return label
    return f"{w}:{h}"


async def _materialize_refs(images: list[Any], tmp: Path) -> list[Path]:
    out: list[Path] = []
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for i, item in enumerate(images or []):
            url = item if isinstance(item, str) else (item or {}).get("url", "")
            if not url:
                continue
            if url.startswith("data:"):
                head, b64 = url.split(",", 1)
                ext = "png" if "png" in head else "jpg" if ("jpeg" in head or "jpg" in head) else "webp"
                data = base64.b64decode(b64)
            else:
                r = await client.get(url)
                r.raise_for_status()
                data = r.content
                ext = "png" if data[:4] == b"\x89PNG" else "jpg"
            p = tmp / f"ref_{i}.{ext}"
            p.write_bytes(data)
            out.append(p)
    return out


web = ChatGPTWeb()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await web.start()
    yield
    await web.stop()


app = FastAPI(title="ChatGPT Web Image Bridge", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "profile": str(PROFILE_DIR)}


@app.post("/v1/images/generations")
async def images_generations(body: dict = Body(...)):
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": {"message": "prompt 不能为空"}}, status_code=400)
    if prompt.lower() == "ping":
        return {"created": int(time.time()), "data": [{"b64_json": _PING_PNG}]}

    ratio = _size_to_ratio(body.get("size"))
    with tempfile.TemporaryDirectory(prefix="gptbridge_") as td:
        try:
            refs = await _materialize_refs(body.get("images") or [], Path(td))
            log.info("新任务 ratio=%s refs=%d prompt=%s", ratio or "默认", len(refs), prompt[:60])
            png = await web.generate(prompt, ratio, refs)
        except BridgeError as exc:
            log.warning("失败：%s", exc)
            return JSONResponse({"error": {"message": str(exc)}}, status_code=exc.status)
        except Exception as exc:  # 网页改版等意外
            log.exception("意外错误")
            return JSONResponse({"error": {"message": f"桥接异常：{exc}"}}, status_code=500)
    log.info("出图完成 %d KB", len(png) // 1024)
    return {"created": int(time.time()), "data": [{"b64_json": base64.b64encode(png).decode()}]}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
