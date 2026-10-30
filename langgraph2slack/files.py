"""Reads files attached to Slack messages so the agent can see them.

- **Images** (PNG, JPEG, WebP) reach the agent as images, up to a size cap per image and a
  cap on images per run. The image type is checked against the file's bytes.
- **Small text-like files** (logs, code, CSV, JSON, ...) are inlined, up to a character cap.
- **Everything else** (large or binary files, GIFs) is listed by name, type, size and ID.
- **Files the bridge can't read** (deleted, external, shared from another workspace, or the
  ``files:read`` scope missing) are noted and skipped; they never fail the run.

File contents are untrusted data: the framer places them in an untrusted section.
"""

import asyncio
import base64
import dataclasses
import logging
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

FileKind = Literal["image", "text", "reference", "unreadable"]

# Image types every major model provider accepts (GIF is not accepted everywhere)
IMAGE_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/webp": (b"RIFF",),  # followed by size and "WEBP"
}
TEXT_MIMETYPES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/javascript",
    "application/x-sh",
    "application/sql",
    "application/toml",
}
TEXT_FILETYPES = {
    "text", "csv", "tsv", "json", "yaml", "yml", "xml", "markdown", "md", "log", "diff",
    "patch", "toml", "ini", "sql", "shell", "bash", "python", "javascript", "typescript",
    "go", "rust", "java", "kotlin", "c", "cpp", "csharp", "ruby", "php", "swift", "html",
    "css", "dockerfile",
}  # fmt: skip

# At most this many files are read per message; the rest are listed by reference
MAX_FILES_PER_MESSAGE = 10
# Hosts Slack serves private files from; anything else is a sign-in redirect
FILE_HOSTS = ("files.slack.com",)

# What a file counts for against the backfill character budget
IMAGE_COST = 1000
REFERENCE_COST = 80


class FileUnavailable(Exception):
    """A file the bridge can't read; the message says why."""


@dataclass(frozen=True)
class FilePart:
    """A Slack file as the agent receives it."""

    id: str
    name: str
    mimetype: str
    size: Optional[int]
    kind: FileKind
    text: Optional[str] = None  # inlined content, for text files
    data_url: Optional[str] = None  # base64 data URL, for images
    note: Optional[str] = None  # why a file was listed or skipped

    @property
    def is_inlined(self) -> bool:
        return self.kind in ("text", "image")

    def describe(self) -> str:
        """'name (type, size, ID)'."""
        size = _human_size(self.size) if self.size is not None else "unknown size"
        return f"{self.name} ({self.mimetype or 'unknown type'}, {size}, {self.id})"

    def cost(self) -> int:
        """What the file counts for against the backfill character budget."""
        if self.kind == "text":
            return len(self.text or "")
        return IMAGE_COST if self.kind == "image" else REFERENCE_COST

    def as_reference(self, note: str) -> "FilePart":
        return dataclasses.replace(self, kind="reference", text=None, data_url=None, note=note)

    def as_unreadable(self, note: str) -> "FilePart":
        return dataclasses.replace(self, kind="unreadable", text=None, data_url=None, note=note)


@dataclass(frozen=True)
class FileLimits:
    """How much of a file the agent may receive."""

    inline_images: bool = True
    max_image_bytes: int = 2_000_000
    max_images: int = 4
    max_inline_text_chars: int = 20_000


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=30)


async def download(client: httpx.AsyncClient, url: str, token: str, max_bytes: int) -> bytes:
    """Download a private Slack file with the bot token, reading at most ``max_bytes``.

    Raises FileUnavailable if the file can't be read or is larger than ``max_bytes``.
    """
    try:
        async with client.stream(
            "GET", url, headers={"Authorization": f"Bearer {token}"}
        ) as response:
            response.raise_for_status()
            # Without the files:read scope Slack redirects to its sign-in page
            if urlparse(str(response.url)).hostname not in FILE_HOSTS:
                raise FileUnavailable("could not be downloaded (missing files:read scope?)")
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise FileUnavailable("too large")
            return bytes(content)
    except FileUnavailable:
        raise
    except Exception as e:
        logger.warning(f"Could not download Slack file {url}: {e}")
        raise FileUnavailable("could not be downloaded") from e


class SlackFileReader:
    def __init__(self, token: str, limits: FileLimits = FileLimits()):
        self._token = token
        self.limits = limits

    async def read(
        self, files: list[dict], *, budget_chars: Optional[int] = None
    ) -> tuple[FilePart, ...]:
        """Read a message's files, concurrently.

        ``budget_chars`` (backfill) lists files that can't fit by reference, before
        downloading them.
        """
        async with _new_client() as client:
            reads = [
                (
                    self._read_one(client, file, budget_chars)
                    if index < MAX_FILES_PER_MESSAGE
                    else _done(
                        _placeholder(file).as_reference("not read: too many files on one message")
                    )
                )
                for index, file in enumerate(files or [])
            ]
            return tuple(await asyncio.gather(*reads))

    async def _read_one(
        self, client: httpx.AsyncClient, file: dict, budget_chars: Optional[int]
    ) -> FilePart:
        part = _placeholder(file)
        if file.get("file_access") == "check_file_info":
            return part.as_unreadable("shared from another workspace: not available")
        if file.get("mode") in ("tombstone", "hidden_by_limit") or file.get("is_external"):
            return part.as_unreadable("not available to the bridge (deleted, external or hidden)")
        url = file.get("url_private_download") or file.get("url_private")
        if not url:
            return part.as_unreadable("no download URL")

        limits = self.limits
        try:
            if part.mimetype in IMAGE_SIGNATURES:
                if not limits.inline_images:
                    return part.as_reference("image inlining is off")
                if (part.size or 0) > limits.max_image_bytes:
                    return part.as_reference("too large to attach as an image")
                if budget_chars is not None and budget_chars < IMAGE_COST:
                    return part.as_reference("not inlined: backfill budget")
                content = await download(client, url, self._token, limits.max_image_bytes)
                if not _matches_signature(part.mimetype, content):
                    return part.as_reference("not attached: content doesn't match its type")
                data = base64.b64encode(content).decode("ascii")
                return dataclasses.replace(
                    part, kind="image", data_url=f"data:{part.mimetype};base64,{data}"
                )

            if _is_text_like(file):
                # UTF-8 text is at most 4 bytes per character: skip downloads that can't fit
                max_chars = limits.max_inline_text_chars
                if budget_chars is not None:
                    max_chars = min(max_chars, budget_chars)
                    if (part.size or 0) > max_chars * 4:
                        return part.as_reference("not inlined: backfill budget")
                if (part.size or 0) > max_chars * 4:
                    return part.as_reference("too large to inline")
                content = await download(client, url, self._token, max_chars * 4)
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    return part.as_reference("binary content")
                if len(text) > max_chars:
                    return part.as_reference("too large to inline")
                return dataclasses.replace(part, kind="text", text=text)
        except FileUnavailable as e:
            if str(e) == "too large":
                return part.as_reference("too large to inline")
            return part.as_unreadable(str(e))

        return part.as_reference("not a text or supported image file")


def cap_images(files: list[FilePart], max_images: int) -> list[FilePart]:
    """Keep at most ``max_images`` inlined images (in order); list the rest by reference."""
    kept = 0
    capped = []
    for file in files:
        if file.kind == "image":
            if kept >= max_images:
                file = file.as_reference("not attached: image limit for one request reached")
            else:
                kept += 1
        capped.append(file)
    return capped


async def _done(part: FilePart) -> FilePart:
    return part


def _placeholder(file: dict) -> FilePart:
    """The file as a reference, before anything is read."""
    return FilePart(
        id=file.get("id", "unknown"),
        name=file.get("name") or file.get("title") or file.get("id", "file"),
        mimetype=file.get("mimetype", ""),
        size=file.get("size"),
        kind="reference",
    )


def _matches_signature(mimetype: str, content: bytes) -> bool:
    if mimetype == "image/webp":
        return content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    return any(content.startswith(sig) for sig in IMAGE_SIGNATURES[mimetype])


def _is_text_like(file: dict) -> bool:
    mimetype = file.get("mimetype") or ""
    return (
        mimetype.startswith("text/")
        or mimetype in TEXT_MIMETYPES
        or (file.get("filetype") or "").lower() in TEXT_FILETYPES
        or file.get("mode") == "snippet"
    )


def _human_size(size: float) -> str:
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KB", "MB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size / 1024:.1f} GB"
