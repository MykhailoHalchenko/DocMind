import hashlib
import json
import logging
import re
import tempfile
import urllib.request
from pathlib import Path

_logger = logging.getLogger(__name__)


def extract_drive_file_id(source: str) -> str | None:
    """Extract a Google Drive file ID from a URL or raw ID string."""
    if not source:
        return None

    source = source.strip()
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"^([a-zA-Z0-9_-]{10,})$",
    ]

    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return match.group(1)

    return None


def _infer_drive_kind(source: str) -> str:
    lowered = source.lower()
    if "docs.google.com/document" in lowered:
        return "document"
    if "docs.google.com/spreadsheets" in lowered:
        return "spreadsheet"
    if "docs.google.com/presentation" in lowered:
        return "presentation"
    return "file"


def _safe_stem(value: str | None, fallback: str) -> str:
    raw = (value or fallback).strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or fallback


def _derive_extension(kind: str, file_name: str | None) -> str:
    if file_name:
        suffix = Path(file_name).suffix.lower()
        if suffix:
            return suffix
    if kind == "document":
        return ".txt"
    if kind == "spreadsheet":
        return ".csv"
    if kind == "presentation":
        return ".pdf"
    return ".bin"


def _build_download_url(file_id: str, kind: str) -> tuple[str, str]:
    if kind == "document":
        return f"https://docs.google.com/document/d/{file_id}/export?format=txt", ".txt"
    if kind == "spreadsheet":
        return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv", ".csv"
    if kind == "presentation":
        return f"https://docs.google.com/presentation/d/{file_id}/export?format=pdf", ".pdf"
    return f"https://drive.google.com/uc?export=download&id={file_id}", ".bin"


def _storage_root(storage_root: str | Path | None) -> Path:
    root = Path(storage_root) if storage_root is not None else Path(__file__).resolve().parent
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_existing_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _download_to_tempfile(download_url: str, temp_path: Path) -> tuple[str, int]:
    sha256 = hashlib.sha256()
    total_bytes = 0
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Mozilla/5.0 DocMind/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as destination:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            destination.write(chunk)
            sha256.update(chunk)
            total_bytes += len(chunk)

    return sha256.hexdigest(), total_bytes


def sync_google_drive_source(
    source: str,
    storage_root: str | Path | None = None,
    file_name: str | None = None,
) -> dict:
    """Synchronize a shared/public Google Drive source into local storage.

    The downloaded file is stored under storage/google_drive/<file_id>/ and
    a manifest keeps track of the latest hash so repeat syncs can skip rewrites.
    """
    file_id = extract_drive_file_id(source)
    if not file_id:
        raise ValueError("Could not extract a Google Drive file ID from the provided source")

    kind = _infer_drive_kind(source)
    download_url, inferred_extension = _build_download_url(file_id, kind)
    extension = _derive_extension(kind, file_name) if kind == "file" else inferred_extension
    safe_name = _safe_stem(file_name, file_id)

    root = _storage_root(storage_root)
    sync_dir = root / "google_drive" / file_id
    sync_dir.mkdir(parents=True, exist_ok=True)

    content_path = sync_dir / f"{safe_name}{extension}"
    manifest_path = sync_dir / "manifest.json"
    existing_manifest = _read_existing_manifest(manifest_path)

    with tempfile.NamedTemporaryFile(delete=False, dir=sync_dir, suffix=".download") as temp_handle:
        temp_path = Path(temp_handle.name)

    try:
        content_hash, byte_count = _download_to_tempfile(download_url, temp_path)

        if existing_manifest and existing_manifest.get("sha256") == content_hash and content_path.exists():
            temp_path.unlink(missing_ok=True)
            return {
                "status": "unchanged",
                "file_id": file_id,
                "source": source,
                "kind": kind,
                "dataset_type": extension.lstrip("."),
                "file_path": str(content_path),
                "manifest_path": str(manifest_path),
                "sha256": content_hash,
                "bytes": byte_count,
            }

        temp_path.replace(content_path)
        manifest = {
            "file_id": file_id,
            "source": source,
            "kind": kind,
            "file_name": file_name or safe_name,
            "dataset_type": extension.lstrip("."),
            "file_path": str(content_path),
            "manifest_path": str(manifest_path),
            "sha256": content_hash,
            "bytes": byte_count,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        return {"status": "synced", **manifest}
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def sync_google_drive_sources(
    sources: list[str],
    storage_root: str | Path | None = None,
) -> list[dict]:
    """Synchronize multiple Google Drive sources into storage."""
    results: list[dict] = []
    for source in sources:
        try:
            results.append(sync_google_drive_source(source, storage_root=storage_root))
        except Exception as e:
            _logger.error(f"Google Drive sync failed for {source}: {e}")
            results.append({"status": "error", "source": source, "error": str(e)})
    return results