"""``gib pack``: the source zip, built reproducibly from a git commit.

The zip holds exactly the files of one commit (what git tracks, so .gitignore
decides), read from git's object store rather than the working tree: uncommitted
edits and per-machine line-ending conversion cannot leak in. Paths are sorted and
every entry gets the commit's timestamp and a fixed mode (0644, or 0755 for
executables), so the same commit gives the same zip bytes (with the same zlib).
``content_sha256`` hashes only paths, modes and contents, and is the same on
every machine.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git(root: Path, *args: str, stdin: Optional[bytes] = None) -> bytes:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True, check=True)
    except FileNotFoundError:
        raise SystemExit("gib pack needs git on PATH")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"git {' '.join(args)} failed: {e.stderr.decode(errors='replace').strip()}")
    return r.stdout


def _blobs(root: Path, shas: List[str]) -> Dict[str, bytes]:
    """Contents of many blobs from one ``git cat-file --batch`` call."""
    out = _git(root, "cat-file", "--batch", stdin="".join(s + "\n" for s in shas).encode())
    res: Dict[str, bytes] = {}
    i = 0
    for _ in shas:
        nl = out.index(b"\n", i)
        sha, _kind, size = out[i:nl].decode().split()
        res[sha] = out[nl + 1:nl + 1 + int(size)]
        i = nl + 1 + int(size) + 1  # each content is followed by a newline
    return res


def pack(root: str | Path, out: Optional[str | Path] = None, ref: str = "HEAD",
         prefix: str = "gib-200k") -> Dict[str, Any]:
    root = Path(_git(Path(root), "rev-parse", "--show-toplevel").decode().strip())
    commit = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
    stamp = int(_git(root, "show", "-s", "--format=%ct", commit).decode().strip())
    date_time = max(time.gmtime(stamp)[:6], (1980, 1, 1, 0, 0, 0))  # ZIP cannot store earlier dates
    entries = []
    for rec in _git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if rec:
            meta, path = rec.split(b"\t", 1)
            mode, kind, sha = meta.decode().split()
            if kind == "blob":  # submodules are not packed
                entries.append((path.decode("utf-8"), mode, sha))
    entries.sort()
    data = _blobs(root, [sha for _, _, sha in entries])

    out = Path(out) if out else root / "dist" / f"{prefix}-{commit[:10]}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    content = hashlib.sha256()
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        for path, mode, sha in entries:
            zi = zipfile.ZipInfo(f"{prefix}/{path}", date_time=date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if mode == "100755" else 0o644) << 16
            zi.create_system = 3  # "Unix" whichever OS builds the zip, so the bytes do not depend on it
            z.writestr(zi, data[sha], compresslevel=9)
            content.update(f"{path}\0{mode}\0".encode() + hashlib.sha256(data[sha]).digest())
    tmp.replace(out)
    dirty = bool(_git(root, "status", "--porcelain").strip())
    res = {"zip": str(out), "commit": commit, "files": len(entries), "bytes": out.stat().st_size,
           "zip_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "content_sha256": content.hexdigest(),
           "uncommitted_changes": dirty}
    if dirty:
        res["note"] = f"the zip holds commit {commit[:10]} only; commit your changes first to include them"
    return res
