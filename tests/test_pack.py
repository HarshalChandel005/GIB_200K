import shutil
import subprocess
import zipfile

import pytest

from gib.release.pack import pack

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway repo with fixed commit dates, isolated from the user's git configuration."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Test\n\temail = test@example.com\n[core]\n\tautocrlf = false\n",
                   encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):
        monkeypatch.setenv(var, "2026-01-02T03:04:06Z")  # even seconds: ZIP stores time in 2 s steps
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_bytes(b"print('a')\n")  # bytes: write_text would give CRLF on Windows
    (root / "run.sh").write_bytes(b"echo hi\n")
    (root / "run.sh").chmod(0o755)
    (root / "notes.log").write_bytes(b"ignored\n")
    (root / ".gitignore").write_bytes(b"*.log\n")
    for cmd in (["init", "-q"], ["add", "-A"], ["update-index", "--chmod=+x", "run.sh"], ["commit", "-q", "-m", "init"]):
        subprocess.run(["git", "-C", str(root), *cmd], check=True, capture_output=True)
    return root


def test_pack_is_reproducible_and_holds_only_the_commit(repo, tmp_path):
    a = pack(repo, tmp_path / "a.zip", prefix="proj")
    (repo / "pkg" / "a.py").write_bytes(b"print('changed')\n")  # uncommitted edit
    (repo / "new.py").write_bytes(b"x = 1\n")                  # untracked file
    b = pack(repo, tmp_path / "b.zip", prefix="proj")
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    assert a["zip_sha256"] == b["zip_sha256"] and a["content_sha256"] == b["content_sha256"]
    assert not a["uncommitted_changes"] and b["uncommitted_changes"] and "note" in b
    z = zipfile.ZipFile(tmp_path / "b.zip")
    assert z.namelist() == ["proj/.gitignore", "proj/pkg/a.py", "proj/run.sh"]  # sorted; *.log ignored
    assert z.read("proj/pkg/a.py") == b"print('a')\n"
    modes = {i.filename: i.external_attr >> 16 for i in z.infolist()}
    assert modes["proj/run.sh"] == 0o755 and modes["proj/pkg/a.py"] == 0o644
    assert {i.date_time for i in z.infolist()} == {(2026, 1, 2, 3, 4, 6)}  # the commit's time, for every entry
