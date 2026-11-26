import dataclasses
import io
import urllib.request

from gib.cli import main
from gib.schema import write_jsonl


def test_fetch_images_downloads_each_image_once(dataset, tmp_path, monkeypatch, capsys):
    # six items on two images (an image can carry up to four items)
    items = [dataclasses.replace(it, image_path=f"val2017/{'a' if k < 4 else 'b'}.jpg",
                                 provenance={"coco_url": f"http://images.example/{'a' if k < 4 else 'b'}.jpg"})
             for k, it in enumerate(dataset["items"][:6])]
    write_jsonl(tmp_path / "data" / "items.jsonl", items)
    calls = []

    def fake_urlopen(url, timeout=None):  # no network: the "image" is the URL's bytes
        calls.append(url)
        return io.BytesIO(url.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    args = ["fetch-images", "--data", str(tmp_path / "data"), "--subset", "all",
            "--image-root", str(tmp_path / "img"), "--workers", "4"]
    main(args)
    assert sorted(calls) == ["http://images.example/a.jpg", "http://images.example/b.jpg"]
    assert (tmp_path / "img" / "val2017" / "a.jpg").read_bytes() == b"http://images.example/a.jpg"
    assert not list((tmp_path / "img").rglob("*.part"))
    main(args)  # a second run finds everything on disk
    assert len(calls) == 2 and "0 images to download" in capsys.readouterr().out
