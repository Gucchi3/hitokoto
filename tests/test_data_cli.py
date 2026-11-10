from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import hashlib
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tool.data import __main__ as cli
from tool.data import corpus
from tool.data import sources as hf_text
from tool.data.sources import HFTextSource
from utils.exceptions import Error


def test_tinystories_download_and_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Response(BytesIO):
        headers = {}

    calls = []

    def open_url(url):
        calls.append(url)
        return Response(b" first <|endoftext|>  <|endoftext|> second ")

    monkeypatch.setattr(hf_text, "urlopen", open_url)
    monkeypatch.setattr(hf_text, "READ_CHUNK", 3)
    hf_text.download_tinystories(tmp_path)
    assert len(calls) == 2
    assert list(hf_text.read_tinystories(tmp_path)) == [[("text", value)] for value in ("first", "second", "first", "second")]
    assert not list(tmp_path.glob("*.part"))
    hf_text.download_tinystories(tmp_path)
    assert len(calls) == 2


def test_tinystories_download_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def open_url(url):
        raise OSError("offline")

    monkeypatch.setattr(hf_text, "urlopen", open_url)
    with pytest.raises(Error) as error:
        hf_text.download_tinystories(tmp_path)
    assert error.value.code == 16
    assert not (tmp_path / hf_text.SOURCE_FILES[0]).exists()
    with pytest.raises(Error) as error:
        list(hf_text.read_tinystories(tmp_path))
    assert error.value.code == 17


def test_no_sources() -> None:
    with pytest.raises(Error) as error:
        cli.choose_corpora(Namespace(tinystories=0, fineweb_edu=0, cosmopedia=0, finemath=0))
    assert error.value.code == 24


@pytest.mark.parametrize("name", ["tinystories", "fineweb_edu", "cosmopedia", "finemath"])
def test_source_selection(name: str) -> None:
    args = Namespace(**{key: key == name for key in ("tinystories", *cli.SOURCE_REGISTRY)})
    assert list(cli.choose_corpora(args)) == [name]


@pytest.mark.parametrize("limit", [None, 4])
def test_hf_download_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int | None) -> None:
    class Api:
        def list_repo_tree(self, *args, **kwargs):
            return [Namespace(path="one.parquet"), Namespace(path="two.parquet")]

    calls = []

    def download(repo, path, **kwargs):
        calls.append(path)
        pq.write_table(pa.Table.from_pylist([{"text": "hello"}]), tmp_path / path)

    monkeypatch.setattr(hf_text, "HfApi", Api)
    monkeypatch.setattr(hf_text, "hf_hub_download", download)
    source = HFTextSource("test/repo", "subset", "text")
    hf_text.download_hf_text(tmp_path, source, None if limit is None else limit / corpus.CHARS_PER_GB)
    assert len(calls) == (2 if limit is None else 1)
    assert list(hf_text.read_hf_text(tmp_path)) == [[("text", "hello")]] * len(calls)
    calls.clear()
    hf_text.download_hf_text(tmp_path, source, None)
    assert len(calls) == (0 if limit is None else 1)
    assert list(hf_text.read_hf_text(tmp_path)) == [[("text", "hello")]] * 2


def test_mixed_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def download_tiny(path: Path) -> None:
        calls.append("tinystories")
        path.mkdir(parents=True)
        for name in ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"):
            (path / name).write_text("".join(f"A little story number {i}.<|endoftext|>" for i in range(20)), encoding="utf-8")

    def download_hf(path: Path, source: HFTextSource, gb: float | None) -> None:
        assert gb is None
        calls.append(source.repo)
        path.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"text": f"Math example number {i} equals four."} for i in range(30)]), path / "sample.parquet")
        (path / "files.json").write_text(json.dumps({"files": ["sample.parquet"], "text_column": "text", "max_chars": None}), encoding="utf-8")

    monkeypatch.setattr(cli, "download_tinystories", download_tiny)
    monkeypatch.setattr(cli, "download_hf_text", download_hf)
    monkeypatch.setattr(corpus, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(corpus, "ENCODE_WORKERS", 1)
    monkeypatch.setattr(corpus, "remove_pycache", lambda: None)
    data = tmp_path / "data"
    tokenizer = tmp_path / "tokenizer"
    monkeypatch.setattr(sys, "argv", ["tool.data", "all", "--tinystories", "--finemath", "--data_root", str(data), "--tokenizer_dir", str(tokenizer), "--vocab_size", "300", "--split", "0.8", "0.2", "0"])
    cli.main()
    assert calls == ["tinystories", "HuggingFaceTB/finemath"]
    digest = hashlib.sha256((tokenizer / "tokenizer.json").read_bytes()).hexdigest()
    for name in ("tinystories", "finemath"):
        meta = json.loads((data / name / "meta.json").read_text(encoding="utf-8"))
        assert meta["tokenizer_sha256"] == digest
        assert meta["tokens"]["train"] > 0 and meta["tokens"]["validation"] > 0
        assert (data / name / "test.bin").stat().st_size == 0
    assert not (data / "fineweb_edu").exists()
    assert not (data / "cosmopedia").exists()
    assert sum(json.loads((data / "tinystories" / "meta.json").read_text())["items"].values()) == 40
    assert sum(json.loads((data / "finemath" / "meta.json").read_text())["items"].values()) == 30
