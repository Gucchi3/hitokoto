# @file sources.py
# @brief Download and read the text and parquet datasets used by the shared corpus pipeline

##########################################################
'''
Add a source
1. Add "name: HFTextSource(repo, folder, text_column)" to SOURCE_REGISTRY.
   folder must hold parquet files with a text column.
2. The option --name selects the dataset, and the token files go to data/name.
'''
##########################################################

from collections.abc import Iterator
from dataclasses     import dataclass
from huggingface_hub import HfApi, hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.progress   import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from urllib.request  import urlopen

import json
import pyarrow.parquet as pq
import random

from utils.exceptions import Error
from .corpus          import CHARS_PER_GB, CorpusItem


@dataclass(frozen=True)
class HFTextSource:
    '''
    #* Locate parquet files and their text column on the Hugging Face Hub.
    '''

    repo       : str
    folder     : str
    text_column: str

SOURCE_REGISTRY: dict[str, HFTextSource] = {
    "fineweb_edu": HFTextSource("HuggingFaceFW/fineweb-edu"  , "sample/10BT"   , "text"),
    "cosmopedia" : HFTextSource("HuggingFaceTB/smollm-corpus", "cosmopedia-v2" , "text"),
    "finemath"   : HFTextSource("HuggingFaceTB/finemath"     , "finemath-4plus", "text"),
}
FILE_ORDER_SEED = 0
READ_BATCH      = 1024
SOURCE_URL      = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
SOURCE_FILES    = ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt")
STORY_END       = "<|endoftext|>"
READ_CHUNK      = 1 << 20



def download_hf_text(raw_dir: Path, source: HFTextSource, gb: float | None) -> None:
    '''
    #* Download parquet files of source into raw_dir until they hold gb GB of text, and save the list of the files and the amount to raw_dir/files.json.
    #? Called from the shared corpus pipeline. read_hf_text() reads files.json.
    #! The files are taken in an order mixed with FILE_ORDER_SEED, so the same amount gives the same files, and a larger amount keeps the files of a smaller one.
    #! A file that is already in raw_dir is not downloaded again.
    #! With gb=None, all parquet files in the configured source folder are downloaded and all their text is used.
    '''
    
    # List the parquet files of the folder, and mix their order with the fixed seed
    try:
        paths = sorted(entry.path for entry in HfApi().list_repo_tree(source.repo, source.folder, recursive=True, repo_type="dataset") if entry.path.endswith(".parquet"))
    except OSError as error:
        raise Error("Failed to download from Hugging Face.", detail=f"{source.repo}/{source.folder}: {error}", hint="Check the network and SOURCE_REGISTRY, then run the download stage again.", code=25)
    random.Random(FILE_ORDER_SEED).shuffle(paths)
    
    # Download one file at a time, and count the characters of its texts, until the files hold the amount
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = int(gb * CHARS_PER_GB) if gb is not None else None
    used   = list[str]()
    chars  = 0
    size   = 0
    for path in paths:
        if target is not None and chars >= target:
            break
        local = raw_dir / path
        if not local.exists():
            try:
                hf_hub_download(source.repo, path, repo_type="dataset", local_dir=raw_dir)
            except OSError as error:
                raise Error("Failed to download from Hugging Face.", detail=f"{source.repo}/{path}: {error}", hint="Check the network, then run the download stage again.", code=25)
        file = pq.ParquetFile(local)
        if source.text_column not in file.schema_arrow.names:
            raise Error("The text column is missing.", detail=f"{source.repo}/{path} has no column {source.text_column!r}.", hint="Fix text_column of the source in SOURCE_REGISTRY.", code=27)
        chars += sum(len(text) for batch in file.iter_batches(batch_size=READ_BATCH, columns=[source.text_column]) for text in batch.column(0).to_pylist() if text)
        size  += local.stat().st_size
        used.append(path)
    
    # Save the files and the amount, and show what was downloaded
    files = {"repo": source.repo, "folder": source.folder, "text_column": source.text_column, "max_chars": target, "files": used, "chars_in_files": chars}
    (raw_dir / "files.json").write_text(json.dumps(files, indent=4), encoding="utf-8")
    Console().print(f"[Download] {source.repo}/{source.folder}: {len(used)} of {len(paths)} files ({size / 1e9:.2f} GB), with {chars / CHARS_PER_GB:.2f} GB of text. {(chars if target is None else min(chars, target)) / CHARS_PER_GB:.2f} GB of it is used.")
    if target is not None and chars < target:
        Console().print(f"[yellow][Download] {source.repo}/{source.folder} has only {chars / CHARS_PER_GB:.2f} GB of text, so all of it is used.[/yellow]")
    return None


def read_hf_text(raw_dir: Path) -> Iterator[CorpusItem]:
    '''
    #* Give the texts in the files of raw_dir/files.json one by one, each as one text item, until they reach the amount of the download stage.
    #? Called from corpus.py by the split, tokenizer, and binarize stages, which must see the same texts in the same order.
    #! Spaces at both ends are removed, and empty texts are skipped.
    '''
    
    # Read the list of the files
    files_path = raw_dir / "files.json"
    if not files_path.exists():
        raise Error("Downloaded files are missing.", detail=f"Missing: {files_path}", hint="Run python -m tool.data download with the same dataset flags.", code=26)
    files      = json.loads(files_path.read_text(encoding="utf-8"))
    
    # Give the texts until they reach the amount
    texts = (text for path in files["files"] for batch in pq.ParquetFile(raw_dir / path).iter_batches(batch_size=READ_BATCH, columns=[files["text_column"]]) for text in batch.column(0).to_pylist())
    chars = 0
    for text in texts:
        if files["max_chars"] is not None and chars >= files["max_chars"]:
            break
        if text and (text := text.strip()):
            chars += len(text)
            yield [("text", text)]


def download_tinystories(raw_dir: Path) -> None:
    '''
    #* Download the two TinyStories GPT-4 text files, skipping existing files.
    #! Downloads go to .part files first, so an interrupted download never looks finished.
    '''
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        path = raw_dir / name
        if path.exists():
            Console().print(f"[Download] {path} is already there, so it is skipped.")
            continue
        part = path.with_name(path.name + ".part")
        try:
            with urlopen(f"{SOURCE_URL}/{name}") as response, part.open("wb") as file, Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn()) as progress:
                task = progress.add_task(name, total=int(response.headers.get("Content-Length", 0)) or None)
                while chunk := response.read(READ_CHUNK):
                    file.write(chunk)
                    progress.advance(task, len(chunk))
        except OSError as error:
            raise Error("Failed to download TinyStories.", detail=f"{SOURCE_URL}/{name}: {error}", hint="Check the network, then run the download stage again.", code=16)
        part.replace(path)
    return None


def read_tinystories(raw_dir: Path) -> Iterator[CorpusItem]:
    '''
    #* Read nonempty stories in source file order, separated by the end-of-text marker.
    #! Every stage must read the same stories in the same order.
    '''
    
    paths = [raw_dir / name for name in SOURCE_FILES]
    if not all(path.exists() for path in paths):
        raise Error("TinyStories files are missing.", detail=f"Missing: {', '.join(str(path) for path in paths if not path.exists())}", hint="Run python -m tool.data download --tinystories.", code=17)
    for path in paths:
        with path.open(encoding="utf-8") as file:
            rest = ""
            while chunk := file.read(READ_CHUNK):
                pieces = (rest + chunk).split(STORY_END)
                rest = pieces.pop()
                for piece in pieces:
                    if story := piece.strip():
                        yield [("text", story)]
            if story := rest.strip():
                yield [("text", story)]
