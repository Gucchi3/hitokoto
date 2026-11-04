# @file hf_text.py
# @brief Download text datasets in parquet files from the Hugging Face Hub, such as FineWeb-Edu, Cosmopedia, and FineMath, and give their texts to corpus.py

##########################################################
'''
Add a source
1. Add "name: HFTextSource(repo, folder, text_column, default_gb)" to SOURCE_REGISTRY.
   folder must hold parquet files with a text column.
2. The option --name GB then sets how many GB of its text to use (0 leaves it out), and the token files go to data/name.
'''
##########################################################

from collections.abc import Iterator
from functools       import partial
from huggingface_hub import HfApi, hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.table      import Table

import argparse
import json
import pyarrow.parquet as pq
import random

from utils.exceptions import Error
from .corpus          import CHARS_PER_GB, Corpora, CorpusItem, run_corpus
from .types           import HFTextSource

SOURCE_REGISTRY: dict[str, HFTextSource] = {
    "fineweb_edu": HFTextSource("HuggingFaceFW/fineweb-edu"  , "sample/10BT"   , "text", 32.0),
    "cosmopedia" : HFTextSource("HuggingFaceTB/smollm-corpus", "cosmopedia-v2" , "text", 6.0 ),
    "finemath"   : HFTextSource("HuggingFaceTB/finemath"     , "finemath-4plus", "text", 2.0 ),
}
FILE_ORDER_SEED = 0
CHARS_PER_TOKEN = 3.4
READ_BATCH      = 1024



def download_hf_text(raw_dir: Path, source: HFTextSource, gb: float) -> None:
    '''
    #* Download parquet files of source into raw_dir until they hold gb GB of text, and save the list of the files and the amount to raw_dir/files.json.
    #? Called from run_corpus() for the download and all stages, through choose_hf_text(). read_hf_text() reads files.json.
    #! The files are taken in an order mixed with FILE_ORDER_SEED, so the same amount gives the same files, and a larger amount keeps the files of a smaller one.
    #! A file that is already in raw_dir is not downloaded again.
    '''
    
    # List the parquet files of the folder, and mix their order with the fixed seed
    try:
        paths = sorted(entry.path for entry in HfApi().list_repo_tree(source.repo, source.folder, recursive=True, repo_type="dataset") if entry.path.endswith(".parquet"))
    except OSError as error:
        raise Error("Failed to download from Hugging Face.", detail=f"{source.repo}/{source.folder}: {error}", hint="Check the network and SOURCE_REGISTRY, then run the download stage again.", code=25)
    random.Random(FILE_ORDER_SEED).shuffle(paths)
    
    # Download one file at a time, and count the characters of its texts, until the files hold the amount
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = int(gb * CHARS_PER_GB)
    used   = list[str]()
    chars  = 0
    size   = 0
    for path in paths:
        if chars >= target:
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
    Console().print(f"[Download] {source.repo}/{source.folder}: {len(used)} of {len(paths)} files ({size / 1e9:.2f} GB), with {chars / CHARS_PER_GB:.2f} GB of text. {min(chars, target) / CHARS_PER_GB:.2f} GB of it is used.")
    if chars < target:
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
        raise Error("Downloaded files are missing.", detail=f"Missing: {files_path}", hint="Run the download stage, such as python -m tool.data.hf_text download.", code=26)
    files      = json.loads(files_path.read_text(encoding="utf-8"))
    
    # Give the texts until they reach the amount
    texts = (text for path in files["files"] for batch in pq.ParquetFile(raw_dir / path).iter_batches(batch_size=READ_BATCH, columns=[files["text_column"]]) for text in batch.column(0).to_pylist())
    chars = 0
    for text in texts:
        if chars >= files["max_chars"]:
            break
        if text and (text := text.strip()):
            chars += len(text)
            yield [("text", text)]


def choose_hf_text(args: argparse.Namespace) -> Corpora:
    '''
    #* Check the amount of each source, show the planned mix before a download, and return the download and read functions of the sources with an amount above 0.
    #? Given to run_corpus() by run_hf_text(). run_corpus() calls it after it reads the command line.
    #! The amounts are for the download stage. In the other stages, an amount of 0 still leaves the source out, but the other amounts do not matter, because files.json keeps the amount of the download.
    '''
    
    # Check the amounts
    amounts = {name: getattr(args, name) for name in SOURCE_REGISTRY}
    if min(amounts.values()) < 0 or sum(amounts.values()) <= 0:
        raise Error("Invalid amount.", detail=", ".join(f"--{name} {gb:g}" for name, gb in amounts.items()), hint="Use 0 or more GB for each source (0 leaves it out), and more than 0 in total.", code=24)
    
    # Show the planned mix before a download
    if args.stage in ("download", "all"):
        total = sum(amounts.values())
        table = Table(title="Mix of the download", caption=f"1 GB = 10^9 characters of text. ~Tokens assumes {CHARS_PER_TOKEN} characters per token.")
        for column in ("Source", "Hugging Face folder", "Text (GB)", "Share", "~Tokens"):
            table.add_column(column, justify="right")
        for name, source in SOURCE_REGISTRY.items():
            table.add_row(name, f"{source.repo}/{source.folder}", f"{amounts[name]:g}", f"{amounts[name] / total:.1%}", f"{amounts[name] * CHARS_PER_GB / CHARS_PER_TOKEN / 1e9:.2f}B")
        table.add_section()
        table.add_row("total", "", f"{total:g}", "100.0%", f"{total * CHARS_PER_GB / CHARS_PER_TOKEN / 1e9:.2f}B")
        Console().print(table)
    
    return {name: (partial(download_hf_text, source=SOURCE_REGISTRY[name], gb=gb), read_hf_text) for name, gb in amounts.items() if gb > 0}


def run_hf_text() -> None:
    '''
    #* Add an option for the amount of each source, and run the corpus stages for the sources, with one tokenizer for all of them.
    #! Run from the project folder, such as python -m tool.data.hf_text all, or with other amounts, such as python -m tool.data.hf_text all --fineweb_edu 8 --cosmopedia 1.5 --finemath 0.5.
    #! The tokenizer goes to model/weight/hotaru_edu by default, so that it does not overwrite the TinyStories tokenizer.
    '''
    
    # Add an option for the amount of each source
    options = argparse.ArgumentParser(add_help=False)
    for name, source in SOURCE_REGISTRY.items():
        options.add_argument(f"--{name}", type=float, default=source.default_gb, metavar="GB", help=f"GB of text from {source.repo}/{source.folder} (default {source.default_gb:g}, 0 leaves it out)")
    
    run_corpus("Make the token files of text datasets on the Hugging Face Hub, one folder for each, with one tokenizer for all of them.", choose_hf_text, Path("model/weight/hotaru_edu"), (options,))
    return None


if __name__ == "__main__":
    run_hf_text()
