# @file tinystories.py
# @brief Download TinyStories (the GPT-4 version), and give its stories to corpus.py

from collections.abc import Iterator
from pathlib         import Path
from rich.console    import Console
from rich.progress   import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from urllib.request  import urlopen

from utils.exceptions import Error
from .corpus          import CorpusItem, run_corpus

SOURCE_URL   = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
SOURCE_FILES = ("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt")
STORY_END    = "<|endoftext|>"
READ_CHUNK   = 1 << 20



def download_tinystories(raw_dir: Path) -> None:
    '''
    #* Download the two TinyStories GPT-4 text files into raw_dir.
    #? Called from run_corpus() in corpus.py for the download and all stages.
    #! A file that is already there is skipped. A download is written to a .part file first, so a stopped download never looks finished.
    '''
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each file that is not there yet
    for name in SOURCE_FILES:
        path = raw_dir / name
        if path.exists():
            Console().print(f"[Download] {path} is already there, so it is skipped.")
            continue
        
        # Write the file in chunks, and show the progress
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
    #* Give the stories of the two text files one by one, each as one text item, in the order of SOURCE_FILES.
    #? Called from corpus.py by the split, tokenizer, and binarize stages. All stages must see the stories in the same order.
    #! Spaces at both ends are removed, and empty stories are skipped.
    '''
    
    # Check that the text files are there
    paths = [raw_dir / name for name in SOURCE_FILES]
    if not all(path.exists() for path in paths):
        raise Error("TinyStories files are missing.", detail=f"Missing: {', '.join(str(path) for path in paths if not path.exists())}", hint="Run python -m tool.data.tinystories download.", code=17)
    
    # Read each file in chunks, and cut it at each end mark
    for path in paths:
        with path.open(encoding="utf-8") as file:
            rest = ""
            while chunk := file.read(READ_CHUNK):
                pieces = (rest + chunk).split(STORY_END)
                rest   = pieces.pop()
                for piece in pieces:
                    if story := piece.strip():
                        yield [("text", story)]
            if story := rest.strip():
                yield [("text", story)]


if __name__ == "__main__":
    run_corpus("Make the TinyStories token files.", lambda _: {"tinystories": (download_tinystories, read_tinystories)}, Path("model/weight/hotaru"))
