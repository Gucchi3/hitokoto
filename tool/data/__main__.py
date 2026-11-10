# @file __main__.py
# @brief Prepare multiple text datasets with one shared tokenizer

from functools       import partial
from pathlib         import Path

import argparse

from utils.exceptions import Error
from .corpus          import Corpora, run_corpus
from .sources         import SOURCE_REGISTRY, download_hf_text, read_hf_text, download_tinystories, read_tinystories


def choose_corpora(args: argparse.Namespace) -> Corpora:
    '''
    #* Select entire datasets by flags, using format-specific downloaders and readers.
    #! Omitted datasets are excluded from every stage, including tokenizer training.
    '''

    if not any(getattr(args, name) for name in ("tinystories", *SOURCE_REGISTRY)):
        raise Error("No datasets selected.", hint="Select at least one dataset, such as --tinystories or --finemath.", code=24)
    corpora: Corpora = {}
    if args.tinystories:
        corpora["tinystories"] = (download_tinystories, read_tinystories)
    for name, source in SOURCE_REGISTRY.items():
        if getattr(args, name):
            corpora[name] = (partial(download_hf_text, source=source, gb=None), read_hf_text)
    return corpora


def main() -> None:
    '''
    #* Prepare selected datasets through the shared corpus pipeline.
    '''

    options = argparse.ArgumentParser(add_help=False)
    for name in ("tinystories", *SOURCE_REGISTRY):
        options.add_argument(f"--{name}", action="store_true", help="Use the entire dataset (omitted by default)")
    run_corpus("Prepare selected datasets with one shared tokenizer.", choose_corpora, Path("model/weight/hotaru_mix"), (options,))
    return None


if __name__ == "__main__":
    main()
