# @file corpus.py
# @brief Split a corpus of texts or talks, train a tokenizer, and write the token files

from collections.abc    import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools          import partial
from itertools          import batched
from pathlib            import Path
from rich.console       import Console
from rich.table         import Table
from tokenizers         import Tokenizer
from typing             import Literal

import argparse
import hashlib
import json
import numpy as np
import os

from utils.exceptions import Error
from utils.runtime    import remove_pycache
from .tokenizer       import TOKENIZER_REGISTRY, TokenizerType, build_tokenizer

Role           = Literal["text", "user", "bot"]
CorpusItem     = list[tuple[Role, str]]
CorpusReader   = Callable[[Path], Iterator[CorpusItem]]
CorpusDownload = Callable[[Path], None]
Corpora        = dict[str, tuple[CorpusDownload, CorpusReader]]

SPECIAL_TOKENS = ("<pad>", "<eos>", "<user>", "<bot>")
EOS_ID         = 1
USER_ID        = 2
BOT_ID         = 3
SPLIT_NAMES    = ("train", "validation", "test")
STAGES         = ("download", "split", "tokenizer", "binarize", "all")
ENCODE_BATCH   = 1000
ENCODE_WORKERS = os.cpu_count() or 1
CHARS_PER_GB   = 10**9

# The IDs put before and after the IDs of a text of each role
ROLE_TOKENS: dict[Role, tuple[list[int], list[int]]] = {
    "text": ([]       , [EOS_ID]),
    "user": ([USER_ID], []      ),
    "bot" : ([BOT_ID] , [EOS_ID]),
}



def split_corpus(data_dir: Path, read: CorpusReader, split_ratio: tuple[float, float, float], seed: int) -> None:
    '''
    #* Split the items into train, validation, and test, and save the split with the size of each item, the ratio, and the seed to split.npz.
    #? Called from run_corpus() for the split and all stages. train_corpus_tokenizer() and binarize_corpus() read split.npz, so all stages use the same split.
    #! The size of an item is the number of characters in its texts. train_corpus_tokenizer() uses the sizes to choose part of the train items.
    #! A test ratio of 0 makes no test items, and the test files of binarize_corpus() are empty.
    '''
    
    # Count the items and the characters in each item
    console = Console()
    console.print("[Split] Counting the items...")
    sizes   = np.fromiter((sum(len(text) for _, text in item) for item in read(data_dir / "raw")), dtype=np.int64)
    count   = len(sizes)
    
    # Decide the number of items in each split
    validation_count = round(count * split_ratio[1])
    test_count       = round(count * split_ratio[2])
    train_count      = count - validation_count - test_count
    if min(train_count, validation_count) < 1 or (split_ratio[2] > 0 and test_count < 1):
        raise Error("Invalid split ratio.", detail=f"items: {count}, train: {train_count}, validation: {validation_count}, test: {test_count}", hint="Make train, validation, and a test above 0 large enough to hold one item each.", code=15)
    
    # Mix the item numbers with the seed, then give each item its split (0 train, 1 validation, 2 test)
    order = np.random.default_rng(seed).permutation(count)
    split = np.zeros(count, dtype=np.uint8)
    split[order[train_count:train_count + validation_count]] = 1
    split[order[train_count + validation_count:]]            = 2
    np.savez(data_dir / "split.npz", split=split, sizes=sizes, ratio=np.array(split_ratio), seed=np.array(seed))
    console.print(f"[Split] Split {count} items of {data_dir.name} into train {train_count}, validation {validation_count}, test {test_count}.")
    return None


def train_corpus_tokenizer(corpora: dict[Path, CorpusReader], tokenizer_dir: Path, tokenizer_type: TokenizerType, vocab_size: int, max_chars: int) -> None:
    '''
    #* Train one tokenizer of tokenizer_type on the train items of all the corpora, and save it to tokenizer_dir/tokenizer.json.
    #? Called from run_corpus() for the tokenizer and all stages. corpora gives the read function of each corpus folder. Run split_corpus() for each corpus first.
    #! When the train items have more than max_chars characters in all, each item is used with the chance max_chars / (all the characters). So about max_chars characters are used, and each corpus keeps its share.
    #! The special tokens get the IDs 0, 1, 2, 3 in the order of SPECIAL_TOKENS. They are not in the texts, because binarize_corpus() adds them as IDs.
    #! This stage overwrites the tokenizer in tokenizer_dir.
    '''
    
    # Choose the train items to use. The seed of each split decides the choice (plus 1, so that the choice does not follow the split)
    splits      = {data_dir: _corpus_load_split(data_dir) for data_dir in corpora}
    train_chars = sum(int(sizes[split == 0].sum()) for split, sizes, _, _ in splits.values())
    chosen      = {data_dir: (split == 0) & (np.random.default_rng(seed + 1).random(len(split)) < max_chars / train_chars) for data_dir, (split, _, _, seed) in splits.items()}
    used_chars  = sum(int(sizes[chosen[data_dir]].sum()) for data_dir, (_, sizes, _, _) in splits.items())
    Console().print(f"[Tokenizer] Training on {used_chars / CHARS_PER_GB:.2f} GB of the {train_chars / CHARS_PER_GB:.2f} GB of train text (1 GB = 10^9 characters)...")
    
    # Train the tokenizer on the chosen items (the texts of one item are joined with new lines)
    tokenizer, trainer = build_tokenizer(tokenizer_type, vocab_size, list(SPECIAL_TOKENS))
    train_texts        = ("\n".join(text for _, text in item) for data_dir, read in corpora.items() for item, keep in zip(read(data_dir / "raw"), chosen[data_dir], strict=True) if keep)
    try:
        tokenizer.train_from_iterator(train_texts, trainer, length=sum(int(keep.sum()) for keep in chosen.values()))
    except ValueError:
        raise Error("Run the split stage again.", detail=f"A split.npz in {', '.join(str(data_dir) for data_dir in corpora)} has another number of items than its raw files.",
                    hint="The raw files changed after the split stage. Run the split stage, then this stage again.", code=18)
    
    # Save the tokenizer
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    Console().print(f"[Tokenizer] Saved {tokenizer_dir / 'tokenizer.json'} with {tokenizer.get_vocab_size()} tokens.")
    return None


def binarize_corpus(name: str, data_dir: Path, tokenizer_dir: Path, read: CorpusReader) -> None:
    '''
    #* Turn every item into token IDs with its role tokens, and write them as uint16 to train.bin, validation.bin, and test.bin, with the start of each item in .idx and meta.json.
    #? Called from run_corpus() for the binarize and all stages. _corpus_encode() encodes the items in the worker processes. Bin2dataset in utils/data/bin2dataset.py reads the .bin files.
    #! A text gets <eos> after it. A user turn gets <user> before it. A bot turn gets <bot> before it and <eos> after it (see ROLE_TOKENS).
    #! Each text is encoded alone and joined to the role tokens by IDs, so the result does not depend on how the tokenizer treats spaces or the language.
    #! Run split_corpus() first. The tokenizer in tokenizer_dir is used as it is, so an existing tokenizer also works. The files get their names only when all items are written.
    #! ENCODE_WORKERS processes encode the items, ENCODE_BATCH items at a time. Each uses one thread (TOKENIZERS_PARALLELISM=false, which the new processes get). One process with many threads used only one or two CPU cores on Windows.
    '''
    
    # Read the split and the tokenizer
    split, _, ratio, seed = _corpus_load_split(data_dir)
    tokenizer_path        = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise Error("Tokenizer is missing.", detail=f"Missing: {tokenizer_path}", hint="Run the tokenizer stage, or set --tokenizer_dir to the folder of an existing tokenizer.", code=20)
    tokenizer             = Tokenizer.from_file(str(tokenizer_path))
    
    # Open the .part files of each split: the token IDs (.bin, uint16) and the start of each item (.idx, uint64)
    bin_parts = [data_dir / f"{split_name}.bin.part" for split_name in SPLIT_NAMES]
    idx_parts = [data_dir / f"{split_name}.idx.part" for split_name in SPLIT_NAMES]
    bin_files = [path.open("wb") for path in bin_parts]
    idx_files = [path.open("wb") for path in idx_parts]
    items     = [0, 0, 0]
    tokens    = [0, 0, 0]
    roles     = set[Role]()
    
    # Encode the batches in the worker processes, and write the results in the order of the items. map() reads the items only buffersize batches ahead, so the memory stays small
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        with ProcessPoolExecutor(ENCODE_WORKERS) as pool:
            batches = batched(zip(read(data_dir / "raw"), split, strict=True), ENCODE_BATCH)
            for ids_by_split, starts_by_split, batch_roles in pool.map(partial(_corpus_encode, str(tokenizer_path)), batches, buffersize=2 * ENCODE_WORKERS):
                # Write the IDs and the starts of the items of each split. The starts in the batch move by the tokens written before
                for part, (ids, starts) in enumerate(zip(ids_by_split, starts_by_split)):
                    bin_files[part].write(ids.tobytes())
                    idx_files[part].write((starts + tokens[part]).tobytes())
                    items[part]  += len(starts)
                    tokens[part] += len(ids)
                roles |= batch_roles
        
        # End each .idx with the number of tokens, so that item i is ids[idx[i]:idx[i + 1]]
        for part, idx_file in enumerate(idx_files):
            idx_file.write(np.array([tokens[part]], dtype=np.uint64).tobytes())
    except ValueError:
        raise Error("Run the split stage again.", detail=f"split.npz has {len(split)} items, but the raw files have another number.", hint="The raw files changed after the split stage. Run the split stage again.", code=18)
    finally:
        for file in (*bin_files, *idx_files):
            file.close()
    
    # Give the finished files their names
    for path in (*bin_parts, *idx_parts):
        path.replace(path.with_suffix(""))
    
    # Save the details of the token files, with the hash of the tokenizer to find a wrong tokenizer later
    meta = {
        "name"            : name,
        "kind"            : "chat" if roles & {"user", "bot"} else "text",
        "tokenizer"       : tokenizer_path.as_posix(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size"      : tokenizer.get_vocab_size(),
        "special_tokens"  : {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        "dtype"           : "uint16",
        "split_ratio"     : ratio,
        "seed"            : seed,
        "items"           : dict(zip(SPLIT_NAMES, items)),
        "tokens"          : dict(zip(SPLIT_NAMES, tokens)),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")
    
    # Show the numbers of each split
    table = Table(title=f"{name} token files")
    for column in ("Split", "Items", "Tokens", "Tokens per item"):
        table.add_column(column, justify="right")
    for split_name, item_count, token_count in zip(SPLIT_NAMES, items, tokens):
        table.add_row(split_name, f"{item_count:,}", f"{token_count:,}", f"{token_count / item_count:.1f}" if item_count else "-")
    Console().print(table)
    return None


def _corpus_encode(tokenizer_path: str, batch: tuple[tuple[CorpusItem, int], ...]) -> tuple[list[np.ndarray], list[np.ndarray], set[Role]]:
    '''
    #* Encode the items of one batch, and return for each split the joined token IDs (uint16) and the start of each item in them (uint64), with the roles in the batch.
    #? Used by binarize_corpus(), which runs it in the worker processes, one batch of ENCODE_BATCH items at a time. batch holds each item with its split (0 train, 1 validation, 2 test).
    #! The tokenizer is read from the file for each batch. It takes a short time next to the encoding.
    '''
    
    # Encode all the texts of the batch at once, without the offsets of the tokens (they are not used)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encodings = iter(tokenizer.encode_batch_fast([text for item, _ in batch for _, text in item]))
    
    # Join the IDs of each item with its role tokens, in the list of its split
    joined = [list[int]() for _ in SPLIT_NAMES]
    starts = [list[int]() for _ in SPLIT_NAMES]
    roles  = set[Role]()
    for item, part in batch:
        starts[part].append(len(joined[part]))
        for role, _ in item:
            prefix, suffix = ROLE_TOKENS[role]
            joined[part].extend(prefix + next(encodings).ids + suffix)
            roles.add(role)
    return [np.array(ids, dtype=np.uint16) for ids in joined], [np.array(item_starts, dtype=np.uint64) for item_starts in starts], roles


def _corpus_load_split(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    '''
    #* Read split.npz, and return the split of each item, the size of each item, the split ratio, and the seed.
    #? Used by train_corpus_tokenizer() and binarize_corpus().
    '''
    
    split_path = data_dir / "split.npz"
    if not split_path.exists():
        raise Error("Run the split stage first.", detail=f"Missing: {split_path}", hint="Run the split stage of the dataset script, such as python -m tool.data.tinystories split.", code=18)
    with np.load(split_path) as saved:
        return saved["split"], saved["sizes"], saved["ratio"].tolist(), int(saved["seed"])


def run_corpus(description: str, choose: Callable[[argparse.Namespace], Corpora], tokenizer_dir: Path, parents: tuple[argparse.ArgumentParser, ...] = ()) -> None:
    '''
    #* Read the command line, check the values, and run the chosen stages for the corpora that choose() returns.
    #? Called at the end of a dataset script, such as tool/data/tinystories.py. choose() gets the read command line, and returns the download and read functions of each corpus by its name.
    #! Each corpus gets its own folder, data_root/name. All the corpora share one tokenizer: the tokenizer stage trains it on the train items of all of them, and the binarize stage uses it for each of them.
    #! tokenizer_dir is the default of --tokenizer_dir. parents adds the options of the dataset script, such as the amounts in hf_text.py.
    #! Run from the project folder, such as python -m tool.data.tinystories all. To keep an existing tokenizer, run download, split, and binarize, and skip the tokenizer stage.
    '''
    
    # Make the argument parser
    parser = argparse.ArgumentParser(description=description, parents=parents)
    parser.add_argument("stage"           , choices=STAGES                                                                        )
    parser.add_argument("--data_root"     , type=Path , default=Path("data")                                                      )
    parser.add_argument("--tokenizer_dir" , type=Path , default=tokenizer_dir                                                     )
    parser.add_argument("--tokenizer_type", choices=list(TOKENIZER_REGISTRY), default="bytelevel_bpe"                             )
    parser.add_argument("--vocab_size"    , type=int  , default=4096                                                              )
    parser.add_argument("--tokenizer_gb"  , type=float, default=2.0, metavar="GB"                                                 )
    parser.add_argument("--split"         , type=float, default=[0.995, 0.005, 0.000], nargs=3, metavar=("TRAIN", "VALID", "TEST"))
    parser.add_argument("--seed"          , type=int  , default=0                                                                 )
    
    # Stop quietly on Ctrl+C, and always remove the caches, even after an error
    try:
        # Read and check the values
        args = parser.parse_args()
        if not 256 + len(SPECIAL_TOKENS) <= args.vocab_size <= 65536:
            raise Error("Invalid vocab size.", detail=f"--vocab_size: {args.vocab_size}", hint=f"Use a number from {256 + len(SPECIAL_TOKENS)} to 65536.", code=14)
        if args.tokenizer_gb <= 0:
            raise Error("Invalid size of the tokenizer text.", detail=f"--tokenizer_gb: {args.tokenizer_gb}", hint="Use a number above 0, such as --tokenizer_gb 2.", code=23)
        if min(args.split[:2]) <= 0 or args.split[2] < 0 or abs(sum(args.split) - 1) > 1e-6:
            raise Error("Invalid split ratio.", detail=f"--split: {args.split}", hint="Use train and validation above 0 and a test of 0 or more that add up to 1, such as --split 0.99 0.005 0.005, or --split 0.995 0.005 0 for no test.", code=15)
        if args.seed < 0:
            raise Error("Invalid seed.", detail=f"--seed: {args.seed}", hint="Use a number of 0 or more.", code=19)
        corpora = choose(args)
        
        # Run the stages in order. The tokenizer stage makes one tokenizer for all the corpora
        if args.stage in ("download", "all"):
            for name, (download, _) in corpora.items():
                download(args.data_root / name / "raw")
        if args.stage in ("split", "all"):
            for name, (_, read) in corpora.items():
                split_corpus(args.data_root / name, read, tuple(args.split), args.seed)
        if args.stage in ("tokenizer", "all"):
            train_corpus_tokenizer({args.data_root / name: read for name, (_, read) in corpora.items()}, args.tokenizer_dir, args.tokenizer_type, args.vocab_size, int(args.tokenizer_gb * CHARS_PER_GB))
        if args.stage in ("binarize", "all"):
            for name, (_, read) in corpora.items():
                binarize_corpus(name, args.data_root / name, args.tokenizer_dir, read)
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None
