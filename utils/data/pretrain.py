# @file pretrain.py
# @brief Build the DataBundle for pretraining from the token files made in tool/data/

from pathlib          import Path
from rich.console     import Console
from rich.table       import Table
from torch.utils.data import ConcatDataset, DataLoader
from types            import SimpleNamespace

import hashlib
import json
import torch

from utils.exceptions import Error
from .bin2dataset     import Bin2dataset
from .epoch_sampler   import EpochSampler
from .types           import DataBundle, DatasetMetadata, PreprocessingMetadata

SPLIT_NAMES = ("train", "validation", "test")



def build_pretrain_bundle(config: SimpleNamespace, seed: int) -> DataBundle:
    '''
    #* Return the train, validation, and test loaders for pretraining (next-token prediction on plain text), with the token files of the folders in data.sources mixed.
    #? Chosen in bundle.py when data.name is "pretrain". Each folder holds train.bin, validation.bin, test.bin, and meta.json, made by a dataset script in tool/data/.
    #! The repeat of a source sets how many times its train windows are used in one epoch: 1 uses each once, 2 twice, and 0.5 the first half. Validation and test use every window of every source once.
    #! Every token is a target, so no loss mask is needed. The train loader gives the windows in the order of EpochSampler (made from the seed and the epoch), and drops the last short batch, so all its batches have the same shape.
    '''
    
    # Check the sources
    if not config.sources or min(source.repeat for source in config.sources) <= 0:
        raise Error("Invalid data sources.", detail=f"data.sources: {[vars(source) for source in config.sources]}", hint='List one or more token folders, each with a repeat above 0, such as [{"root": "data/tinystories", "repeat": 1.0}].', code=28)
    
    # Check that the token files of each source are there, and that they were made with the tokenizer in data.tokenizer_dir
    tokenizer_path = Path(config.tokenizer_dir) / "tokenizer.json"
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() if tokenizer_path.exists() else None
    metas          = []
    for source in config.sources:
        root  = Path(source.root)
        paths = [root / "meta.json", *(root / f"{split}.bin" for split in SPLIT_NAMES)]
        if not all(path.exists() for path in paths):
            raise Error("Token files are missing.", detail=f"Missing: {', '.join(str(path) for path in paths if not path.exists())}",
                        hint="Run python -m tool.data all with dataset flags, such as --tinystories, and list their folders in data.sources.", code=21)
        meta  = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if meta["tokenizer_sha256"] != tokenizer_hash:
            raise Error("The token files do not match the tokenizer.", detail=f"The tokenizer hash in {root / 'meta.json'} differs from {tokenizer_path}.",
                        hint="Make all the token files in data.sources with one tokenizer (run the binarize stage again), and set data.tokenizer_dir to it.", code=22)
        metas.append(meta)
    
    # Make the windows of each source for each split, and join the sources (the train windows of a source are used repeat times)
    parts    = {split: [Bin2dataset(Path(source.root) / f"{split}.bin", config.max_length, source.repeat if split == "train" else 1.0) for source in config.sources] for split in SPLIT_NAMES}
    datasets = {split: ConcatDataset(windows) for split, windows in parts.items()}
    
    # Show the mix of the train windows in one epoch
    table = Table(title="Train data mix (one epoch)")
    for column in ("Source", "Train tokens", "Repeat", "Tokens per epoch", "Share"):
        table.add_column(column, justify="right")
    for source, meta, windows in zip(config.sources, metas, parts["train"]):
        table.add_row(source.root, f"{meta['tokens']['train']:,}", f"{source.repeat:g}", f"{len(windows) * config.max_length:,}", f"{len(windows) / len(datasets['train']):.1%}")
    Console().print(table)
    
    # Make the loaders. Only the train loader mixes the order (by the seed and the epoch), drops the last short batch, and keeps its workers between epochs
    workers           = config.num_workers
    pin_memory        = torch.cuda.is_available()
    train_sampler     = EpochSampler(len(datasets["train"]), seed)
    train_loader      = DataLoader(datasets["train"]     , batch_size=config.batch_size, sampler=train_sampler, num_workers=workers, pin_memory=pin_memory, drop_last=True , persistent_workers=workers > 0)
    validation_loader = DataLoader(datasets["validation"], batch_size=config.batch_size, sampler=None         , num_workers=workers, pin_memory=pin_memory, drop_last=False, persistent_workers=False      )
    test_loader       = DataLoader(datasets["test"]      , batch_size=config.batch_size, sampler=None         , num_workers=workers, pin_memory=pin_memory, drop_last=False, persistent_workers=False      )
    
    # Record the details of the dataset for metadata.json (the split seed is the one of the first source)
    preprocessing = PreprocessingMetadata(name="tokenizer", parameters={"tokenizer": metas[0]["tokenizer"], "vocab_size": metas[0]["vocab_size"], "max_length": config.max_length}, output_dtype="int64")
    metadata      = DatasetMetadata(name="+".join(meta["name"] for meta in metas), input_shape=(config.max_length,), num_classes=metas[0]["vocab_size"], class_names=None, train_size=len(datasets["train"]), validation_size=len(datasets["validation"]),
                                    test_size=len(datasets["test"]), split_seed=metas[0]["seed"], preprocessing=preprocessing)
    return DataBundle(train_loader, train_sampler, validation_loader, test_loader, metadata)
