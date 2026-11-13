# @file byte_text.py
# @brief Turn text files into byte token files (token i is the byte i) and save the byte tokenizer
# @author Kohei Yamaguchi, Claude

from pathlib      import Path
from rich.console import Console
from tokenizers   import Tokenizer, decoders, models, pre_tokenizers

import argparse
import hashlib
import json
import numpy as np

from utils.exceptions import Error
from utils.runtime    import remove_pycache

# The bytes read at a time, and the names of the token files
READ_CHUNK  = 1 << 24
SPLIT_NAMES = ("train", "validation", "test")



def byte_text_convert(train_path: Path, valid_path: Path, data_dir: Path, tokenizer_dir: Path) -> None:
    '''
    #* Save the byte tokenizer to tokenizer_dir/tokenizer.json, and write the bytes of the text files as uint16 token IDs to train.bin, validation.bin, and test.bin, with meta.json.
    #? Called from main(). Bin2dataset in utils/data/bin2dataset.py reads the .bin files, and build_pretrain_bundle() in utils/data/pretrain.py reads meta.json.
    #! The files are used as raw bytes (UTF-8), as in lab/tinylm_subbit: nothing is split, mixed, or added (no <eos>; "<|endoftext|>" in TinyStories stays as text). The vocabulary is the 256 byte values with no special tokens.
    #! test.bin gets the same bytes as validation.bin, because TinyStories has only train and valid files. The pipeline then tests on them instead of running the benchmarks.
    '''
    
    # Make the byte tokenizer: the 256 characters of the byte-level alphabet (as in GPT-2), with the ID of each character set to its byte, and no merges
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    others    = [byte for byte in range(256) if byte not in printable]
    chars     = {byte: chr(byte) for byte in printable} | {byte: chr(256 + index) for index, byte in enumerate(others)}
    tokenizer = Tokenizer(models.BPE(vocab={chars[byte]: byte for byte in range(256)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder       = decoders.ByteLevel()
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    
    # Write the bytes of each file as uint16, a chunk at a time
    data_dir.mkdir(parents=True, exist_ok=True)
    tokens = {}
    for split, source in zip(SPLIT_NAMES, (train_path, valid_path, valid_path)):
        tokens[split] = 0
        with source.open("rb") as reader, (data_dir / f"{split}.bin").open("wb") as writer:
            while chunk := reader.read(READ_CHUNK):
                writer.write(np.frombuffer(chunk, dtype=np.uint8).astype(np.uint16).tobytes())
                tokens[split] += len(chunk)
    
    # Save the details of the token files, with the hash of the tokenizer (pretrain.py checks it)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    meta           = {
        "name"            : data_dir.name,
        "kind"            : "text",
        "tokenizer"       : tokenizer_path.as_posix(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "vocab_size"      : tokenizer.get_vocab_size(),
        "special_tokens"  : {},
        "dtype"           : "uint16",
        "split_ratio"     : None,
        "seed"            : 0,
        "sources"         : {"train": train_path.as_posix(), "validation": valid_path.as_posix(), "test": valid_path.as_posix()},
        "tokens"          : tokens,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")
    Console().print(f"[Byte text] Saved {tokenizer_path} ({tokenizer.get_vocab_size()} tokens) and the token files in {data_dir}: " + ", ".join(f"{split} {count:,}" for split, count in tokens.items()))
    return None


def main() -> None:
    '''
    #* Read the command line, check the files, and make the byte token files and the byte tokenizer.
    #! Run from the project folder, such as python -m tool.data.byte_text --train ../lab/tinylm_data/train_part.txt --valid ../lab/tinylm_data/valid.txt.
    '''
    
    # Read the command line
    parser = argparse.ArgumentParser(description="Turn text files into byte token files, and save the byte tokenizer.")
    parser.add_argument("--train"        , type=Path, required=True                             , help="text file for train.bin")
    parser.add_argument("--valid"        , type=Path, required=True                             , help="text file for validation.bin and test.bin")
    parser.add_argument("--data_dir"     , type=Path, default=Path("data/tinystories_byte")     , help="folder of the token files")
    parser.add_argument("--tokenizer_dir", type=Path, default=Path("model/weight/hotaru_byte")  , help="folder of the byte tokenizer")
    
    # Check the files, and stop quietly on Ctrl+C, and always remove the caches
    try:
        args = parser.parse_args()
        if not (args.train.is_file() and args.valid.is_file()):
            raise Error("A text file is missing.", detail=f"--train {args.train}, --valid {args.valid}", hint="Give the paths of existing text files, such as TinyStoriesV2-GPT4-valid.txt.", code=39)
        byte_text_convert(args.train, args.valid, args.data_dir, args.tokenizer_dir)
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None


if __name__ == "__main__":
    main()
