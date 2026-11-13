# @file test_byte_text.py
# @brief Tests of tool/data/byte_text.py: the token files hold the bytes, and the tokenizer gives token i for byte i
# @author Kohei Yamaguchi, Claude

from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib    import Path
from tokenizers import Tokenizer

import hashlib
import json
import numpy as np
import pytest
import tempfile

from tests.strategies    import strategies_byte_texts
from tool.data           import byte_text
from tool.data.byte_text import byte_text_convert



def test_byte_text_files_hold_the_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    '''
    #* train.bin holds every byte of the train file in order (also across the read chunks), validation.bin and test.bin hold the valid file, and meta.json fits pretrain.py.
    #! The files hold all 256 byte values (also bytes that are not UTF-8), and the read chunk is smaller than the files, so a lost or doubled chunk is found.
    '''
    
    # Files with every byte value, longer than the read chunk
    monkeypatch.setattr(byte_text, "READ_CHUNK", 7)
    train = bytes(range(256)) * 3 + b"<|endoftext|> Once"
    valid = bytes(reversed(range(256))) + b"\xff\x00\n"
    (tmp_path / "train.txt").write_bytes(train)
    (tmp_path / "valid.txt").write_bytes(valid)
    byte_text_convert(tmp_path / "train.txt", tmp_path / "valid.txt", tmp_path / "data", tmp_path / "tok")
    
    # The token files are the bytes as uint16
    assert np.fromfile(tmp_path / "data" / "train.bin", dtype=np.uint16).tolist() == list(train)
    assert np.fromfile(tmp_path / "data" / "validation.bin", dtype=np.uint16).tolist() == list(valid)
    assert np.fromfile(tmp_path / "data" / "test.bin", dtype=np.uint16).tolist() == list(valid)
    
    # meta.json has the hash of the saved tokenizer, 256 tokens, and the token counts
    meta = json.loads((tmp_path / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer_sha256"] == hashlib.sha256((tmp_path / "tok" / "tokenizer.json").read_bytes()).hexdigest()
    assert meta["vocab_size"] == 256
    assert meta["tokens"] == {"train": len(train), "validation": len(valid), "test": len(valid)}
    return None


@settings(max_examples=200, deadline=None)
@given(strategies_byte_texts())
def test_byte_text_tokenizer_gives_the_bytes(text: str) -> None:
    '''
    #* For any text, the byte tokenizer gives the UTF-8 bytes of the text as the token IDs, and decoding them gives the text back.
    '''
    
    tokenizer = _byte_text_tokenizer()
    ids       = tokenizer.encode(text).ids
    assert ids == list(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text
    return None


def test_byte_text_tokenizer_has_every_byte() -> None:
    '''
    #* The vocabulary has the 256 IDs 0 to 255 and nothing else, and each ID decodes to its own byte.
    '''
    
    tokenizer = _byte_text_tokenizer()
    assert tokenizer.get_vocab_size() == 256
    assert sorted(tokenizer.get_vocab().values()) == list(range(256))
    assert [tokenizer.decode([byte]) for byte in range(128)] == [chr(byte) for byte in range(128)]
    assert tokenizer.encode("A é").ids == [0x41, 0x20, 0xC3, 0xA9]
    return None


_TOKENIZER: list[Tokenizer] = []


def _byte_text_tokenizer() -> Tokenizer:
    '''
    #* Return the byte tokenizer saved by byte_text_convert() (made once, in a temporary folder).
    '''
    
    if not _TOKENIZER:
        folder = Path(tempfile.mkdtemp())
        (folder / "text.txt").write_bytes(b"a")
        byte_text_convert(folder / "text.txt", folder / "text.txt", folder / "data", folder / "tok")
        _TOKENIZER.append(Tokenizer.from_file(str(folder / "tok" / "tokenizer.json")))
    return _TOKENIZER[0]


@given(st.binary(max_size=64))
@settings(max_examples=50, deadline=None)
def test_byte_text_any_bytes(data: bytes) -> None:
    '''
    #* Any bytes (also empty files and bytes that are not UTF-8) are written to the token files unchanged.
    '''
    
    folder = Path(tempfile.mkdtemp())
    (folder / "t.txt").write_bytes(data)
    byte_text_convert(folder / "t.txt", folder / "t.txt", folder / "data", folder / "tok")
    assert np.fromfile(folder / "data" / "train.bin", dtype=np.uint16).tolist() == list(data)
    return None
