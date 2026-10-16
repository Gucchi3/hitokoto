# @file remove_pycache.py
# @brief Remove Python caches from the training workflow
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import shutil



def remove_pycache() -> None:
    '''
    #* Remove all __pycache__ folders under the current folder.
    #? Called from utils/trainer/pipeline.py.
    '''
    
    # Remove the deepest folders first
    cache_dirs = sorted(Path.cwd().rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True)
    for cache_dir in cache_dirs:
        if cache_dir.is_dir() and not cache_dir.is_symlink():
            shutil.rmtree(cache_dir)
    return None
