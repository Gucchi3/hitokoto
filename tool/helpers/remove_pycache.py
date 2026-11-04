# @file remove_pycache.py
# @brief Remove Python caches from the command line

from utils.runtime import remove_pycache



def run_remove_pycache() -> None:
    '''
    #* Remove all __pycache__ folders under the current folder.
    #! Run from the project folder, such as python -m tool.helpers.remove_pycache.
    '''
    
    remove_pycache()
    return None


if __name__ == "__main__":
    run_remove_pycache()
