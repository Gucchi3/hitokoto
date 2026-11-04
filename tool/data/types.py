# @file types.py
# @brief Data types for the dataset scripts

from dataclasses import dataclass



@dataclass(frozen=True)
class HFTextSource:
    '''
    #* Hold where a text dataset is on the Hugging Face Hub, and how many GB of its text to use by default.
    #? Used by SOURCE_REGISTRY in edu_mix.py.
    #! folder must hold parquet files that have the column text_column. One GB here means 10^9 characters of text.
    '''
    
    repo       : str
    folder     : str
    text_column: str
    default_gb : float
