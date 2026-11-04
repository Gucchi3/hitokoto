from .print          import print_metadata, print_progress, print_validation
from .remove_pycache import remove_pycache
from .save           import save_graph, save_history, save_metadata
from .setup          import setup_device, setup_log_dir, setup_seed

'''
runtime/
├── __init__.py
├── print.py
│   ├── print_metadata()
│   ├── print_validation()
│   └── print_progress()
├── remove_pycache.py
│   └── remove_pycache()
├── save.py
│   ├── save_metadata()
│   ├── save_history()
│   └── save_graph()
└── setup.py
    ├── setup_seed()
    ├── setup_device()
    └── setup_log_dir()
'''
