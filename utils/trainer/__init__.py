from .epoch    import evaluate_epoch, train_epoch
from .loop     import train_loop
from .pipeline import train_pipeline
from .types    import EpochMetrics, TrainingComponents

'''
trainer/
├── __init__.py
├── epoch.py
│   ├── train_epoch()
│   └── evaluate_epoch()
├── loop.py
│   └── train_loop()
├── pipeline.py
│   └── train_pipeline()
└── types.py
    ├── EpochMetrics
    └── TrainingComponents
'''
