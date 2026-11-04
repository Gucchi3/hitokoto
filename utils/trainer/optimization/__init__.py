from .cross_entropy_z_loss import CrossEntropyZLoss
from .loss                 import IGNORE_INDEX, LossKind, build_loss_fn, loss_distill, loss_label
from .muon_adamw           import MuonAdamW
from .optimizer            import build_adamw_optimizer, build_muon_optimizer, build_normuon_optimizer, build_optimizer
from .scheduler            import build_cosine_annealing_scheduler, build_scheduler, build_wsd_scheduler

'''
optimization/
├── __init__.py
├── cross_entropy_z_loss.py
│   └── CrossEntropyZLoss
│       └── forward()
├── loss.py
│   ├── LossKind
│   ├── build_loss_fn()
│   ├── loss_label()
│   └── loss_distill()
├── muon_adamw.py
│   └── MuonAdamW
│       └── step()
├── optimizer.py
│   ├── OptimizerBuilder
│   ├── build_adamw_optimizer()
│   ├── build_muon_optimizer()
│   ├── build_normuon_optimizer()
│   ├── _optimizer_muon_groups()
│   └── build_optimizer()
└── scheduler.py
    ├── SchedulerBuilder
    ├── build_cosine_annealing_scheduler()
    ├── build_wsd_scheduler()
    └── build_scheduler()
'''
