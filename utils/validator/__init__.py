from .benchmark import BENCHMARK_REGISTRY, benchmark_arc_easy, benchmark_blimp, benchmark_check, benchmark_model, benchmark_tasks, benchmark_wikitext2
from .chat      import chat_model
from .generate  import generate_tokens
from .load      import load_model
from .score     import score_texts

'''
validator/
├── __init__.py
├── benchmark.py
│   ├── BenchmarkFunction
│   ├── benchmark_blimp()
│   ├── benchmark_arc_easy()
│   ├── benchmark_wikitext2()
│   ├── benchmark_check()
│   ├── benchmark_tasks()
│   └── benchmark_model()
├── chat.py
│   └── chat_model()
├── generate.py
│   └── generate_tokens()
├── load.py
│   └── load_model()
└── score.py
    └── score_texts()
'''
