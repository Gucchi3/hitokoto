# @file main.py
# @brief Main entry point

##########################################################
'''
Add a run mode
1. Write a function that takes the config, such as chat_model(config) in utils/validator/chat.py.
2. Add "name: function" to RUN_REGISTRY.
3. Set run.mode to that name, or pass it with --mode.
'''
##########################################################

from collections.abc import Callable
from json            import JSONDecodeError
from pathlib         import Path
from types           import SimpleNamespace

import argparse
import json

from utils.exceptions import Error
from utils.runtime    import remove_pycache
from utils.trainer    import pipeline_train
from utils.validator  import benchmark_model, chat_model

RUN_REGISTRY: dict[str, Callable[[SimpleNamespace], None]] = {
    "train"    : pipeline_train,
    "chat"     : chat_model,
    "benchmark": benchmark_model,
}



def main() -> None:
    '''
    #* Read the config, and run the mode in run.mode, or the mode given with --mode.
    #! Pass a JSON config file, such as python main.py --config config/hotaru.json. Add --mode chat or --mode benchmark to use another mode without changing the config.
    '''
    
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path                 , required=True                                            )
    parser.add_argument("--mode"  , choices=list(RUN_REGISTRY), default=None , help="run mode to use instead of run.mode")
    args   = parser.parse_args()
    
    # Load the config
    try:
        with args.config.open(encoding="utf-8") as file:
            config = json.load(file, object_hook=lambda data: SimpleNamespace(**data))
    except (OSError, JSONDecodeError) as error:
        raise Error("Failed to load the config file.", detail=f"{args.config}: {error}", hint="Check the path and the JSON format.", code=1)
    
    # Choose the run mode
    mode = args.mode if args.mode is not None else config.run.mode
    if mode not in RUN_REGISTRY:
        raise Error("Unknown run mode.", detail=f"run mode: {mode!r}\nAvailable: {', '.join(RUN_REGISTRY)}", hint="Set run.mode, or pass --mode with one of the available modes.", code=2)
    
    # Run the mode, stop quietly on Ctrl+C, and always remove the caches
    try:
        RUN_REGISTRY[mode](config)
    except KeyboardInterrupt:
        Error.show_interrupted()
    finally:
        remove_pycache()
    return None


if __name__ == "__main__":
    main()
