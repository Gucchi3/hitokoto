# @file main.py
# @brief Main entry point
# @author Kohei Yamaguchi, Claude

from json    import JSONDecodeError
from pathlib import Path
from types   import SimpleNamespace

import argparse
import json

from utils.exceptions import Error
from utils.trainer    import train_pipeline

RUN_MODES = ("train",)



def main() -> None:
    '''
    #* Read the config, and run the mode in run.mode.
    #! Pass a JSON config file, such as python main.py --config config/hotaru.json.
    '''

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args   = parser.parse_args()

    # Load the config
    try:
        with args.config.open(encoding="utf-8") as file:
            config = json.load(file, object_hook=lambda data: SimpleNamespace(**data))
    except (OSError, JSONDecodeError) as error:
        raise Error("Failed to load the config file.", detail=f"{args.config}: {error}", hint="Check the path and the JSON format.", code=1)

    # Check the run mode
    if config.run.mode not in RUN_MODES:
        raise Error("Unknown run mode.", detail=f"run.mode: {config.run.mode!r}\nAvailable: {', '.join(RUN_MODES)}", hint="Set run.mode to 'train'.", code=2)

    # Run the training, and stop quietly on Ctrl+C
    try:
        train_pipeline(config)
    except KeyboardInterrupt:
        Error.show_interrupted()
    return None


if __name__ == "__main__":
    main()
