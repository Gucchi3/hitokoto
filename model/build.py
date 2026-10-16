# @file build.py
# @brief Build a model from its name
# @author Kohei Yamaguchi, Claude

##########################################################
'''
Add a model
1. Make a folder model/xxx/ and write the model class in model/xxx/xxx.py.
2. Add "name: class" to MODEL_REGISTRY.
3. Set config.model.name (or config.model.teacher_name) to that name.
'''
##########################################################

from rich.console import Console
from rich.text    import Text
from torch        import nn

import torch

from utils.exceptions import Error
from .hotaru          import Hotaru

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "hotaru": Hotaru,
}



def build_model(name: str, checkpoint: str | None) -> nn.Module:
    '''
    #* Make the model for the name, and load the weights when a checkpoint is given.
    #? Called from utils/trainer/pipeline.py for the student and the teacher.
    #! With a null checkpoint, the model starts from the default layer initialization.
    #! Weights whose names do not match (such as a head changed from MLM to classification) are skipped and listed. A layer with the same name and another shape stops with an error, so give a changed layer a new name.
    '''
    
    # Make the model
    if name not in MODEL_REGISTRY:
        raise Error("Unknown model.", detail=f"model name: {name!r}\nAvailable: {', '.join(MODEL_REGISTRY)}", hint="Add the model to MODEL_REGISTRY in model/build.py.", code=6)
    model = MODEL_REGISTRY[name]()
    if checkpoint is None:
        return model
    
    # Read the trained weights
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except OSError as error:
        raise Error("Could not load the checkpoint.", detail=f"{checkpoint}: {error}", hint="Check model.checkpoint and model.teacher_checkpoint, or set them to null.", code=8)
    
    # Load the weights whose names match
    try:
        result = model.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        raise Error("The checkpoint does not fit the model.", detail=f"{checkpoint}: {error}", hint="A layer has the same name but another shape. Give the changed layer a new name, so its weights are skipped.", code=13)
    
    # List the weights that were not loaded
    console = Console()
    if result.missing_keys:
        console.print(Text(f"[Checkpoint] Not in {checkpoint}, so they keep the new initial values: {', '.join(result.missing_keys)}", style="yellow"))
    if result.unexpected_keys:
        console.print(Text(f"[Checkpoint] Not in the model {name}, so they are not used: {', '.join(result.unexpected_keys)}", style="yellow"))
    return model
