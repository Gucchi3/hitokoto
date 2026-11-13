# @file test_hotaru.py
# @brief Tests of model/hotaru/: the same first weights and outputs as TinyGpt of lab/tinylm_subbit, and causality
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import os
import pytest
import subprocess
import torch

from model.hotaru import Hotaru

# The folder of lab/tinylm_subbit (the reference), next to the folder that holds this project, or from TINYLM_SUBBIT
PROJECT     = Path(__file__).resolve().parents[3]
TINYLM_DIR  = Path(os.environ.get("TINYLM_SUBBIT", PROJECT.parent.parent / "lab" / "tinylm_subbit"))
TINYLM_PY   = TINYLM_DIR / ".venv" / "Scripts" / "python.exe" if os.name == "nt" else TINYLM_DIR / ".venv" / "bin" / "python"
REFERENCE   = Path(__file__).with_name("tinylm_reference.py")



@pytest.fixture(scope="module")
def tinylm_reference(tmp_path_factory: pytest.TempPathFactory) -> dict:
    '''
    #* Run tinylm_reference.py with the Python of lab/tinylm_subbit, and return what it saved: the first weights, random weights, inputs, and outputs of TinyGpt.
    #! The tests that use this are skipped when lab/tinylm_subbit or its .venv is not on this machine.
    '''
    
    if not TINYLM_PY.exists():
        pytest.skip(f"lab/tinylm_subbit is not found ({TINYLM_DIR}). Set TINYLM_SUBBIT to its folder.")
    path = tmp_path_factory.mktemp("tinylm") / "reference.pt"
    subprocess.run([str(TINYLM_PY), str(REFERENCE), str(path)], cwd=TINYLM_DIR, check=True)
    return torch.load(path, weights_only=True)


def test_hotaru_first_weights_match_tinylm(tinylm_reference: dict) -> None:
    '''
    #* With seed 0, Hotaru has the same weight names, in the same order, with the same first values as TinyGpt (ccca, float).
    '''
    
    torch.manual_seed(0)
    weights = Hotaru().state_dict()
    first   = tinylm_reference["first"]
    assert list(weights) == list(first)
    assert all(torch.equal(weights[name], first[name]) for name in weights)
    return None


def test_hotaru_outputs_match_tinylm(tinylm_reference: dict) -> None:
    '''
    #* With the random weights of TinyGpt loaded, Hotaru gives the same logits for a full window and a short one.
    '''
    
    # Load the weights (every name must fit), and check that the norms are not at their first values (1 and 0)
    weights = tinylm_reference["weights"]
    model   = Hotaru()
    model.load_state_dict(weights, strict=True)
    assert not torch.allclose(weights["blocks.0.norm1.weight"], torch.ones(256))
    assert not torch.allclose(weights["norm.bias"], torch.zeros(256))
    
    # The same logits for each input
    model.eval()
    with torch.no_grad():
        for tokens, expected in zip(tinylm_reference["inputs"], tinylm_reference["outputs"]):
            logits = model(tokens, torch.ones_like(tokens))
            assert logits.shape == (*tokens.shape, 256)
            assert torch.allclose(logits, expected, rtol=1e-5, atol=1e-5), (logits - expected).abs().max()
    return None


def test_hotaru_is_causal() -> None:
    '''
    #* Changing the token at one position changes the logits at that position and after it, but not before it (both the convolution and the attention blocks).
    #! The test checks positions near the start, inside the kernel of the convolution, and near the end, and that the change really reaches the later positions.
    '''
    
    # Random weights (the norms away from 1 and 0) and one window
    torch.manual_seed(3)
    model = Hotaru().eval()
    with torch.no_grad():
        for value in model.parameters():
            value.add_(torch.randn_like(value) * 0.05)
    tokens = torch.randint(0, 256, (1, 64))
    
    # Change one token at a time, and compare the logits before and after it
    with torch.no_grad():
        base = model(tokens, torch.ones_like(tokens))
        for position in (0, 1, 2, 30, 63):
            changed               = tokens.clone()
            changed[0, position]  = (tokens[0, position] + 1) % 256
            logits                = model(changed, torch.ones_like(changed))
            assert torch.equal(logits[0, :position], base[0, :position]), position
            assert (logits[0, position:] - base[0, position:]).abs().amax(dim=-1).min() > 1e-6, position
    return None
