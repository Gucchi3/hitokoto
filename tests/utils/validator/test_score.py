# @file test_score.py
# @brief Tests of utils/validator/score.py against the float64 reference of FakeCausalModel

from hypothesis import given, settings
from hypothesis import strategies as st

import math
import torch

from tests.fake_causal_model import FakeCausalModel
from tests.strategies        import strategies_continuation_requests
from utils.validator         import score_continuations, score_texts

CPU = torch.device("cpu")



def test_score_continuations_matches_reference() -> None:
    '''
    #* The score of each request is the float64 reference, for every batch size, and comes back in the order of the requests.
    '''
    
    # Lists of unsorted lengths, with counts other than 1 and the whole list
    model     = FakeCausalModel(vocab=50, positions=16, seed=1)
    requests  = [([3, 7, 9, 1], 2), ([5, 2], 1), ([8, 8, 8, 8, 8, 8, 4], 6), ([1, 2, 3], 1), ([9, 4, 4, 2, 0], 4)]
    reference = [model.log_likelihood(ids, count) for ids, count in requests]
    assert len(set(round(value, 3) for value in reference)) == len(reference)
    for batch_size in (1, 2, 3, 5, 8):
        scores = score_continuations(model, requests, CPU, batch_size)
        assert all(math.isfinite(score) for score in scores)
        assert all(abs(score - value) < 1e-4 for score, value in zip(scores, reference)), (batch_size, scores, reference)
    return None


def test_score_continuations_runs_longest_first() -> None:
    '''
    #* The longest lists run first, and each batch is as wide as its longest list - 1 (padding on the right does not change the scores).
    '''
    
    model = FakeCausalModel(vocab=50, positions=16, seed=2)
    score_continuations(model, [([1, 2, 3, 4], 2), ([5, 6], 1), ([7, 8, 9, 1, 2, 3, 4], 3), ([4, 3, 2], 1), ([1, 1, 1, 1, 1], 4)], CPU, 2)
    assert model.widths == [6, 3, 1]
    return None


@settings(max_examples=60, deadline=None)
@given(strategies_continuation_requests(vocab=30, max_tokens=10), st.integers(1, 6))
def test_score_continuations_property(requests: list[tuple[list[int], int]], batch_size: int) -> None:
    '''
    #* For any requests and batch size, the scores are the float64 reference.
    '''
    
    model  = FakeCausalModel(vocab=30, positions=10, seed=3)
    scores = score_continuations(model, requests, CPU, batch_size)
    assert len(scores) == len(requests)
    assert all(abs(score - model.log_likelihood(ids, count)) < 1e-4 for score, (ids, count) in zip(scores, requests))
    return None


def test_score_texts_matches_reference() -> None:
    '''
    #* score_texts() scores every token after the first one, and gives -inf to a list of fewer than 2 tokens.
    '''
    
    model     = FakeCausalModel(vocab=50, positions=16, seed=4)
    ids_list  = [[3, 7, 9, 1], [5], [8, 8, 8, 8, 8, 8, 4], [], [1, 2]]
    scores    = score_texts(model, ids_list, CPU, 2)
    reference = [model.log_likelihood(ids, len(ids) - 1) if len(ids) >= 2 else -math.inf for ids in ids_list]
    assert scores[1] == scores[3] == -math.inf
    assert all(abs(score - value) < 1e-4 for score, value in zip(scores, reference) if math.isfinite(value))
    return None
