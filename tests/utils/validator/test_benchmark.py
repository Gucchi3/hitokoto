# @file test_benchmark.py
# @brief Tests of utils/validator/benchmark.py: the Int Index against the Open SLM Leaderboard page, and the Open SLM tests against lm-evaluation-harness and bencharithmark-3.py

from collections.abc import Callable
from fractions       import Fraction
from hypothesis      import given, settings
from hypothesis      import strategies as st
from pathlib         import Path
from tokenizers      import Tokenizer
from types           import SimpleNamespace

import json
import math
import pyarrow         as pa
import pyarrow.parquet as pq
import pytest
import torch

from tests.fake_causal_model import FakeCausalModel
from tests.strategies        import strategies_choice_questions, strategies_open_slm_scores
from utils.exceptions        import Error
from utils.validator         import benchmark

CPU       = torch.device("cpu")
ROOT      = Path(__file__).parents[3]
TOKENIZER = Tokenizer.from_file(str(ROOT / "model" / "weight" / "hotaru" / "tokenizer.json"))

# Scores of models on the Open SLM Leaderboard page, with the Int Index and the Avg that the page shows (getIntelligenceIndex() and getAvg() run in Node.js, not rounded)
LEADERBOARD_MODELS = [
    ({"hellaswag": 27.05, "arc_easy": 44.4, "arc_challenge": 22.7, "piqa": 58.27, "arithmark3": 33.1}, 10.32694063926941, 37.9925),                                   # Nexus-Erebus-50M
    ({"hellaswag": 43.22, "arc_easy": 58.63, "arc_challenge": 29.69, "piqa": 68.44, "arithmark3": 39.2}, 27.130593607305936, 48.754999999999995),                     # SmolLM2-135M
    ({"hellaswag": 29.2, "arc_easy": 41.12, "arc_challenge": 20.99, "piqa": 58.22}, 10.037777777777777, 39.49166666666667),                                           # Shard-1 (no ArithMark-3)
    ({"hellaswag": 31.26, "arc_easy": 39.35, "arc_challenge": 22.35, "piqa": 62.08, "arithmark3": 35.7}, 13.583561643835619, 39.9725),                                # GPT-2
    ({"hellaswag": 34.24, "arc_easy": 44.61, "arc_challenge": 26.37, "piqa": 65.89, "arithmark3": 35.3}, 18.359817351598174, 42.730000000000004),                      # Mantis-50M-Base
    ({"hellaswag": 46.98619842529297, "piqa": 66.90635681152344}, 31.563822428385414, 56.9462776184082),                                                               # random parts, from here
    ({"hellaswag": 23.27, "arithmark3": 44.0}, 8.581818181818184, 33.635),
    ({"arc_challenge": 86.34426891803741, "arithmark3": 69.43451166152954}, 72.91046585699524, 77.88939028978348),
    ({"hellaswag": 24.68125820159912, "arc_easy": 91.24648571014404, "arc_challenge": 33.81917476654053, "piqa": 28.197360038757324, "arithmark3": 71.52307033538818}, 12.694109302677521, 46.73362970352173),
]



def _test_benchmark_int_index_reference(scores: dict[str, float]) -> dict[str, float]:
    '''
    #* The Int Index and the Avg of the leaderboard page, computed with exact fractions: normalize(s, chance) = (s - chance) / (100 - chance) * 100, the ARC part is the mean of the ARC scores that exist, and ArithMark-3 has the weight 0.65.
    '''
    
    parts = list[tuple[Fraction, Fraction, Fraction]]()
    if "hellaswag" in scores:
        parts.append((Fraction(scores["hellaswag"]), Fraction(25), Fraction(1)))
    arcs = [Fraction(scores[name]) for name in ("arc_easy", "arc_challenge") if name in scores]
    if arcs:
        parts.append((sum(arcs, Fraction(0)) / len(arcs), Fraction(25), Fraction(1)))
    if "piqa" in scores:
        parts.append((Fraction(scores["piqa"]), Fraction(50), Fraction(1)))
    if "arithmark3" in scores:
        parts.append((Fraction(scores["arithmark3"]), Fraction(25), Fraction(65, 100)))
    if len(parts) < 2:
        return {}
    index = sum(((score - chance) / (100 - chance) * 100 * weight for score, chance, weight in parts), Fraction(0)) / sum((weight for _, _, weight in parts), Fraction(0))
    return {"int_index": float(index), "open_slm_avg": float(sum((score for score, _, _ in parts), Fraction(0)) / len(parts))}


def _test_benchmark_lm_eval_requests(questions: list[tuple[str, list[str], int]], max_length: int) -> list[list[tuple[list[int], int]]]:
    '''
    #* The requests of lm-evaluation-harness: _encode_pair() and the cut in _loglikelihood_tokens(), excluding whole questions with an overlong choice.
    '''
    
    requests = list[list[tuple[list[int], int]]]()
    for context, choices, _ in questions:
        question_requests = list[list[tuple[list[int], int]]]()
        for choice in choices:
            pair_context, continuation = context, " " + choice
            n_spaces                   = len(pair_context) - len(pair_context.rstrip())
            if n_spaces > 0:
                continuation = pair_context[-n_spaces:] + continuation
                pair_context = pair_context[:-n_spaces]
            whole_enc        = TOKENIZER.encode(pair_context + continuation).ids
            context_enc      = TOKENIZER.encode(pair_context).ids
            continuation_enc = whole_enc[len(context_enc):]
            question_requests.append([((context_enc + continuation_enc)[-(max_length + 1):], len(continuation_enc))])
        if all(count <= max_length for parts in question_requests for _, count in parts):
            requests += question_requests
    return requests


def _test_benchmark_lm_eval_accuracy(name: str, questions: list[tuple[str, list[str], int]], scores: list[float], skipped: int = 0) -> dict[str, float]:
    '''
    #* The accuracy of process_results() of ConfigurableTask in lm-evaluation-harness: acc is argmax(lls), and acc_norm is argmax(lls / the number of characters of the choice). numpy.argmax takes the first of equal values.
    '''
    
    acc, acc_norm, start = 0, 0, 0
    for _, choices, answer in questions:
        lls       = scores[start:start + len(choices)]
        norms     = [ll / float(len(choice)) for ll, choice in zip(lls, choices)]
        acc      += lls.index(max(lls)) == answer
        acc_norm += norms.index(max(norms)) == answer
        start    += len(choices)
    return {name: 100 * acc_norm / len(questions), f"{name}_acc": 100 * acc / len(questions), f"{name}_questions": len(questions), f"{name}_skipped": skipped}


def _test_benchmark_official_arithmark3(items: list[dict], max_context: int, score: Callable[[list[int], int], float]) -> tuple[list[tuple[list[int], int]], dict[str, float]]:
    '''
    #* The requests and the accuracy of bencharithmark-3.py: tokenize_request() (the context and the ending apart, the context cut to max_context - the ending tokens), and the choice of evaluate_arithmark_3() (raw and raw / tokens, the first of equal values).
    #* Whole questions with any ending of max_context or more tokens are excluded.
    '''
    
    requests        = list[tuple[list[int], int]]()
    raw, normalized, evaluated = 0, 0, 0
    for item in items:
        if any(len(TOKENIZER.encode(ending, add_special_tokens=False).ids) >= max_context for ending in item["endings"]):
            continue
        evaluated += 1
        item_requests = list[list[tuple[list[int], int]]]()
        for ending in item["endings"]:
            ctx_tokens  = TOKENIZER.encode(item["ctx"], add_special_tokens=False).ids
            cont_tokens = TOKENIZER.encode(ending, add_special_tokens=False).ids
            assert ctx_tokens and cont_tokens
            item_requests.append([(ctx_tokens[-(max_context - len(cont_tokens)):] + cont_tokens, len(cont_tokens))])
        raw_scores        = [sum(score(tokens, count) for tokens, count in parts) for parts in item_requests]
        normalized_scores = [value / sum(count for _, count in parts) for value, parts in zip(raw_scores, item_requests)]
        raw              += max(range(len(raw_scores)), key=raw_scores.__getitem__) == int(item["label"])
        normalized       += max(range(len(normalized_scores)), key=normalized_scores.__getitem__) == int(item["label"])
        requests         += [request for parts in item_requests for request in parts]
    return requests, {"arithmark3": 100 * normalized / evaluated, "arithmark3_acc": 100 * raw / evaluated, "arithmark3_questions": evaluated, "arithmark3_skipped": len(items) - evaluated}


def _test_benchmark_stub_scores(requests: list[tuple[list[int], int]]) -> list[float]:
    '''
    #* Scores made from the token lists, with many equal values, to stand in for score_continuations().
    '''
    
    return [-float((sum(ids) * 7 + count * 13) % 23) / 4 for ids, count in requests]


@pytest.mark.parametrize(("scores", "int_index", "avg"), LEADERBOARD_MODELS)
def test_benchmark_int_index_matches_leaderboard(scores: dict[str, float], int_index: float, avg: float) -> None:
    '''
    #* The Int Index and the Avg are the same floats as on the leaderboard page, also when some parts are missing.
    '''
    
    assert benchmark.benchmark_int_index(scores) == {"int_index": int_index, "open_slm_avg": avg}
    return None


def test_benchmark_int_index_needs_two_parts() -> None:
    '''
    #* With fewer than 2 parts, nothing is returned. The two ARC tests are one part, and the scores of the other tests are not used.
    '''
    
    assert benchmark.benchmark_int_index({}) == {}
    assert benchmark.benchmark_int_index({"hellaswag": 40.0, "blimp": 70.0, "arc_easy_tinyml": 50.0, "hellaswag_acc": 38.0}) == {}
    assert benchmark.benchmark_int_index({"arc_easy": 78.9471983909607, "arc_challenge": 74.46516752243042}) == {}
    assert benchmark.benchmark_int_index({"arc_easy": 60.0, "piqa": 70.0, "blimp": 99.0}) == {"int_index": (35 * 100 / 75 + 40) / 2, "open_slm_avg": 65.0}
    return None


def test_benchmark_int_index_rounds_like_leaderboard() -> None:
    '''
    #* The scores of lm-evaluation-harness with all their digits (Mantis-50M-Base) give the Int Index of the leaderboard (18.36) after rounding to 2 decimals.
    '''
    
    scores = {"hellaswag": 34.23620792670783, "arc_easy": 44.612794612794615, "arc_challenge": 26.36518771331058, "piqa": 65.88683351468988, "arithmark3": 35.3}
    result = benchmark.benchmark_int_index(scores)
    assert round(result["int_index"], 2) == 18.36
    assert round(result["open_slm_avg"], 2) == 42.73
    return None


@settings(max_examples=300, deadline=None)
@given(strategies_open_slm_scores())
def test_benchmark_int_index_property(scores: dict[str, float]) -> None:
    '''
    #* For any set of scores, the Int Index and the Avg are the exact values of the formulas of the leaderboard page.
    '''
    
    result    = benchmark.benchmark_int_index(scores)
    reference = _test_benchmark_int_index_reference(scores)
    assert result.keys() == reference.keys()
    assert all(math.isclose(result[name], reference[name], rel_tol=1e-12, abs_tol=1e-9) for name in reference)
    return None


@pytest.mark.parametrize(("text", "cleaned"), [
    ("  Baking cookies: A woman [header] puts flour in a bowl. She adds sugar ", "Baking cookies: A woman puts flour in a bowl. She adds sugar"),
    ("and runs [title] away"                                                   , "and runs. away"                                              ),
    ("Then [title] Add [step] salt."                                           , "Then. Add salt."                                             ),
    ("[step] [substeps] Mix it."                                               , " Mix it."                                                    ),
    ("a    b"                                                                  , "a  b"                                                        ),
])
def test_benchmark_hellaswag_text(text: str, cleaned: str) -> None:
    '''
    #* The texts are cleaned as preprocess() of lm-evaluation-harness does: " [title]" becomes ". ", other brackets are removed, and double spaces are made single once (so 4 spaces become 2).
    '''
    
    assert benchmark._benchmark_hellaswag_text(text) == cleaned
    return None


def test_benchmark_choices_matches_lm_eval(capsys: pytest.CaptureFixture[str]) -> None:
    '''
    #* With a real tokenizer and model, the accuracy is the one of lm-evaluation-harness, with spaces and new lines at the end of a context, and with a cut context.
    '''
    
    questions = [
        ("Question: Which is a gas?\nAnswer:"          , ["Ice", "Steam", "Rock"], 1),
        ("Fill in: the cat sat on the  "               , ["mat", "hat"]          , 0),
        ("Q: 2+2 =\n"                                  , ["4", "5", "22"]        , 0),
        ("A very long context " * 5 + "ends here."     , ["yes", "no", "maybe"]  , 2),
        ("Question: What is two plus two?\nAnswer:"    , ["four", "three"]       , 1),
    ]
    model     = FakeCausalModel(vocab=TOKENIZER.get_vocab_size(), positions=16, seed=5)
    requests  = _test_benchmark_lm_eval_requests(questions, 16)
    reference = _test_benchmark_lm_eval_accuracy("test", questions, [model.log_likelihood(*parts[0]) for parts in requests])
    result    = benchmark._benchmark_choices(model, TOKENIZER, CPU, 16, 3, "test", questions)
    assert all(len(parts) == 1 for parts in requests)
    assert result == reference
    assert 0 < reference["test"] < 100 and 0 < reference["test_acc"] < 100 and reference["test"] != reference["test_acc"]
    
    # The texts of more than 17 tokens are cut, and their number is shown
    cut    = sum(len(parts[0][0]) > 17 for parts in _test_benchmark_lm_eval_requests(questions, 10 ** 6))
    output = " ".join(capsys.readouterr().out.split())
    assert 0 < cut < len(requests)
    assert f"test: {cut} of {len(requests)} texts have more than 17 tokens, so their first tokens were cut." in output
    assert "parts" not in output
    return None


def test_benchmark_choices_skips_whole_questions(capsys: pytest.CaptureFixture[str]) -> None:
    '''
    #* Any overlong choice removes the whole question, even when it is not the answer. Remaining answers and the accuracy denominator stay aligned.
    '''
    
    questions = [
        ("Question: How do you open a jar?\nAnswer:", ["Twist the lid to the left until it comes off.", "Hit the jar on the floor.", "Lid."], 0),
        ("Question: How do you open a jar?\nAnswer:", ["Lid.", "Hit the jar on the floor.", "Twist the lid to the left until it comes off."], 0),
        ("Q: 2+2 ="                                 , ["4", "22", "four"]                                                             , 2),
        ("Question: Which is a gas?\nAnswer:"       , ["Ice", "Steam", "Rock"]                                                        , 1),
    ]
    model = FakeCausalModel(vocab=TOKENIZER.get_vocab_size(), positions=8, seed=7)
    for max_length in (3, 5):
        kept      = questions[2:]
        requests  = _test_benchmark_lm_eval_requests(questions, max_length)
        reference = _test_benchmark_lm_eval_accuracy("test", kept, [model.log_likelihood(*parts[0]) for parts in requests], skipped=2)
        assert benchmark._benchmark_choices(model, TOKENIZER, CPU, max_length, 2, "test", questions) == reference
        assert f"test: skipped 2 of 4 questions because a choice exceeds {max_length} tokens; evaluated 2 questions." in " ".join(capsys.readouterr().out.split())
    return None


@pytest.mark.parametrize("name", ["hellaswag", "arc_easy", "arc_challenge", "piqa"])
@pytest.mark.parametrize("skip_index", [0, 1, 2])
def test_benchmark_choices_skip_denominator(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str, skip_index: int) -> None:
    '''
    #* A skipped question, at any position and with multiple overlong choices, counts once and affects neither accuracy.
    '''
    
    questions = [("Q:", ["a", "b"], 1), ("Q:", ["a", "b"], 0)]
    questions.insert(skip_index, ("Q:", ["a", "cdef", "cdef"], 0))
    requests = list[tuple[list[int], int]]()
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.extend(batch) or [-2.0, -1.0, -1.0, -2.0])
    assert benchmark._benchmark_choices(None, TOKENIZER, CPU, 2, 4, name, questions) == {name: 100.0, f"{name}_acc": 100.0, f"{name}_questions": 2, f"{name}_skipped": 1}  # type: ignore[arg-type]
    assert len(requests) == 4 and all(count <= 2 for _, count in requests)
    assert f"{name}: skipped 1 of 3 questions" in " ".join(capsys.readouterr().out.split())
    return None


@settings(max_examples=300, deadline=None)
@given(st.lists(st.integers(0, 9), min_size=1, max_size=20), st.lists(st.integers(0, 9), min_size=1, max_size=40), st.integers(2, 12))
def test_benchmark_score_skipping_property(context: list[int], choice: list[int], size: int) -> None:
    '''
    #* Only whole fitting questions reach the scorer; each fitting choice is scored once with its context tail.
    '''
    
    captured = list[tuple[list[int], int]]()
    groups   = [[(context, [1], "short"), (context, choice, "test")], [(context, [2], "kept")]]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(benchmark, "score_continuations", lambda model, requests, device, batch_size: captured.extend(requests) or _test_benchmark_stub_scores(requests))
        scores, indices = benchmark._benchmark_score(None, CPU, 4, "test", groups, size)  # type: ignore[arg-type]
    expected = [((context + [2])[-size:], 1)]
    if len(choice) < size:
        expected = [((context + [1])[-size:], 1), ((context + choice)[-size:], len(choice))] + expected
    assert captured == expected
    assert indices == ([0, 1] if len(choice) < size else [1])
    assert [value for group in scores for value in group] == _test_benchmark_stub_scores(expected)
    return None


@settings(max_examples=150, deadline=None)
@given(strategies_choice_questions(), st.integers(1, 24))
def test_benchmark_choices_property(questions: list[tuple[str, list[str], int]], max_length: int) -> None:
    '''
    #* For any questions and max_length, fitting questions follow lm-evaluation-harness; overlong questions are excluded from both accuracy denominators.
    '''
    
    requests = list[list[tuple[list[int], int]]]()
    kept     = [question for question in questions if _test_benchmark_lm_eval_requests([question], max_length)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.append(batch) or _test_benchmark_stub_scores(batch))
        if not kept:
            with pytest.raises(Error) as error:
                benchmark._benchmark_choices(None, TOKENIZER, CPU, max_length, 4, "test", questions)  # type: ignore[arg-type]
            assert error.value.code == 38
            assert requests == []
            return None
        result = benchmark._benchmark_choices(None, TOKENIZER, CPU, max_length, 4, "test", questions)  # type: ignore[arg-type]
    reference = _test_benchmark_lm_eval_requests(questions, max_length)
    assert requests == [[request for parts in reference for request in parts]]
    assert result == _test_benchmark_lm_eval_accuracy("test", kept, [sum(_test_benchmark_stub_scores(parts)) for parts in reference], len(questions) - len(kept))
    return None


def test_benchmark_choices_picks_like_lm_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    '''
    #* acc takes the largest log-likelihood, acc_norm takes the largest log-likelihood / the number of characters of the choice (not tokens), and the first choice wins a tie.
    '''
    
    # The choices have 1, 3 | 1, 3 | 2, 1, 1 | 4, 1 tokens after "Q:", so dividing by the tokens would choose otherwise in the 2nd question
    questions = [("Q:", ["a", "bcd"], 1), ("Q:", ["ab", "cdef"], 0), ("Q:", ["x", "y", "z"], 1), ("Q:", ["xyz", "w"], 0)]
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, requests, device, batch_size: [-3.0, -6.0, -4.0, -8.0, -1.0, -1.0, -2.0, -3.0, -2.5][:len(requests)])
    assert benchmark._benchmark_choices(None, TOKENIZER, CPU, 64, 4, "test", questions) == {"test": 75.0, "test_acc": 25.0, "test_questions": 4, "test_skipped": 0}  # type: ignore[arg-type]
    return None


def test_benchmark_choices_cut_boundary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    '''
    #* Context truncation and the exact choice threshold are preserved; all-skipped questions and invalid inputs stop with error 38.
    '''
    
    requests  = list[tuple[list[int], int]]()
    questions = [("Question: What is two plus two?\nAnswer:", ["four"], 0)]
    whole     = TOKENIZER.encode("Question: What is two plus two?\nAnswer: four").ids
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.extend(batch) or [0.0] * len(batch))
    assert len(whole) == 19
    
    benchmark._benchmark_choices(None, TOKENIZER, CPU, 18, 4, "test", questions)  # type: ignore[arg-type]
    assert requests.pop() == (whole, 1)
    assert "cut" not in capsys.readouterr().out
    
    benchmark._benchmark_choices(None, TOKENIZER, CPU, 17, 4, "test", questions)  # type: ignore[arg-type]
    assert requests.pop() == (whole[1:], 1)
    assert "test: 1 of 1 texts have more than 18 tokens, so their first tokens were cut." in " ".join(capsys.readouterr().out.split())
    
    # " cdef" has 3 tokens after "Q:"
    ids = TOKENIZER.encode("Q: cdef").ids
    assert len(ids) == 5
    benchmark._benchmark_choices(None, TOKENIZER, CPU, 3, 4, "test", [("Q:", ["cdef"], 0)])  # type: ignore[arg-type]
    assert requests.pop() == (ids[-4:], 3)
    capsys.readouterr()
    for max_length in (2, 1):
        with pytest.raises(Error) as error:
            benchmark._benchmark_choices(None, TOKENIZER, CPU, max_length, 4, "test", [("Q:", ["cdef"], 0)])  # type: ignore[arg-type]
        assert error.value.code == 38
        assert requests == []
        assert "test: skipped 1 of 1 questions" in " ".join(capsys.readouterr().out.split())
    for max_length, context in ((0, "Q:"), (64, " \n ")):
        with pytest.raises(Error) as error:
            benchmark._benchmark_choices(None, TOKENIZER, CPU, max_length, 4, "test", [(context, ["cdef"], 0)])  # type: ignore[arg-type]
        assert error.value.code == 38
    return None


def _test_benchmark_capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[tuple[str, list[str], int]]]]:
    '''
    #* Replace _benchmark_choices() with a function that keeps the name and the questions it gets.
    '''
    
    calls = list[tuple[str, list[tuple[str, list[str], int]]]]()
    monkeypatch.setattr(benchmark, "_benchmark_choices", lambda model, tokenizer, device, max_length, batch_size, name, questions: calls.append((name, questions)) or {name: 0.0})
    return calls


def test_benchmark_hellaswag_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    '''
    #* The questions of HellaSwag are made as in process_docs() of lm-evaluation-harness: activity_label + ": " + ctx_a + " " + ctx_b.capitalize(), cleaned, and the label as an int.
    '''
    
    rows = [
        {"activity_label": "Baking cookies", "ctx_a": "A woman [header] puts flour in a bowl.", "ctx_b": "SHE ADDS sugar", "endings": ["and stirs it. [step] Then bakes.", " and leaves.", "then eats  it.", "and runs [title] away"], "label": "2"},
        {"activity_label": "Cooking"       , "ctx_a": "A man cuts onions."                    , "ctx_b": ""              , "endings": ["He cries.", "He sings."]                                                                     , "label": "0"},
    ]
    (tmp_path / "hellaswag" / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "hellaswag" / "data" / "validation-00000-of-00001.parquet")
    calls = _test_benchmark_capture(monkeypatch)
    benchmark.benchmark_hellaswag(None, TOKENIZER, CPU, tmp_path, 64, 4)  # type: ignore[arg-type]
    assert calls == [("hellaswag", [
        ("Baking cookies: A woman puts flour in a bowl. She adds sugar", ["and stirs it. Then bakes.", "and leaves.", "then eats it.", "and runs. away"], 2),
        ("Cooking: A man cuts onions."                                 , ["He cries.", "He sings."]                                                   , 0),
    ])]
    return None


@pytest.mark.parametrize(("function", "folder", "name"), [(benchmark.benchmark_arc_easy, "ARC-Easy", "arc_easy"), (benchmark.benchmark_arc_challenge, "ARC-Challenge", "arc_challenge")])
def test_benchmark_arc_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, function: benchmark.BenchmarkFunction, folder: str, name: str) -> None:
    '''
    #* The questions of ARC are "Question: " + question + "\\nAnswer:", with the index of answerKey in the labels (letters or numbers).
    '''
    
    rows = [
        {"id": "q1", "question": "Which is a gas?", "choices": {"text": ["Ice", "Steam", "Rock"], "label": ["A", "B", "C"]}     , "answerKey": "B"},
        {"id": "q2", "question": "What is 2+2?"   , "choices": {"text": ["3", "4", "5", "6"]   , "label": ["1", "2", "3", "4"]}, "answerKey": "4"},
    ]
    (tmp_path / "arc" / folder).mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "arc" / folder / "test-00000-of-00001.parquet")
    calls = _test_benchmark_capture(monkeypatch)
    function(None, TOKENIZER, CPU, tmp_path, 64, 4)  # type: ignore[arg-type]
    assert calls == [(name, [("Question: Which is a gas?\nAnswer:", ["Ice", "Steam", "Rock"], 1), ("Question: What is 2+2?\nAnswer:", ["3", "4", "5", "6"], 3)])]
    return None


def test_benchmark_piqa_questions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    '''
    #* The questions of PIQA are "Question: " + goal + "\\nAnswer:", with sol1 and sol2 as the choices.
    '''
    
    rows = [{"goal": "Open a jar", "sol1": "Twist the lid.", "sol2": "Hit the jar.", "label": 0}, {"goal": "Dry your hair", "sol1": "Use water.", "sol2": "Use a towel.", "label": 1}]
    (tmp_path / "piqa").mkdir()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "piqa" / "piqa_validation.parquet")
    calls = _test_benchmark_capture(monkeypatch)
    benchmark.benchmark_piqa(None, TOKENIZER, CPU, tmp_path, 64, 4)  # type: ignore[arg-type]
    assert calls == [("piqa", [("Question: Open a jar\nAnswer:", ["Twist the lid.", "Hit the jar."], 0), ("Question: Dry your hair\nAnswer:", ["Use water.", "Use a towel."], 1)])]
    return None


def _test_benchmark_write_arithmark3(folder: Path, items: list[dict]) -> None:
    '''
    #* Write the items as arithmark-3.jsonl with a BOM, Windows line ends, and empty lines, which bencharithmark-3.py skips.
    '''
    
    (folder / "arithmark3").mkdir(parents=True, exist_ok=True)
    lines = ["", *(json.dumps(item) for item in items[:1]), "   ", *(json.dumps(item) for item in items[1:]), ""]
    (folder / "arithmark3" / "arithmark-3.jsonl").write_bytes(b"\xef\xbb\xbf" + "\r\n".join(lines).encode("utf-8"))
    return None


def test_benchmark_arithmark3_matches_official(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    '''
    #* With a real tokenizer and model, the token lists and the accuracy are the ones of bencharithmark-3.py, with cut contexts and a label written as a string.
    '''
    
    items = [
        {"ind": 0, "ctx": "Tom has 3 apples and gets 2 more apples from his friend. Now Tom has", "endings": [" 5 apples.", " 6 apples.", " 1 apple.", " 32 apples."], "label": 0, "activity_label": "add"},
        {"ind": 1, "ctx": "12 - 5 ="                                                           , "endings": [" 7", " 8", " 17"]                                , "label": "0"},
        {"ind": 2, "ctx": "Half of 10 is"                                                      , "endings": [" 5", " 2"]                                       , "label": 0},
        {"ind": 3, "ctx": "Q: 2+2 ="                                                           , "endings": [" 4", " 22", " four"]                             , "label": 2},
    ]
    _test_benchmark_write_arithmark3(tmp_path, items)
    model                         = FakeCausalModel(vocab=TOKENIZER.get_vocab_size(), positions=12, seed=6)
    reference_requests, reference = _test_benchmark_official_arithmark3(items, 12, model.log_likelihood)
    requests                      = list[tuple[list[int], int]]()
    original                      = benchmark.score_continuations
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.extend(batch) or original(model, batch, device, batch_size))
    assert benchmark.benchmark_arithmark3(model, TOKENIZER, CPU, tmp_path, 12, 3) == reference
    assert requests == reference_requests
    assert 0 < reference["arithmark3"] < 100 and reference["arithmark3"] != reference["arithmark3_acc"]
    
    # The texts of more than 12 tokens are cut, and their number is shown
    cut    = sum(len(ids) > 12 for ids, _ in _test_benchmark_official_arithmark3(items, 10 ** 6, lambda ids, count: 0.0)[0])
    output = " ".join(capsys.readouterr().out.split())
    assert 0 < cut < len(requests)
    assert f"arithmark3: {cut} of {len(requests)} texts have more than 12 tokens, so their first tokens were cut." in output
    assert "parts" not in output
    
    # With max_length 3, any ending of 3 or more tokens removes the whole question
    reference_requests, reference = _test_benchmark_official_arithmark3(items, 3, model.log_likelihood)
    requests.clear()
    assert benchmark.benchmark_arithmark3(model, TOKENIZER, CPU, tmp_path, 3, 3) == reference
    assert requests == reference_requests
    assert 0 < reference["arithmark3_skipped"] < len(items)
    assert f"arithmark3: skipped {reference['arithmark3_skipped']} of 4 questions because a choice exceeds 2 tokens; evaluated {reference['arithmark3_questions']} questions." in " ".join(capsys.readouterr().out.split())
    return None


def test_benchmark_arithmark3_picks_like_official(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    '''
    #* acc takes the largest log-likelihood, acc_norm takes the largest log-likelihood / the number of tokens of the ending (not characters), and the first ending wins a tie.
    '''
    
    items = [
        {"ctx": "Q:", "endings": [" cat", " 12"]            , "label": 1},
        {"ctx": "Q:", "endings": [" a", " bcd"]             , "label": 0},
        {"ctx": "Q:", "endings": [" y", " z", " w", " ab"]  , "label": 1},
        {"ctx": "Q:", "endings": [" x", " yes"]             , "label": 1},
    ]
    assert [[len(TOKENIZER.encode(ending, add_special_tokens=False).ids) for ending in item["endings"]] for item in items] == [[1, 3], [1, 3], [1, 1, 1, 1], [2, 1]]
    _test_benchmark_write_arithmark3(tmp_path, items)
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, requests, device, batch_size: [-2.0, -3.0, -1.0, -3.0, -5.0, -1.0, -1.0, -2.0, -1.0, -3.0][:len(requests)])
    assert benchmark.benchmark_arithmark3(None, TOKENIZER, CPU, tmp_path, 64, 4) == {"arithmark3": 75.0, "arithmark3_acc": 50.0, "arithmark3_questions": 4, "arithmark3_skipped": 0}  # type: ignore[arg-type]
    return None


@pytest.mark.parametrize("skip_index", [0, 1, 2])
def test_benchmark_arithmark3_skip_denominator(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, skip_index: int) -> None:
    '''
    #* ArithMark also excludes whole questions from both accuracy denominators and retains answer indices after filtering.
    '''
    
    items = [{"ctx": "Q:", "endings": [" y", " z"], "label": "1"}, {"ctx": "Q:", "endings": [" y", " z"], "label": "0"}]
    items.insert(skip_index, {"ctx": "Q:", "endings": [" y", " 12", " 12"], "label": "0"})
    _test_benchmark_write_arithmark3(tmp_path, items)
    requests = list[tuple[list[int], int]]()
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.extend(batch) or [-2.0, -1.0, -1.0, -2.0])
    assert benchmark.benchmark_arithmark3(None, TOKENIZER, CPU, tmp_path, 3, 4) == {"arithmark3": 100.0, "arithmark3_acc": 100.0, "arithmark3_questions": 2, "arithmark3_skipped": 1}  # type: ignore[arg-type]
    assert len(requests) == 4 and all(count < 3 for _, count in requests)
    assert "arithmark3: skipped 1 of 3 questions" in " ".join(capsys.readouterr().out.split())
    return None


def test_benchmark_arithmark3_cut_boundary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    '''
    #* Context truncation and the exact ending threshold are preserved; all-skipped questions and invalid inputs stop with error 38.
    '''
    
    requests = list[tuple[list[int], int]]()
    context  = TOKENIZER.encode("Q: 2+2 =", add_special_tokens=False).ids
    ending   = TOKENIZER.encode(" 12", add_special_tokens=False).ids
    monkeypatch.setattr(benchmark, "score_continuations", lambda model, batch, device, batch_size: requests.extend(batch) or [0.0] * len(batch))
    assert (len(context), len(ending)) == (8, 3)
    
    for max_length, expected in ((11, [(context + ending, 3)]), (10, [(context[1:] + ending, 3)]), (4, [(context[-1:] + ending, 3)])):
        _test_benchmark_write_arithmark3(tmp_path, [{"ctx": "Q: 2+2 =", "endings": [" 12", " 12"], "label": 0}])
        requests.clear()
        benchmark.benchmark_arithmark3(None, TOKENIZER, CPU, tmp_path, max_length, 4)  # type: ignore[arg-type]
        output = " ".join(capsys.readouterr().out.split())
        assert requests == expected * 2
        assert ("arithmark3: 2 of 2 texts" in output) == (4 <= max_length < 11)
        assert "skipped" not in output
    with pytest.raises(Error) as error:
        benchmark.benchmark_arithmark3(None, TOKENIZER, CPU, tmp_path, 3, 4)  # type: ignore[arg-type]
    assert error.value.code == 38
    assert "arithmark3: skipped 1 of 1 questions" in " ".join(capsys.readouterr().out.split())
    for max_length, ctx in ((1, "Q: 2+2 ="), (64, "")):
        _test_benchmark_write_arithmark3(tmp_path, [{"ctx": ctx, "endings": [" 12", " 12"], "label": 0}])
        with pytest.raises(Error) as error:
            benchmark.benchmark_arithmark3(None, TOKENIZER, CPU, tmp_path, max_length, 4)  # type: ignore[arg-type]
        assert error.value.code == 38
    return None


def test_benchmark_tasks_groups_and_saves(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    '''
    #* benchmark_tasks() runs the tests with the settings of the config, adds the Int Index and the Avg, shows each leaderboard in the table, and saves all the scores.
    '''
    
    calls  = list[tuple[str, Path, int, int]]()
    scores = {"blimp": {"blimp": 70.5, "blimp_pairs": 67000}, "hellaswag": {"hellaswag": 43.22, "hellaswag_acc": 40.0, "hellaswag_questions": 10040, "hellaswag_skipped": 2}, "arc_easy": {"arc_easy": 58.63}, "arc_challenge": {"arc_challenge": 29.69}, "piqa": {"piqa": 68.44}, "arithmark3": {"arithmark3": 39.2}}
    for name, result in scores.items():
        monkeypatch.setitem(benchmark.BENCHMARK_REGISTRY, name, (benchmark.BENCHMARK_REGISTRY[name][0], lambda model, tokenizer, device, data_dir, max_length, batch_size, name=name, result=result: calls.append((name, data_dir, max_length, batch_size)) or result))
    config = SimpleNamespace(validator=SimpleNamespace(benchmark=SimpleNamespace(tasks=list(scores), data_dir=str(tmp_path / "data"), batch_size=8)), data=SimpleNamespace(max_length=256, tokenizer_dir="model/weight/hotaru"))
    
    result = benchmark.benchmark_tasks(None, TOKENIZER, CPU, config, tmp_path / "out", "log/run/best.pth")  # type: ignore[arg-type]
    assert calls == [(name, tmp_path / "data", 256, 8) for name in scores]
    assert result == {name: value for scores_of_task in scores.values() for name, value in scores_of_task.items()} | {"int_index": 27.130593607305936, "open_slm_avg": 48.754999999999995}
    saved = json.loads((tmp_path / "out" / "benchmark.json").read_text(encoding="utf-8"))
    assert (saved["weights"], saved["tokenizer_dir"], saved["max_length"], saved["scores"]) == ("log/run/best.pth", "model/weight/hotaru", 256, result)
    output = capsys.readouterr().out
    assert all(text in output for text in ("Tiny-ML", "Open SLM", "67,000", "70.5000", "int_index", "27.1306", "open_slm_avg", "48.7550", "hellaswag_skipped", "hellaswag_questions", "10,040"))
    
    # Without 2 parts of the Int Index, it is not added
    config.validator.benchmark.tasks = ["blimp", "piqa"]
    assert benchmark.benchmark_tasks(None, TOKENIZER, CPU, config, tmp_path / "out", None) == {"blimp": 70.5, "blimp_pairs": 67000, "piqa": 68.44}  # type: ignore[arg-type]
    assert "int_index" not in capsys.readouterr().out
    return None


@pytest.mark.parametrize("path", sorted((ROOT / "config").glob("*.json")), ids=lambda path: path.name)
def test_benchmark_check_configs(path: Path) -> None:
    '''
    #* Every name in validator.benchmark.tasks of the configs is a benchmark, and the configs run the Tiny-ML tests and the Open SLM tests.
    '''
    
    tasks = json.loads(path.read_text(encoding="utf-8"))["validator"]["benchmark"]["tasks"]
    benchmark.benchmark_check(SimpleNamespace(validator=SimpleNamespace(benchmark=SimpleNamespace(tasks=tasks))))
    assert {"hellaswag", "arc_easy", "arc_challenge", "piqa", "arithmark3"} <= set(tasks)
    return None


def test_benchmark_check_unknown() -> None:
    '''
    #* An unknown name, such as the old name arc_easy_lm_eval, stops with the error 35.
    '''
    
    with pytest.raises(Error) as error:
        benchmark.benchmark_check(SimpleNamespace(validator=SimpleNamespace(benchmark=SimpleNamespace(tasks=["blimp", "arc_easy_lm_eval"]))))
    assert error.value.code == 35
    return None
