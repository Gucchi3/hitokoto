# @file benchmark.py
# @brief Measure a model on the tests of the Tiny-ML Leaderboard (BLiMP, ARC-Easy, WikiText-2) and the Open SLM Leaderboard (HellaSwag, ARC-Easy, ARC-Challenge, PIQA, ArithMark-3, Int Index) by the rules of each leaderboard

##########################################################
'''
Add a benchmark
1. Write benchmark_xxx(model, tokenizer, device, data_dir, max_length, batch_size) to return a dict of scores.
2. Add "name: (leaderboard, benchmark_xxx)" to BENCHMARK_REGISTRY. The scores are shown in the table under the name of the leaderboard.
3. Add the name to validator.benchmark.tasks.
'''
##########################################################

from collections.abc import Callable
from datetime        import datetime
from huggingface_hub import hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.table      import Table
from rich.text       import Text
from tokenizers      import Tokenizer
from torch           import nn
from types           import SimpleNamespace

import json
import math
import pyarrow.parquet as pq
import re
import torch

from utils.exceptions import Error
from .load            import load_model
from .score           import score_continuations, score_texts

BenchmarkFunction = Callable[[nn.Module, Tokenizer, torch.device, Path, int, int], dict[str, float]]

# The 67 BLiMP tests (1,000 sentence pairs each), the same as in benchmark.py of Glint-1.3
BLIMP_CONFIGS = (
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement", "animate_subject_passive", "animate_subject_trans", "causative", "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch", "coordinate_structure_constraint_object_extraction", "determiner_noun_agreement_1", "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2", "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun", "distractor_agreement_relative_clause", "drop_argument",
    "ellipsis_n_bar_1", "ellipsis_n_bar_2", "existential_there_object_raising", "existential_there_quantifiers_1", "existential_there_quantifiers_2", "existential_there_subject_raising",
    "expletive_it_object_raising", "inchoative", "intransitive", "irregular_past_participle_adjectives", "irregular_past_participle_verbs", "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2", "left_branch_island_echo_question", "left_branch_island_simple_question", "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2", "principle_A_c_command", "principle_A_case_1", "principle_A_case_2", "principle_A_domain_1", "principle_A_domain_2",
    "principle_A_domain_3", "principle_A_reconstruction", "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2", "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope", "sentential_subject_island", "superlative_quantifiers_1", "superlative_quantifiers_2", "tough_vs_raising_1", "tough_vs_raising_2", "transitive", "wh_island",
    "wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
)

# The commit of AxiomicLabs/Arithmark-3.0 on the Hugging Face Hub, with the arithmark-3.jsonl (1,000 questions) that the Open SLM Leaderboard uses
ARITHMARK3_REVISION = "6f6e59dd9b7e2c63455f7af7f838f9ecc3d0a746"

# The parts of the Int Index (Intelligence Index) of the Open SLM Leaderboard: (name, score of random guessing (%), weight), in the order of getIntelligenceIndex() of the leaderboard page
# "arc" is the mean of arc_easy and arc_challenge
INT_INDEX_PARTS = (
    ("hellaswag" , 25.0, 1.0 ),
    ("arc"       , 25.0, 1.0 ),
    ("piqa"      , 50.0, 1.0 ),
    ("arithmark3", 25.0, 0.65),
)



def _benchmark_download(repo: str, paths: list[str], folder: Path, revision: str | None = None) -> list[Path]:
    '''
    #* Download files of a dataset on the Hugging Face Hub into folder, and return their local paths.
    #? Used by the benchmark_xxx() functions.
    #! A file that is already in folder is not downloaded again, so the tests run without the network after the first time.
    #! revision fixes the commit of the dataset. With None, the newest files are downloaded, as lm-evaluation-harness does.
    '''
    
    for path in paths:
        if not (folder / path).exists():
            try:
                hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=folder)
            except OSError as error:
                raise Error("Failed to download the benchmark data.", detail=f"{repo}/{path}: {error}", hint="Check the network, then run the benchmark again.", code=36)
    return [folder / path for path in paths]


def _benchmark_score(model: nn.Module, device: torch.device, batch_size: int, name: str, questions: list[list[tuple[list[int], list[int], str]]], size: int) -> tuple[list[list[float]], list[int]]:
    '''
    #* Return the choice log-likelihoods of each evaluated question, and its original index.
    #? Used by _benchmark_choices() and benchmark_arithmark3(). size is data.max_length + 1 for lm-evaluation-harness, and data.max_length for bencharithmark-3.py.
    #! If any choice exceeds size - 1 tokens, the whole question is skipped. Otherwise, only the first context tokens are cut. Both counts are shown.
    #! Empty contexts or choices, size < 2, and no evaluable questions stop with error 38.
    '''
    
    requests = list[tuple[list[int], int]]()
    indices  = list[int]()
    counts   = list[int]()
    cut      = 0
    for index, pairs in enumerate(questions):
        for context_ids, choice_ids, choice in pairs:
            if not context_ids or not choice_ids or size < 2:
                raise Error("A choice cannot be scored.", detail=f"{name}: the context has {len(context_ids)} tokens, and the choice {choice!r} has {len(choice_ids)} tokens (requests of up to {size} tokens).",
                            hint="The context and the choice need 1 or more tokens each. Check the data and data.max_length, or remove the test from validator.benchmark.tasks.", code=38)
        if any(len(choice_ids) >= size for _, choice_ids, _ in pairs):
            continue
        indices.append(index)
        counts.append(len(pairs))
        for context_ids, choice_ids, _ in pairs:
            cut += len(context_ids) + len(choice_ids) > size
            requests.append(((context_ids + choice_ids)[-size:], len(choice_ids)))
    skipped = len(questions) - len(indices)
    if skipped > 0:
        Console().print(Text(f"[Benchmark] {name}: skipped {skipped:,} of {len(questions):,} questions because a choice exceeds {size - 1:,} tokens; evaluated {len(indices):,} questions.", style="yellow"))
    if not indices:
        raise Error("No benchmark questions can be evaluated.", detail=f"{name}: {skipped:,} of {len(questions):,} questions were skipped.",
                    hint="Increase data.max_length or check the benchmark data.", code=38)
    if cut > 0:
        Console().print(Text(f"[Benchmark] {name}: {cut:,} of {len(requests):,} texts have more than {size:,} tokens, so their first tokens were cut.", style="yellow"))
    
    scores = iter(score_continuations(model, requests, device, batch_size))
    return [[next(scores) for _ in range(count)] for count in counts], indices


def _benchmark_choices(model: nn.Module, tokenizer: Tokenizer, device: torch.device, max_length: int, batch_size: int, name: str, questions: list[tuple[str, list[str], int]]) -> dict[str, float]:
    '''
    #* Answer each (context, choices, answer) question by the rules of lm-evaluation-harness (multiple_choice, zero-shot), and return the accuracy (%) with and without length normalization.
    #? Used by benchmark_hellaswag(), benchmark_arc_easy(), benchmark_arc_challenge(), and benchmark_piqa(). The rules follow _encode_pair() and _loglikelihood_tokens() of lm-evaluation-harness.
    #! Each choice is scored as " " + choice after the context. The spaces at the end of the context move to the front of the choice. The tokens of the choice are the tokens of context + " " + choice after as many tokens as the context alone has.
    #! When the context and the choice have more than max_length + 1 tokens, the first context tokens are cut. If any choice exceeds max_length tokens, the whole question is skipped and excluded from the accuracy denominator.
    #! name gets acc_norm, the number on the leaderboard (the largest log-likelihood / the number of characters of the choice), and name_acc gets acc (the largest log-likelihood). On a tie, the first choice wins.
    '''
    
    # Make the tokens of each choice after its context
    pairs = list[list[tuple[list[int], list[int], str]]]()
    for context, choices, _ in questions:
        context_ids  = tokenizer.encode(context.rstrip()).ids
        pairs.append([(context_ids, tokenizer.encode(context + " " + choice).ids[len(context_ids):], choice) for choice in choices])
    
    # Choose the choice with the largest log-likelihood, with and without dividing it by the number of characters of the choice
    scores, indices = _benchmark_score(model, device, batch_size, name, pairs, max_length + 1)
    acc, acc_norm   = 0, 0
    for index, totals in zip(indices, scores):
        _, choices, answer = questions[index]
        means     = [total / len(choice) for total, choice in zip(totals, choices)]
        acc      += max(range(len(totals)), key=totals.__getitem__) == answer
        acc_norm += max(range(len(means)), key=means.__getitem__) == answer
    return {name: 100 * acc_norm / len(indices), f"{name}_acc": 100 * acc / len(indices), f"{name}_questions": len(indices), f"{name}_skipped": len(questions) - len(indices)}


def _benchmark_hellaswag_text(text: str) -> str:
    '''
    #* Clean a text of HellaSwag, as preprocess() in tasks/hellaswag/utils.py of lm-evaluation-harness does.
    #? Used by benchmark_hellaswag().
    #! " [title]" becomes ". ", the other words in square brackets (marks of the WikiHow part) are removed, and each double space becomes one space (only once).
    '''
    
    text = text.strip().replace(" [title]", ". ")
    return re.sub("\\[.*?\\]", "", text).replace("  ", " ")


def benchmark_blimp(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the BLiMP accuracy (%): the share of the 67,000 sentence pairs where the grammatical sentence gets the larger sum of log probabilities.
    #! Each sentence is cut to max_length tokens. A tie counts as wrong.
    '''
    
    # Read the sentence pairs of the 67 tests
    files = _benchmark_download("nyu-mll/blimp", [f"{name}/train-00000-of-00001.parquet" for name in BLIMP_CONFIGS], data_dir / "blimp")
    pairs = [(row["sentence_good"], row["sentence_bad"]) for path in files for row in pq.read_table(path, columns=["sentence_good", "sentence_bad"]).to_pylist()]
    
    # Score both sentences of each pair, and count the pairs where the grammatical one is more likely
    good = score_texts(model, [tokenizer.encode(sentence).ids[:max_length] for sentence, _ in pairs], device, batch_size)
    bad  = score_texts(model, [tokenizer.encode(sentence).ids[:max_length] for _, sentence in pairs], device, batch_size)
    return {"blimp": 100 * sum(good_score > bad_score for good_score, bad_score in zip(good, bad)) / len(pairs), "blimp_pairs": len(pairs)}


def benchmark_arc_easy_tinyml(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the ARC-Easy accuracy (%) on the 2,376 test questions, by the rules of the Tiny-ML Leaderboard. Each question gets the choice with the largest log P(question + " " + choice) - log P(question).
    #! The score is not divided by the length, and no prompt format (such as "Question:") is added. Each text is cut to max_length tokens. On a tie, the first choice wins.
    #! The Open SLM Leaderboard uses other rules for ARC-Easy, so its score comes from benchmark_arc_easy().
    '''
    
    # Score each question alone and with each of its choices, all in the same batches
    files  = _benchmark_download("allenai/ai2_arc", ["ARC-Easy/test-00000-of-00001.parquet"], data_dir / "arc")
    rows   = pq.read_table(files[0]).to_pylist()
    texts  = [text for row in rows for text in (row["question"], *(row["question"] + " " + choice for choice in row["choices"]["text"]))]
    scores = iter(score_texts(model, [tokenizer.encode(text).ids[:max_length] for text in texts], device, batch_size))
    
    # Choose the choice whose log probability grows most from the question alone
    correct = 0
    for row in rows:
        question = next(scores)
        choices  = [next(scores) - question for _ in row["choices"]["text"]]
        correct += row["choices"]["label"][max(range(len(choices)), key=choices.__getitem__)] == row["answerKey"]
    return {"arc_easy_tinyml": 100 * correct / len(rows), "arc_easy_tinyml_questions": len(rows)}


def benchmark_wikitext2(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the perplexity of WikiText-2 (raw, test) for each byte (byte_ppl, the number on the leaderboard) and for each token.
    #! The lines are joined with spaces into one text. Its tokens are cut into windows of max_length + 1 tokens that do not overlap, and each window is scored alone.
    #! byte_ppl = exp(the negative log-likelihood of all the scored tokens / the UTF-8 bytes of the text), so models with other tokenizers can be compared.
    '''
    
    # Join the lines into one text, and cut its tokens into windows that move by max_length tokens
    files   = _benchmark_download("Salesforce/wikitext", ["wikitext-2-raw-v1/test-00000-of-00001.parquet"], data_dir / "wikitext")
    text    = " ".join(pq.read_table(files[0], columns=["text"]).column(0).to_pylist()).strip()
    ids     = tokenizer.encode(text).ids
    windows = [ids[start:start + max_length + 1] for start in range(0, len(ids) - 1, max_length)]
    
    # Add the negative log-likelihood of all the windows
    nll    = -sum(score_texts(model, windows, device, batch_size))
    tokens = sum(len(window) - 1 for window in windows)
    return {"wikitext2_byte_ppl": math.exp(nll / len(text.encode("utf-8"))), "wikitext2_token_ppl": math.exp(nll / tokens), "wikitext2_tokens": tokens}


def benchmark_hellaswag(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the HellaSwag accuracy (%) on the 10,042 validation questions, by the rules of lm-evaluation-harness that the Open SLM Leaderboard uses.
    #! The context is activity_label + ": " + ctx_a + " " + ctx_b (with str.capitalize(), which also makes the other letters small), and the 4 choices are the endings. All are cleaned by _benchmark_hellaswag_text().
    #! "hellaswag" is acc_norm, the number on the leaderboard. See _benchmark_choices() for the scoring.
    '''
    
    files     = _benchmark_download("Rowan/hellaswag", ["data/validation-00000-of-00001.parquet"], data_dir / "hellaswag")
    rows      = pq.read_table(files[0], columns=["activity_label", "ctx_a", "ctx_b", "endings", "label"]).to_pylist()
    questions = [(_benchmark_hellaswag_text(row["activity_label"] + ": " + row["ctx_a"] + " " + row["ctx_b"].capitalize()), [_benchmark_hellaswag_text(ending) for ending in row["endings"]], int(row["label"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "hellaswag", questions)


def benchmark_arc_easy(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the ARC-Easy accuracy (%) on the 2,376 test questions, by the rules of lm-evaluation-harness that the Open SLM Leaderboard uses.
    #! The context is "Question: " + question + "\\nAnswer:". The data is the same as in benchmark_arc_easy_tinyml() (the Tiny-ML Leaderboard), but the rules are not, so the two scores differ.
    #! "arc_easy" is acc_norm, the number on the leaderboard. See _benchmark_choices() for the scoring.
    '''
    
    files     = _benchmark_download("allenai/ai2_arc", ["ARC-Easy/test-00000-of-00001.parquet"], data_dir / "arc")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['question']}\nAnswer:", row["choices"]["text"], row["choices"]["label"].index(row["answerKey"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "arc_easy", questions)


def benchmark_arc_challenge(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the ARC-Challenge accuracy (%) on the 1,172 test questions, by the rules of lm-evaluation-harness that the Open SLM Leaderboard uses.
    #! The context is "Question: " + question + "\\nAnswer:". "arc_challenge" is acc_norm, the number on the leaderboard. See _benchmark_choices() for the scoring.
    '''
    
    files     = _benchmark_download("allenai/ai2_arc", ["ARC-Challenge/test-00000-of-00001.parquet"], data_dir / "arc")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['question']}\nAnswer:", row["choices"]["text"], row["choices"]["label"].index(row["answerKey"])) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "arc_challenge", questions)


def benchmark_piqa(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the PIQA accuracy (%) on the 1,838 validation questions, by the rules of lm-evaluation-harness that the Open SLM Leaderboard uses.
    #! The context is "Question: " + goal + "\\nAnswer:", and the 2 choices are sol1 and sol2. "piqa" is acc_norm, the number on the leaderboard. See _benchmark_choices() for the scoring.
    '''
    
    files     = _benchmark_download("baber/piqa", ["piqa_validation.parquet"], data_dir / "piqa")
    rows      = pq.read_table(files[0]).to_pylist()
    questions = [(f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], row["label"]) for row in rows]
    return _benchmark_choices(model, tokenizer, device, max_length, batch_size, "piqa", questions)


def benchmark_arithmark3(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the ArithMark-3 accuracy (%) on the 1,000 math word problems, by the rules of bencharithmark-3.py, the official script that the Open SLM Leaderboard uses.
    #! Unlike lm-evaluation-harness, the context and each ending are encoded apart (without special tokens) and joined. The first tokens of the context are cut, so that both fit in max_length tokens.
    #! If any ending has max_length or more tokens, the whole question is skipped and excluded from the accuracy denominator (the official script stops at it).
    #! "arithmark3" is acc_norm, the number on the leaderboard (the largest log-likelihood / the number of tokens of the ending), and "arithmark3_acc" is acc (the largest log-likelihood). On a tie, the first ending wins.
    #! The official script runs in bfloat16 by default. This runs in float32, its --dtype float32 setting, as the other tests do.
    '''
    
    # Read the questions, one JSON object in each line that is not empty
    files = _benchmark_download("AxiomicLabs/Arithmark-3.0", ["arithmark-3.jsonl"], data_dir / "arithmark3", ARITHMARK3_REVISION)
    with files[0].open(encoding="utf-8-sig") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    
    # Make the tokens of each ending after its context
    pairs = list[list[tuple[list[int], list[int], str]]]()
    for item in items:
        context_ids  = tokenizer.encode(item["ctx"], add_special_tokens=False).ids
        pairs.append([(context_ids, tokenizer.encode(ending, add_special_tokens=False).ids, ending) for ending in item["endings"]])
    
    # Choose the ending with the largest log-likelihood, with and without dividing it by the number of tokens of the ending
    scores, indices = _benchmark_score(model, device, batch_size, "arithmark3", pairs, max_length)
    acc, acc_norm   = 0, 0
    for index, totals in zip(indices, scores):
        item      = items[index]
        means     = [total / len(ending_ids) for total, (_, ending_ids, _) in zip(totals, pairs[index])]
        acc      += max(range(len(totals)), key=totals.__getitem__) == int(item["label"])
        acc_norm += max(range(len(means)), key=means.__getitem__) == int(item["label"])
    return {"arithmark3": 100 * acc_norm / len(indices), "arithmark3_acc": 100 * acc / len(indices), "arithmark3_questions": len(indices), "arithmark3_skipped": len(items) - len(indices)}


BENCHMARK_REGISTRY: dict[str, tuple[str, BenchmarkFunction]] = {
    "blimp"          : ("Tiny-ML" , benchmark_blimp          ),
    "arc_easy_tinyml": ("Tiny-ML" , benchmark_arc_easy_tinyml),
    "wikitext2"      : ("Tiny-ML" , benchmark_wikitext2      ),
    "hellaswag"      : ("Open SLM", benchmark_hellaswag      ),
    "arc_easy"       : ("Open SLM", benchmark_arc_easy       ),
    "arc_challenge"  : ("Open SLM", benchmark_arc_challenge  ),
    "piqa"           : ("Open SLM", benchmark_piqa           ),
    "arithmark3"     : ("Open SLM", benchmark_arithmark3     ),
}


def benchmark_int_index(scores: dict[str, float]) -> dict[str, float]:
    '''
    #* Return the Int Index ("int_index") and the Avg ("open_slm_avg") of the Open SLM Leaderboard from the scores (%) of HellaSwag, ARC, PIQA, and ArithMark-3.
    #? Called by benchmark_tasks(). The math is the same as getIntelligenceIndex() and getAvg() of the leaderboard page.
    #! Int Index: each score s becomes 100 * (s - chance) / (100 - chance), so that random guessing is 0 and all correct is 100 (below chance, it is less than 0), and these are averaged with the weights of INT_INDEX_PARTS.
    #! The ARC part is the mean of arc_easy and arc_challenge before the change (or the one that was run). Avg is the plain mean of the parts without the change.
    #! A part that was not run is left out of both means. With fewer than 2 parts, an empty dict is returned. The values are not rounded (the leaderboard shows 2 decimals).
    '''
    
    # Take the parts that were run. The ARC part is the mean of the two ARC scores, or the one that was run
    values = {name: scores[name] for name in ("hellaswag", "piqa", "arithmark3") if name in scores}
    arcs   = [scores[name] for name in ("arc_easy", "arc_challenge") if name in scores]
    if arcs:
        values["arc"] = arcs[0] if len(arcs) == 1 else (arcs[0] + arcs[1]) / 2
    parts = [(values[name], chance, weight) for name, chance, weight in INT_INDEX_PARTS if name in values]
    if len(parts) < 2:
        return {}
    
    # Add the parts one by one from the left, as the leaderboard page does (sum() of Python 3.12 or later adds with more precision, so its last digits can differ)
    index_total, index_weight, avg_total = 0.0, 0.0, 0.0
    for score, chance, weight in parts:
        index_total  += (score - chance) / (100 - chance) * 100 * weight
        index_weight += weight
        avg_total    += score
    return {"int_index": index_total / index_weight, "open_slm_avg": avg_total / len(parts)}


def benchmark_check(config: SimpleNamespace) -> None:
    '''
    #* Stop with an error when validator.benchmark.tasks has a name that is not in BENCHMARK_REGISTRY.
    #? Called by benchmark_model(), and by pipeline_train() in utils/trainer/pipeline.py before a training without test data, so that a wrong name stops the run before the training.
    '''
    
    unknown = [name for name in config.validator.benchmark.tasks if name not in BENCHMARK_REGISTRY]
    if unknown:
        raise Error("Unknown benchmark.", detail=f"validator.benchmark.tasks: {unknown}\nAvailable: {', '.join(BENCHMARK_REGISTRY)}", hint="Add the benchmark to BENCHMARK_REGISTRY in utils/validator/benchmark.py.", code=35)
    return None


def benchmark_tasks(model: nn.Module, tokenizer: Tokenizer, device: torch.device, config: SimpleNamespace, folder: Path, weights: str | None) -> dict[str, float]:
    '''
    #* Run the tests in validator.benchmark.tasks on the model, add the Int Index and the Avg of the Open SLM Leaderboard, show the scores in a table, and save them with the name of the weights to folder/benchmark.json.
    #? Called by benchmark_model(), and by pipeline_train() in utils/trainer/pipeline.py after a training without test data. Call benchmark_check() first.
    #! The Tiny-ML Leaderboard tests follow benchmark.py of Glint-1.3, and the Open SLM Leaderboard tests follow lm-evaluation-harness and bencharithmark-3.py. All run in float32 without a BOS token, and the model reads up to data.max_length tokens.
    #! The data is downloaded into validator.benchmark.data_dir the first time.
    '''
    
    # Run each test, and keep the names of the scores of each leaderboard
    settings = config.validator.benchmark
    console  = Console()
    scores   = dict[str, float]()
    groups   = dict[str, list[str]]()
    for name in settings.tasks:
        console.print(f"[Benchmark] Running {name}...")
        leaderboard, function = BENCHMARK_REGISTRY[name]
        result                = function(model, tokenizer, device, Path(settings.data_dir), config.data.max_length, settings.batch_size)
        scores               |= result
        groups.setdefault(leaderboard, []).extend(result)
    
    # Add the Int Index and the Avg when 2 or more of their parts were run
    index = benchmark_int_index(scores)
    if index:
        scores |= index
        groups.setdefault("Open SLM", []).extend(index)
    
    # Show the scores of each leaderboard in its own part of the table, with the Int Index and the Avg in bold
    table = Table(title="Benchmark", min_width=40)
    table.add_column("Leaderboard", style="bold magenta")
    table.add_column("Score"      , style="bold cyan"   )
    table.add_column("Value"      , justify="right"     )
    for leaderboard, names in groups.items():
        for row, name in enumerate(names):
            value = scores[name]
            table.add_row(leaderboard if row == 0 else "", name, f"{value:,}" if isinstance(value, int) else f"{value:.4f}", style="bold green" if name in index else None)
        table.add_section()
    console.print(table)
    
    # Save the scores with the name of the weights they belong to
    result = {"weights": weights, "tokenizer_dir": config.data.tokenizer_dir, "max_length": config.data.max_length, "date": datetime.now().isoformat(timespec="seconds"), "scores": scores}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "benchmark.json").write_text(json.dumps(result, indent=4), encoding="utf-8")
    console.print(f"[Benchmark] Saved {folder / 'benchmark.json'}.")
    return scores


def benchmark_model(config: SimpleNamespace) -> None:
    '''
    #* Load the model with the weights in model.checkpoint and its tokenizer, and run the tests in validator.benchmark.tasks with benchmark_tasks().
    #? Chosen in main.py with run.mode (or --mode) benchmark. Uses load_model() in load.py.
    #! benchmark.json goes next to the weights, or into run.log_dir when model.checkpoint is null.
    '''
    
    benchmark_check(config)
    model, tokenizer, device = load_model(config)
    folder                   = Path(config.model.checkpoint).parent if config.model.checkpoint is not None else Path(config.run.log_dir)
    benchmark_tasks(model, tokenizer, device, config, folder, config.model.checkpoint)
    return None
