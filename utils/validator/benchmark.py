# @file benchmark.py
# @brief Measure a model on the tests of the Tiny-ML Leaderboard (BLiMP, ARC-Easy, WikiText-2) by the rules of the leaderboard

##########################################################
'''
Add a benchmark
1. Write benchmark_xxx(model, tokenizer, device, data_dir, max_length, batch_size) to return a dict of scores.
2. Add "name: benchmark_xxx" to BENCHMARK_REGISTRY.
3. Add the name to validator.benchmark.tasks.
'''
##########################################################

from collections.abc import Callable
from datetime        import datetime
from huggingface_hub import hf_hub_download
from pathlib         import Path
from rich.console    import Console
from rich.table      import Table
from tokenizers      import Tokenizer
from torch           import nn
from types           import SimpleNamespace

import json
import math
import pyarrow.parquet as pq
import torch

from utils.exceptions import Error
from .load            import load_model
from .score           import score_texts

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



def _benchmark_download(repo: str, paths: list[str], folder: Path) -> list[Path]:
    '''
    #* Download the parquet files of a dataset on the Hugging Face Hub into folder, and return their local paths.
    #? Used by benchmark_blimp(), benchmark_arc_easy(), and benchmark_wikitext2().
    #! A file that is already in folder is not downloaded again, so the tests run without the network after the first time.
    '''
    
    for path in paths:
        if not (folder / path).exists():
            try:
                hf_hub_download(repo, path, repo_type="dataset", local_dir=folder)
            except OSError as error:
                raise Error("Failed to download the benchmark data.", detail=f"{repo}/{path}: {error}", hint="Check the network, then run the benchmark again.", code=36)
    return [folder / path for path in paths]


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


def benchmark_arc_easy(model: nn.Module, tokenizer: Tokenizer, device: torch.device, data_dir: Path, max_length: int, batch_size: int) -> dict[str, float]:
    '''
    #* Return the ARC-Easy accuracy (%) on the 2,376 test questions. Each question gets the choice with the largest log P(question + " " + choice) - log P(question).
    #! The score is not divided by the length, and no prompt format (such as "Question:") is added. Each text is cut to max_length tokens. On a tie, the first choice wins.
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
    return {"arc_easy": 100 * correct / len(rows), "arc_easy_questions": len(rows)}


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


BENCHMARK_REGISTRY: dict[str, BenchmarkFunction] = {
    "blimp"    : benchmark_blimp,
    "arc_easy" : benchmark_arc_easy,
    "wikitext2": benchmark_wikitext2,
}


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
    #* Run the tests in validator.benchmark.tasks on the model, show the scores in a table, and save them with the name of the weights to folder/benchmark.json.
    #? Called by benchmark_model(), and by pipeline_train() in utils/trainer/pipeline.py after a training without test data. Call benchmark_check() first.
    #! The rules follow benchmark.py of Glint-1.3, on which the Tiny-ML Leaderboard is based: float32, no BOS token, the first token is not scored, and each text is cut to data.max_length tokens.
    #! The data is downloaded into validator.benchmark.data_dir the first time.
    '''
    
    # Run each test
    settings = config.validator.benchmark
    console  = Console()
    scores   = dict[str, float]()
    for name in settings.tasks:
        console.print(f"[Benchmark] Running {name}...")
        scores |= BENCHMARK_REGISTRY[name](model, tokenizer, device, Path(settings.data_dir), config.data.max_length, settings.batch_size)
    
    # Show the scores
    table = Table(title="Benchmark (Tiny-ML Leaderboard rules)", min_width=40)
    table.add_column("Score", style="bold cyan")
    table.add_column("Value", justify="right")
    for name, value in scores.items():
        table.add_row(name, f"{value:,}" if isinstance(value, int) else f"{value:.4f}")
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
