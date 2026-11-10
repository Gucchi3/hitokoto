# @file strategies.py
# @brief Hypothesis strategies (ways to make the inputs of the tests) shared by the tests

from hypothesis import strategies as st

OPEN_SLM_TESTS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "arithmark3")



def strategies_open_slm_scores() -> st.SearchStrategy[dict[str, float]]:
    '''
    #* Return a strategy of score dicts with any of the 5 Open SLM Leaderboard tests. Each score (%) is from 0 to 100, with 2 decimals (as on the leaderboard) or with all its digits.
    #? Used by tests/utils/validator/test_benchmark.py for benchmark_int_index().
    '''
    
    score = st.one_of(st.integers(0, 10000).map(lambda value: value / 100), st.floats(0, 100, allow_nan=False))
    return st.fixed_dictionaries({}, optional={name: score for name in OPEN_SLM_TESTS})


@st.composite
def strategies_continuation_requests(draw: st.DrawFn, vocab: int, max_tokens: int) -> list[tuple[list[int], int]]:
    '''
    #* Return a strategy of request lists for score_continuations(): 1 to 12 token lists of 2 to max_tokens tokens, each with a count from 1 to its length - 1.
    #? Used by tests/utils/validator/test_score.py.
    '''
    
    requests = list[tuple[list[int], int]]()
    for _ in range(draw(st.integers(1, 12))):
        ids = draw(st.lists(st.integers(0, vocab - 1), min_size=2, max_size=max_tokens))
        requests.append((ids, draw(st.integers(1, len(ids) - 1))))
    return requests


@st.composite
def strategies_choice_questions(draw: st.DrawFn) -> list[tuple[str, list[str], int]]:
    '''
    #* Return a strategy of 1 to 4 (context, choices, answer) questions for _benchmark_choices(). The texts use letters, digits, marks, spaces, and new lines, and a context can end with spaces or new lines.
    #? Used by tests/utils/validator/test_benchmark.py.
    #! Each context has a letter, so it has 1 or more tokens after the spaces at its end are removed.
    '''
    
    text      = st.text(alphabet="ab xyz12.?:\n", min_size=1, max_size=16)
    questions = list[tuple[str, list[str], int]]()
    for _ in range(draw(st.integers(1, 4))):
        context = draw(text) + draw(st.sampled_from("abxyz")) + draw(st.sampled_from(["", " ", "\n", "  ", " \n"]))
        choices = draw(st.lists(text, min_size=2, max_size=4))
        questions.append((context, choices, draw(st.integers(0, len(choices) - 1))))
    return questions
