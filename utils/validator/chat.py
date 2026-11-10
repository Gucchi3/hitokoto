# @file chat.py
# @brief Talk with a trained model in the terminal

from rich.console import Console
from rich.text    import Text
from types        import SimpleNamespace

from utils.exceptions import Error
from .generate        import generate_tokens
from .load            import load_model

CHAT_STYLES = ("text", "chat")



def chat_model(config: SimpleNamespace) -> None:
    '''
    #* Load the model, its weights, and the tokenizer, and talk with the model in the terminal.
    #? Chosen in main.py with run.mode (or --mode) chat. Uses load_model() in load.py and generate_tokens() in generate.py.
    #! validator.chat.style text: the model goes on from what you type (for a model after pretraining only). chat: what you type goes in as <user> text <bot>, and the model answers until <eos> (for a model after SFT).
    #! In the chat style, the earlier turns stay in the context, and the model sees the last data.max_length tokens. Type /reset to forget the turns, and /exit or an empty line to end.
    '''
    
    # Check the style
    settings = config.validator.chat
    if settings.style not in CHAT_STYLES:
        raise Error("Unknown chat style.", detail=f"validator.chat.style: {settings.style!r}\nAvailable: {', '.join(CHAT_STYLES)}", hint="Use text for a model after pretraining, and chat for a model after SFT.", code=32)
    
    # Load the model and the tokenizer, and check that the tokenizer has the tokens of the turns for the chat style
    model, tokenizer, device = load_model(config)
    vocab                    = tokenizer.get_vocab()
    if settings.style == "chat" and not {"<user>", "<bot>", "<eos>"} <= vocab.keys():
        raise Error("The tokenizer has no tokens for the turns.", detail=f"{config.data.tokenizer_dir} has no <user>, <bot>, or <eos>.", hint="Use validator.chat.style text, or a tokenizer made by tool/data.", code=34)
    
    # Talk until /exit or an empty line
    console = Console()
    history = list[int]()
    console.print(Text(f"[Chat] Style: {settings.style}. Type /reset to forget the talk, and /exit or an empty line to end.", style="cyan"))
    while True:
        try:
            text = console.input("[bold cyan]You[/bold cyan]: ").strip()
        except EOFError:
            break
        if text in ("", "/exit"):
            break
        if text == "/reset":
            history.clear()
            continue
        
        # Make the answer. In the chat style, the turn goes after the earlier turns, and the answer and <eos> are kept for the next turn
        if settings.style == "chat":
            prompt = history + [vocab["<user>"]] + tokenizer.encode(text).ids + [vocab["<bot>"]]
        else:
            # prompt = tokenizer.encode(text).ids
            prompt = [vocab["<eos>"]] + tokenizer.encode(text).ids
        answer = generate_tokens(model, prompt, device, config.data.max_length, settings.max_new_tokens, settings.temperature, settings.top_k, vocab.get("<eos>"))
        if settings.style == "chat":
            history = prompt + answer + [vocab["<eos>"]]
        console.print(Text("Model: ", style="bold green") + Text(tokenizer.decode(answer)))
    return None
