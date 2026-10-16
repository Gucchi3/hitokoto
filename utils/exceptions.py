# @file exceptions.py
# @brief Error class with error details
# @author Kohei Yamaguchi, Claude

from rich.console import Console
from rich.text    import Text



class Error(SystemExit):
    '''
    #* Show an error message with its details and hint, then stop the program with the error code.
    #? The error codes are listed in docs/動作要求.md.
    #! Pass detail, hint, and code as keyword arguments, such as Error("Invalid device.", detail=f"run.device: {device!r}", hint="Use auto, cpu, or cuda.", code=4).
    '''
    
    def __init__(self, message: str, *, detail: str | None = None, hint: str | None = None, code: int) -> None:
        '''
        #* Keep the details, and show them on stderr without a traceback.
        '''
        
        # Keep the details and the exit code
        super().__init__(code)
        self.message = message
        self.detail  = detail
        self.hint    = hint
        
        # Show the error on stderr
        console = Console(stderr=True)
        console.print(Text(f"[ERROR] {message}", style="bold red"))
        if detail is not None:
            console.print(Text(detail))
        if hint is not None:
            console.print(Text(f"Hint: {hint}"))
        return None
    
    @staticmethod
    def show_interrupted() -> None:
        '''
        #* Show a message on stderr when Ctrl+C is pressed.
        #? Called from main.py.
        '''
        
        Console(stderr=True).print(Text("[Stopped] The user stopped the program."))
        return None
