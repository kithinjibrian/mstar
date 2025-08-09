import atexit
from typing import Optional
from .utils.tools import tool
import threading
import io
import contextlib
import traceback
import sys
from code import InteractiveInterpreter


class _Python:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_shell()
            return cls._instance

    def _init_shell(self):
        # Keep a persistent interpreter with its own namespace
        self.locals = {}
        self.interpreter = InteractiveInterpreter(self.locals)

    def run(self, code_str: str):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                try:
                    compiled = compile(code_str, "<python_shell>", "exec")
                    self.interpreter.runcode(compiled)
                except BaseException:
                    # Catch fatal issues during compile or exec
                    traceback.print_exc(file=sys.stderr)

        except Exception as fatal:
            stderr_buffer.write("Fatal error in Python shell:\n")
            stderr_buffer.write("".join(traceback.format_exception(type(fatal), fatal, fatal.__traceback__)))

        # Determine if there was an error by checking stderr contents
        error_flag = bool(stderr_buffer.getvalue().strip())
        output = stdout_buffer.getvalue() + stderr_buffer.getvalue()

        return error_flag, output.strip()
    
    def close(self):
        self.locals.clear()
        self.interpreter = None


@tool()
def python(code: str, wait_for: Optional[str] = None):
    """
    Execute Python code in a persistent interpreter session. Maintains state between calls (variables, imports, and functions persist). To get results, use print.

    Args:
        code: Python code to execute.
        wait_for: Reference outputs from other actions.
    """
    return _Python().run(code)

@tool()
def close_python():
    """
    Close the persistent Python shell
    """
    if _Python._instance:
        _Python._instance.close()
        _Python._instance = None
    
    return (False, "Python session closed")

atexit.register(close_python)