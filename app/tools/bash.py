import atexit
from typing import Optional
from .utils.tools import tool
import subprocess
import threading

class _Bash:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_shell()
            return cls._instance

    def _init_shell(self):
        # Merge stderr into stdout so we only read one stream
        self.proc = subprocess.Popen(
            ["/bin/bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

    def run(self, command: str):
        marker = "__END_OF_COMMAND__"
        full_cmd = f"{command}\necho {marker} $?\n"

        self.proc.stdin.write(full_cmd)
        self.proc.stdin.flush()

        output_lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:  # process ended unexpectedly
                return True, "Shell terminated unexpectedly"
            if line.startswith(marker):
                _, exit_code_str = line.strip().rsplit(" ", 1)
                exit_code = int(exit_code_str)
                break
            output_lines.append(line)

        output_str = "".join(output_lines).strip()
        error_flag = exit_code != 0

        return error_flag, output_str
    
    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()


@tool()
def bash(command: str, wait_for: Optional[str] = None):
    """
    Execute a Bash command in a persistent shell session. Maintains state between calls (e.g., `cd` affects future calls).

    Args:
        command: A Bash command to run.
        wait_for: Reference outputs from other actions.
    """
    return _Bash().run(command)


@tool()
def close_bash():
    """Close the persistent bash shell"""
    if _Bash._instance:
        _Bash._instance.close()
        _Bash._instance = None

    return (False, "Bash session closed")

atexit.register(close_bash)
