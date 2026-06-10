import sys
import io
import threading


class Tee(io.StringIO):
    """
    Duplicates stdout so prints go to both terminal and buffer.
    """
    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout

    def write(self, s):
        self.original_stdout.write(s)
        return super().write(s)

    def flush(self):
        self.original_stdout.flush()
        super().flush()


# Create stdout capture buffer
_stdout_buffer = Tee(sys.stdout)

# Redirect stdout
sys.stdout = _stdout_buffer


def get_stdout():
    """
    Returns all captured stdout text.
    """
    return _stdout_buffer.getvalue()


def _handle_cli_test_commands(g):
    """
    Background thread that reads CLI commands
    and forwards them to the object `g`.
    """
    while True:
        try:
            cmd = input("> ")

            if not cmd:
                continue

            print(f"CMD: {cmd}")

            g.cli(cmd)

        except EOFError:
            break
        except Exception as e:
            print(f"CLI thread error: {e}")


def handle_cli_test_commands(g):
    """
    Starts the CLI command handler thread.
    """
    thread = threading.Thread(
        target=_handle_cli_test_commands,
        args=(g,),
        daemon=True
    )

    thread.start()