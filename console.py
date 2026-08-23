"""
Console encoding for the command-line entry points.

Windows consoles default to cp1252, and LLM output routinely contains
characters it cannot encode - narrow no-break spaces, em dashes, curly quotes.
Printing a judge's reasoning was enough to kill the evaluation harness with a
UnicodeEncodeError partway through a run, losing every result gathered so far.

Called explicitly by the numbered entry-point scripts rather than on import of
a library module, since reconfiguring global streams is the caller's decision.
"""

import sys


def enable_utf8_output():
    """
    Switch stdout and stderr to UTF-8, replacing anything unencodable rather
    than raising. A mangled character in a log line is a cosmetic problem; a
    crashed benchmark run is not.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Not a real console (piped output, captured streams in tests).
            # Nothing to reconfigure, and nothing to fail over.
            pass
