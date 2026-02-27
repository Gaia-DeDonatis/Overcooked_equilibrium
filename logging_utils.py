"""Color-coded logging utilities for the Overcooked Equilibrium project."""

import logging
import sys

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright versions
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


# Color mapping for different identifier prefixes
IDENTIFIER_COLORS = {
    "OPT": Colors.BRIGHT_MAGENTA,      # Optimizer-related (BO, Sobol, kNN)
    "BACKEND": Colors.BRIGHT_CYAN,      # Backend/Flask routes
    "SESSION": Colors.BRIGHT_GREEN,     # Session management
    "ENV": Colors.BRIGHT_YELLOW,        # Environment operations
    "POLICY": Colors.BRIGHT_BLUE,       # Policy loading
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that color-codes the identifier portion of log messages."""

    def format(self, record):
        message = record.getMessage()

        # Check if message starts with a bracket identifier
        if message.startswith("["):
            bracket_end = message.find("]")
            if bracket_end != -1:
                identifier = message[1:bracket_end]
                rest = message[bracket_end + 1:]

                # Find the color for this identifier
                color = Colors.WHITE
                for prefix, c in IDENTIFIER_COLORS.items():
                    if identifier.startswith(prefix):
                        color = c
                        break

                # Apply color to identifier
                colored_message = f"{color}{Colors.BOLD}[{identifier}]{Colors.RESET}{rest}"
                record.msg = colored_message
                record.args = ()

        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with color-coded output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(ColoredFormatter("%(message)s"))

        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Pre-configured loggers
opt_logger = get_logger("optimizer")
backend_logger = get_logger("backend")
