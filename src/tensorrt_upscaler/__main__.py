"""
Main entry point for the package.
Supports both CLI and GUI modes.
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--gui":
        # CLI mode
        from .cli import main as cli_main
        cli_main()
    else:
        # GUI mode
        from .main_window import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
