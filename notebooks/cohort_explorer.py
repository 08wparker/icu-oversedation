import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import logging
    from datetime import datetime as dt_now
    from pathlib import Path
    return Path, sys


@app.cell
def _(Path, sys):
    # Add utils directory relative to current script location
    CURRENT_DIR = Path(__file__).resolve().parent
    UTILS_DIR = CURRENT_DIR.parent / 'utils'
    sys.path.append(str(UTILS_DIR))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
