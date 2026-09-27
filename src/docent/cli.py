"""Command-line entry point (`docent ...`). Session 2 adds the `ask` command."""

import typer

from docent import __version__

app = typer.Typer(help="Docent: ask questions about the Kubernetes docs, locally.")


@app.callback()
def main() -> None:
    """A callback makes `docent` a command *group*.

    Without it, Typer treats a one-command app as that command, and `docent version`
    fails with "unexpected extra argument". With it, subcommands work from day one.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)
