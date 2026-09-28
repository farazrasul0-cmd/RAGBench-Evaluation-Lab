"""Root Click CLI entrypoint for RAGBench Evaluation Laboratory."""

import click

from app.cli.commands.dataset import dataset_group
from app.cli.commands.experiment import experiment_group


@click.group(name="ragbench")
@click.version_option(version="1.0.0", prog_name="ragbench")
def cli() -> None:
    """RAGBench: Retrieval-Augmented Generation Evaluation Laboratory CLI."""


cli.add_command(experiment_group)
cli.add_command(dataset_group)

if __name__ == "__main__":
    cli()
