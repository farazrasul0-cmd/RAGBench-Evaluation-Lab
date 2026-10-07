"""Report CLI commands: generate academic tables, benchmark markdown, and whitepapers."""

from pathlib import Path

import click

from app.services.report_generator import AcademicReportGenerator


@click.group(name="report")
def report_group() -> None:
    """Academic report and publication artifact generation."""


@report_group.command(name="generate")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("reports"),
    help="Target directory for generated publication artifacts.",
)
def generate_report(output_dir: Path) -> None:
    """Generate peer-reviewed LaTeX tables, benchmark report, and whitepaper."""
    generator = AcademicReportGenerator()
    results = generator.generate_full_report_package(output_dir)
    click.echo(f"Successfully generated {len(results)} publication artifacts:")
    for key, path in results.items():
        click.echo(f"  • {key}: {path}")
