import click


@click.command(
    context_settings={"show_default": True},
    short_help="Download tilt series and alignments from the CryoET Data Portal.",
    no_args_is_help=True,
)
@click.option(
    "-ds",
    "--dataset",
    required=True,
    type=str,
    help="Dataset ID to download from the CryoET Data Portal.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    default=".",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory to save the downloaded files.",
)
def project(dataset: str, output: str):
    """
    Download tilt series and alignments from the CryoET Data Portal.

    Fetches the AreTomo files (the raw tilt series plus their alignment files) for every
    run in a CryoET Data Portal dataset and writes them to a local directory, laid out for
    sub-tomogram averaging with py2rely. Pass the numeric portal dataset ID via `--dataset`;
    all matching tilt series are downloaded in parallel into `--output`.

    Examples:

        \b
        # Download a dataset's tilt series and alignments into the current directory
        copick download project -ds 10000 -o .

        \b
        # Download into a dedicated output directory
        copick download project --dataset 10445 --output ./aretomo_inputs

    See Also:

        \b
        copick config dataportal: build a copick config from CryoET Data Portal datasets
    """
    download_project(dataset, output)


def download_project(dataset: str, output: str):
    import copick_utils.io.portal as portal

    portal.download_aretomo_files(dataset, output)
