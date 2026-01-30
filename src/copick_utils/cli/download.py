import click

@click.command(
    context_settings={"show_default": True},
    short_help="Download tilt series and alignments from the CryoET Data Portal.",
    no_args_is_help=True,
)
@click.option(
    '-d', '--dataset', 
    required=True, type=str,
    help='Dataset ID to download from the CryoET Data Portal.',
)
@click.option(
    '-o', '--output',
    required=True, default='.', type=str,
    help='Output directory to save the downloaded files.',
)
def project(dataset: str, output: str):
    """
    Download tilt series and alignments from the CryoET Data Portal for sub-tomogram averaging with py2rely.
    """
    download_project(dataset, output)

def download_project(dataset: str, output: str):
    import copick_utils.io.portal as portal
    portal.download_aretomo_files(dataset, output)