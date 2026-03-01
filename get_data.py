import kagglehub
import os
import click


@click.command()
@click.option("--path", default="data", help="Path to save the dataset")
def download_dataset(path: str) -> None:
    click.echo("Donwloading Dataset...")

    if not os.path.exists(path):
        click.echo("No such path or directory. Creating the directory...")
        os.makedirs(path, exist_ok=True)
    else:
        click.echo("Directory already exists. Proceeding to download...")

    save_path = os.path.join(".", path)
    kagglehub.dataset_download(
        "verack/spanish-handwritten-characterswords",
        force_download=False,
        output_dir=save_path,
    )

    click.echo(f"Path to dataset files: {path}")


if __name__ == "__main__":
    download_dataset()
