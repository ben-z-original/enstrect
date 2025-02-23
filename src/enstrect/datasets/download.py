import gdown
import zipfile
from pathlib import Path
from argparse import ArgumentParser


def download(name, url, target_path):
    # doẃnload and unzip plan
    target_path.mkdir(exist_ok=True)
    zippath = target_path / f"{name}.zip"
    if not zippath.exists():
        gdown.download(url, str(zippath), quiet=False)
        with zipfile.ZipFile(str(zippath), 'r') as zip_ref:
            zip_ref.extractall(str(target_path / name))
    else:
        raise RuntimeError(f"The dataset already exists in the path: {zippath}")


if __name__ == "__main__":
    parser = ArgumentParser(description="""Downloads the dataset to a given directoy path.""")
    parser.add_argument('-p', '--target_path', type=Path,
                        default=Path(__file__).parents[1] / "assets",
                        help="Directory where the dataset will be downloaded and unzipped.")
    args = parser.parse_args()
    
    download(name="segments",
             url="https://drive.google.com/uc?id=1QkyoZ1o9uKuxpLIlSZ-iA9jcba46oIwW",
             target_path=args.target_path)
