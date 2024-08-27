import gdown
import zipfile
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="""Downloads the dataset to a given directoy path.""")
    parser.add_argument('-p', '--segments_path', type=Path,
                        default=Path(__file__).parents[1] / "assets",
                        help="Directory where the segments will be downloaded and unzipped.")
    args = parser.parse_args()

    # doẃnload and unzip plan
    zippath = Path(args.segments_path / "segments.zip")
    if not zippath.exists():
        url = "https://drive.google.com/uc?id=1a1zwuuvaDVfmovGbcEsazfxr7OLfusLM"
        gdown.download(url, str(zippath), quiet=False)
        with zipfile.ZipFile(str(zippath), 'r') as zip_ref:
            zip_ref.extractall(str(args.segments_path / "segments"))
    else:
        raise RuntimeError(f"The segments dataset already exists in the path: {zippath}")
