import icedata

from .config import Config


def get_data(url: str = Config.data_url, dest_dir: str = Config.dest_dir) -> None:
    """[download the trainval data]

    Keyword Arguments:
        url {str} -- [the url to download data from] (default: {Config.data_url})
        dest_dir {str} -- [directory to save data in] (default: {Config.dest_dir})
    """
    icedata.load_data(url, dest_dir, force_download=Config.force_data_download)


if __name__ == "__main__":
    get_data()
