from .config import Config
from .get_data import get_data
from .mycocoparser import SubCocoParser, load_stats


def do_parsing():
    """
    downloads the data, and parses it according to min bbox height-width margins
    """
    get_data()
    stats = load_stats(
        Config.annotation_file,
        img_dir=Config.img_dir,
        force_reload=Config.force_parser_stats_reload,
    )
    parser = SubCocoParser(
        stats,
        min_margin_ratio=Config.parser_min_margin_ratio,
        min_width_height_ratio=Config.parser_min_width_height_ratio,
    )
    # Records
    train_records, valid_records = parser.parse(autofix=True)
    print(
        f"#records(train) : {len(train_records)}, #records(valid) : {len(valid_records)}"
    )
    return train_records, valid_records
