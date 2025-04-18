from pathlib import Path

DATA_DIR = Path("data")
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

DATASET_CONFIGS = {
    "pol": {
        "id": "365",
        "download_url": "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/365/data.csv",
        "label_columns": None,

    },
    "elevators": {
        "id": "unk",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00326/eleclass.zip",
        "data_url": None,
        "label_columns": None
    },
    "bike": {
        "id": "00275",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/275/data.csv",
        "label_columns": None
    },
    "protein": {
        "id": "265",
        "download_url": "https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip",
        "data_url": None,
        "label_columns": None
    },
    "keggdir": {
        "id": "220",
        "download_url": "https://archive.ics.uci.edu/static/public/220/kegg+metabolic+relation+network+directed.zip",
        "data_url": None,
        "label_columns": None
    },
    "3droad": {
        "id": "246",
        "download_url": "https://archive.ics.uci.edu/static/public/246/3d+road+network+north+jutland+denmark.zip",
        "data_url": None,
        "label_columns": None
    },
    "song": {
        "id": "203",
        "download_url": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
        "data_url": None,
        "label_columns": None
    },
    "buzz": {
        "id": "248",
        "download_url": "https://archive.ics.uci.edu/static/public/248/buzz+in+social+media.zip",
        "data_url": None,
        "label_columns": None
    },
    "houseelec": {
        "id": "235",
        "download_url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/235/data.csv",
        "label_columns": None
    }
}