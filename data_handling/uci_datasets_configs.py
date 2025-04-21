from pathlib import Path

DATA_DIR = Path("data")


DATASET_CONFIGS = {
    "pol": {
        "id": "365",
        "download_url": "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/365/data.csv",
        "num_instances": 10503,
        "num_features": 65,
        "target_columns": None,
        "ignore_columns": []
    },
    "elevators": { # Needs Clarification: This dataset cannot be found in the UCI repo.
        "id": None,
        "download_url": None,
        "data_url": None,
        "num_instances": None,
        "num_features": None,
        "target_columns": None,
        "ignore_columns": []
    },
    "bike": {
        "id": "00275",
        "download_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/275/data.csv",
        "num_instances": 17389,
        "num_features": 13,
        "target_columns": None,
        "ignore_columns": []
    },
    "protein": {
        "id": "265",
        "download_url": "https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip",
        "data_url": None,
        "num_instances": 45730,
        "num_features": 9,
        "target_columns": ["RMSD"],
        "ignore_columns": []
    },
    "keggdir": { # Needs Clarification: Not clear (in Lin et al.) which feature is set as target and which columns are dropped.
        "id": "220",
        "download_url": "https://archive.ics.uci.edu/static/public/220/kegg+metabolic+relation+network+directed.zip",
        "data_url": None,
        "num_instances": 53414,
        "num_features": 20,
        "target_columns": None,
        "ignore_columns": []
    },
    "3droad": {
        "id": "246",
        "download_url": "https://archive.ics.uci.edu/static/public/246/3d+road+network+north+jutland+denmark.zip",
        "data_url": None,
        "num_instances": 434874,
        "num_features": 2,
        "target_columns": [3],
        "ignore_columns": [0]
    },
    "song": {
        "id": "203",
        "download_url": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
        "data_url": None,
        "num_instances": 515345,
        "num_features": 90,
        "target_columns": [0],
        "ignore_columns": []
    },
    "buzz": { # Needs Clarification: See https://lig-aptikal.imag.fr/buzz-prediction-in-online-social-media/
        "id": "248",
        "download_url": "https://archive.ics.uci.edu/static/public/248/buzz+in+social+media.zip",
        "data_url": None,
        "num_instances": 140000,
        "num_features": 0,     
        "target_columns": None,
        "ignore_columns": []
    },
    "houseelec": {
        "id": "235",
        "download_url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/235/data.csv",
        "num_instances": 2075259,
        "num_features": 8,
        "target_columns": ["Global_active_power"],   # In Lin et al.
        "ignore_columns": []
    }
}