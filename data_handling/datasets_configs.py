from pathlib import Path

DATA_DIR = Path("data")


DATASET_CONFIGS = {
    "3droad": {
        "source": "uci",
        "id": "246",
        "download_url": "https://archive.ics.uci.edu/static/public/246/3d+road+network+north+jutland+denmark.zip",
        "data_url": None,
        "num_instances": 434874,
        "num_features": 2,
        "target_columns": [3],
        "ignore_columns": [0]
    },
    "song": {
        "source": "uci",
        "id": "203",
        "download_url": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
        "data_url": None,
        "num_instances": 515345,
        "num_features": 90,
        "target_columns": [0],
        "ignore_columns": []
    },
    "buzz": { # Needs Clarification: See https://lig-aptikal.imag.fr/buzz-prediction-in-online-social-media/
        "source": "uci",
        "id": "248",
        "download_url": "https://archive.ics.uci.edu/static/public/248/buzz+in+social+media.zip",
        "data_url": None,
        "num_instances": 140000,
        "num_features": 0,     
        "target_columns": None,
        "ignore_columns": []
    },
    "houseelec": {
        "source": "uci",
        "id": "235",
        "download_url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "data_url": "https://archive.ics.uci.edu/static/public/235/data.csv",
        "num_instances": 2075259,
        "num_features": 8,
        "target_columns": ["Global_active_power"],   # In Lin et al.
        "ignore_columns": []
    },

    "acsincome": {
        "source": "openml",
        "id": 43141,
        "num_instances": 1664500,
        "num_features": 11,
        "target_columns": ["target"],
        "ignore_columns": [],
    },
    "benzene": {
        "source": "sgdml",
        "download_url": "http://www.quantum-machine.org/gdml/data/npz/md17_benzene2017.npz",
        "num_instances": 627983,
        "num_features": 66,
        "target_columns": ["E"],
        "ignore_columns": [],
    },
    "malonaldehyde": {
        "source": "sgdml",
        "download_url": "http://www.quantum-machine.org/gdml/data/npz/md17_malonaldehyde.npz",
        "num_instances": 993237,
        "num_features": 36,
        "target_columns": ["E"],
        "ignore_columns": [],
    },
}