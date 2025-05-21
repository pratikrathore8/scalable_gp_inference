# Turbocharging GP Inference

Companion code for [***"Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project"***](https://arxiv.org/abs/2505.13723).

## Setting up the environment
Make sure you are using Python 3.10 or later.
To set up the environment, clone the repo, `cd` into the root of the repo, and run the following command in a `python` virtual environment:
```bash
pip install -e .
```

Please make sure that `pykeops` and `keopscore` are both installed with version `2.2.3`. We have found version `2.3` to be too slow for some of our experiments (particularly Bayesian optimization).

If you are developing the code, you should also run
```bash
pre-commit install
```

## Instructions for reproducing our experiments
Below, we provide an overview of the steps needed to reproduce our results.
Our experiments are run on up to three 48 GB NVIDIA A6000 GPUs.
The experiments are configured to automatically log to Weights & Biases.

### Downloading the datasets
Run `python experiments/download_data.py` to download all datasets (besides taxi).

To get the taxi dataset, please follow the instructions in this fork of the [nyc-taxi-data repo](https://anonymous.4open.science/r/nyc-taxi-data). Run `filter_runs.py` and `yellow_taxi_processing.sh` (NOTE: you may have to turn off the move to Google Drive step in this shell script) in this repo.

This shell script will generate a `.h5py` file for each month from January 2009 to December 2015. Move these files to a new folder `scalable_gp_inference/data/taxi` and run `python experiments/taxi_processing.py` to process the data.

### Running benchmark GP inference experiments
Activate your python environment and run the following command:
```bash
./run_gp_inference_experiments.sh <seed>
```

This will run the experiments with the specified seed on the devices specified in the `run_gp_inference_experiments.sh` script. You may have to change the device IDs in the script to match your setup. We run our experiments using 2 GPUs and seeds 0, 1, 2, 3, and 4.

### Running the taxi experiments
Activate your python environment and run the following command:
```bash
./run_gp_inference_experiments_taxi.sh <seed>
```

This will run the experiments with the specified seed on the devices specified in the `run_gp_inference_experiments_taxi.sh` script. You may have to change the device IDs in the script to match your setup. We run our experiments using 3 GPUs and seed 0.

### Running the taxi timing experiments
Activate your python environment and run the following command:
```bash
python experiments/gp_inference_dataset_seed.py --dataset taxi --seed <seed> --devices <device_ids> --timing
```

This will run the timing experiments on the taxi dataset with the specified seed on the devices specified in the command. You may have to change the device IDs in the command to match your setup. We run our experiments using 1, 2, 3, and 4 GPUs and seed 0.

### Running the Bayesian optimization experiments
Activate your python environment and run the following command:
```bash
python experiments/bayes_opt.py --lengthscale <lengthscale> --seed <seed> --device <device_id>
```

This will run the Bayesian optimization experiments with the specified lengthscale and seed on the device specified in the command. We run our experiments using lengthscales 2.0 and 3.0, 1 GPU, and seeds 0, 1, 2, 3, and 4.

## Citation

If you find our work useful, please consider citing our paper:

```
@article{rathore2025turbocharging,
  title={Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project},
  author={Pratik Rathore and Zachary Frangella and Sachin Garg and Shaghayegh Fazliani and Michał Dereziński and Madeleine Udell},
  journal={arXiv preprint arXiv:2505.13723},
  year={2025}
}
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
