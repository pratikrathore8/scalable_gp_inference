# Turbocharging GP Inference

Companion code for "Turbocharing Gaussian Process Inference with Approximate Sketch-and-Project".

## Setting up the environment
Make sure you are using Python 3.10 or later.
To set up the environment, run the following command in a `python` virtual environment:
```bash
pip install -e .
```

Please make sure that `pykeops` and `keopscore` are both installed with version `2.2.3`. We have found version `2.3` to be too slow for some of our experiments (particularly Bayesian optimization).

If you are developing the code, you should also run
```bash
pre-commit install
```

## Reproducing the experiments
