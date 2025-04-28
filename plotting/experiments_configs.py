"""Central configuration for scalable GP experiments."""

# -----------------------------------------------------------------------------
# DATASETS and GP HYPERPARAMS
# -----------------------------------------------------------------------------

runs: dict[str, dict] = {
    "datasets": {
        "acsincome": {
            "dataset_configs": {
                "loader": "acsincome",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "rbf",
                "nu": None,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }
        },
        "yolanda": {
            "dataset_configs": {
                "loader": "yolanda",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "rbf",
                "nu": None,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }
        },
        "malonaldehyde": {
            "dataset_configs": {
                "loader": "malonaldehyde",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "matern",
                "nu": 5 / 2,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }
        },
        "benzene": {
            "dataset_configs": {
                "loader": "benzene",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "matern",
                "nu": 5 / 2,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }
        },
        "taxi": {
            "dataset_configs": {
                "loader": "taxi",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "rbf",
                "nu": None,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }

        },
        "song": {
            "dataset_configs": {
                "loader": "song",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "matern",
                "nu": 3 / 2,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }

        },

        "3droad": {
            "dataset_configs": {
                "loader": "3droad",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "matern",
                "nu": 3 / 2,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }

        },

        "houseelec": {
            "dataset_configs": {
                "loader": "houseelec",
                "split_proportion": 0.9,
                "split_shuffle": True,
                "split_seed": 0,
                "standardize": True,
                "dtype": "float32",
                "device": "cpu",
            },

            "gp_hyperparams": {
                "kernel_type": "matern",
                "nu": 3 / 2,
                "use_full_kernel": True,
                "signal_variance": 1.0,
                "kernel_lengthscale": 1.0,
                "noise_variance": 1e-2,
            }

        },

    },


    # -------------------------------------------------------------------------
    # SOLVER configurations
    # -------------------------------------------------------------------------
    "solvers": {

        "pcg": {
            "class": "rlaopt.solvers.PCGConfig",
            "kwargs": {
                "max_iters": 500,
                "atol": 1e-6,
                "rtol": 1e-6,

                "precond_config": {
                    "class": "rlaopt.preconditioners.NystromConfig",
                    "kwargs": {
                        "rank": 1000,
                        "rho": 1e-2,
                        "damping_mode": "adaptive",
                    },
                },
            },
        },

        "sap": {
            "class": "rlaopt.solvers.SAPConfig",
            "kwargs": {
                "max_iters": 500,
                "atol": 1e-6,
                "rtol": 1e-6,
            },
        },
    },
}
