"""Code for getting features from molecules."""
from __future__ import annotations
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from typing import Dict

FP_Dict = Dict[int, int]


def _smiles_to_mols(
    smiles_list: list[str], n_jobs: Optional[int] = None
) -> list[Chem.Mol]:
    """
    Helper function to convert list of SMILES to list of mols,
    raising an error if any invalid SMILES are found.
    """

    # Define a separate function since rdkit functions cannot be pickled by joblib
    def mol_from_smiles(s):
        return Chem.MolFromSmiles(s)

    if n_jobs is None:
        mols = [mol_from_smiles(s) for s in smiles_list]
    else:
        mols = Parallel(n_jobs=n_jobs)(delayed(mol_from_smiles)(s) for s in smiles_list)

    assert not any(m is None for m in mols)
    return mols


def mol_to_fp_dict(
    mols: list[Chem.Mol],
    radius: int,
    use_counts: bool = True,
    n_jobs: Optional[int] = None,
) -> list[FP_Dict]:
    """Get Morgan fingerprint bit dict from a list of mols."""

    # Define a separate function since rdkit functions cannot be pickled by joblib
    def fp_func(mol):
        return rdMolDescriptors.GetMorganFingerprint(
            mol, radius=radius, useCounts=use_counts
        ).GetNonzeroElements()

    if n_jobs is None:
        fps = [fp_func(mol) for mol in mols]
    else:
        fps = Parallel(n_jobs=n_jobs)(delayed(fp_func)(mol) for mol in mols)

    return fps


def fp_dicts_to_arr(
    fp_dicts: list[FP_Dict], nbits: int, binarize: bool = False
) -> np.ndarray:
    """Convert a list of fingerprint dicts to a numpy array."""

    # Fold fingerprints into array
    out = np.zeros((len(fp_dicts), nbits))
    for i, fp in enumerate(fp_dicts):
        for k, v in fp.items():
            out[i, k % nbits] += v

    # Potentially binarize
    if binarize:
        out = np.minimum(out, 1.0)
        assert set(np.unique(out)) <= {0.0, 1.0}

    return out


def mol_to_fingerprint_arr(
    mols: list[Chem.Mol], nbits: int, binarize: bool = False, **kwargs
) -> np.ndarray:
    """Returns a fingerprint mapped into a numpy array."""
    fp_dicts = mol_to_fp_dict(mols=mols, **kwargs)
    return fp_dicts_to_arr(fp_dicts, nbits=nbits, binarize=binarize)


def smiles_to_fingerprint_arr(
    smiles_list: list[str], n_jobs: Optional[int] = None, **kwargs
) -> np.array:
    mol_list = _smiles_to_mols(smiles_list, n_jobs=n_jobs)
    return mol_to_fingerprint_arr(mols=mol_list, n_jobs=n_jobs, **kwargs)
