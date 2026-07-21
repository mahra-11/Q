"""
Fraction of native contacts (Q) reaction coordinate.

Implements the Best-Hummer-Eaton (2013, PNAS) smooth native-contact metric:

    Q(X) = (1/N) * sum_{(i,j) in native contacts}
               1 / (1 + exp(beta * (r_ij(X) - lambda * r_ij(native))))

Q = 1 means every native contact is formed (fully folded); Q = 0 means none
are. Unlike RMSD, Q is a standard, structure-agnostic order parameter widely
used to define folded/unfolded ensembles in protein folding studies.
"""

import itertools

import mdtraj as md
import numpy as np

BETA_CONST = 50.0     # 1/nm, sharpness of the contact switching function
LAMBDA_CONST = 1.8    # unitless tolerance on native distance
NATIVE_CUTOFF = 0.45  # nm, max heavy-atom distance to count as a native contact
MIN_RESIDUE_SEP = 3   # exclude contacts between residues closer than this in sequence


def native_contact_pairs(native, min_residue_sep=MIN_RESIDUE_SEP, cutoff=NATIVE_CUTOFF):
    """
    Identify native heavy-atom contact pairs from a reference (folded) structure.

    Parameters
    ----------
    native : mdtraj.Trajectory
        Single-frame reference structure (e.g. the folded PDB).
    min_residue_sep : int
        Minimum residue-index separation for a pair to be considered a contact.
    cutoff : float
        Maximum heavy-atom distance (nm) in the native structure to count as a contact.

    Returns
    -------
    np.ndarray, shape (n_contacts, 2)
        0-based atom index pairs defining the native contacts.
    """
    heavy = native.topology.select_atom_indices("heavy")
    pairs = np.array([
        (i, j) for i, j in itertools.combinations(heavy, 2)
        if abs(native.topology.atom(i).residue.index
               - native.topology.atom(j).residue.index) > min_residue_sep
    ])
    distances = md.compute_distances(native, pairs)[0]
    contacts = pairs[distances < cutoff]
    if len(contacts) == 0:
        raise ValueError(
            f"No native contacts found within cutoff={cutoff} nm "
            f"(min heavy-atom distance in `native` was {distances.min():.3f} nm). "
            "Check that `native` is a single well-folded reference conformation, "
            "or loosen `cutoff` / `min_residue_sep` — small peptides (e.g. Chignolin's "
            "10 residues) may need a larger cutoff than the default 0.45 nm."
        )
    return contacts


def best_hummer_q(traj, native, pairs=None,
                   beta=BETA_CONST, lam=LAMBDA_CONST, cutoff=NATIVE_CUTOFF):
    """
    Compute the fraction of native contacts Q for each frame of `traj`.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Trajectory to evaluate Q on.
    native : mdtraj.Trajectory
        Single-frame reference (folded) structure.
    pairs : np.ndarray, optional
        Precomputed native contact pairs (see `native_contact_pairs`).
        Computed from `native` if not given.
    beta, lam, cutoff : float
        Best-Hummer-Eaton parameters.

    Returns
    -------
    np.ndarray, shape (n_frames,)
        Q value per frame.
    """
    if pairs is None:
        pairs = native_contact_pairs(native, cutoff=cutoff)
    r0 = md.compute_distances(native, pairs)[0]
    r = md.compute_distances(traj, pairs)
    return np.mean(1.0 / (1.0 + np.exp(beta * (r - lam * r0))), axis=1)


def compute_q(traj_files, topology_pdb, native_pdb=None, stride=1):
    """
    Load a trajectory with MDTraj and compute Q per frame against a native structure.

    Parameters
    ----------
    traj_files : str or list of str
        Trajectory file(s) (.xtc).
    topology_pdb : str
        Topology file matching `traj_files`.
    native_pdb : str, optional
        Reference (folded) structure defining native contacts. Defaults to
        `topology_pdb`, which is already the folded reference structure for
        both Chignolin and WW domain in this repo's configs.
    stride : int
        Load every `stride`-th frame.

    Returns
    -------
    q : np.ndarray, shape (n_frames,)
    n_native_contacts : int
    """
    traj = md.load(traj_files, top=topology_pdb, stride=stride)
    native = md.load(native_pdb) if native_pdb else md.load(topology_pdb)
    pairs = native_contact_pairs(native)
    q = best_hummer_q(traj, native, pairs=pairs)
    return q, len(pairs)


def compute_q_sample(traj_files, topology_pdb, native_pdb=None, n_frames=2000, stride=1):
    """
    Compute Q on just the first `n_frames` frames of a trajectory, for a
    quick trial run on a large trajectory without reading the whole file.

    Same result semantics as `compute_q`, but reads only the first chunk
    from disk via `mdtraj.iterload` instead of loading everything into memory.

    Parameters
    ----------
    traj_files : str or list of str
        Trajectory file(s). If a list, only the first file is sampled.
    topology_pdb : str
        Topology file matching `traj_files`.
    native_pdb : str, optional
        Reference (folded) structure defining native contacts. Defaults to
        `topology_pdb`.
    n_frames : int
        Number of frames to sample (after striding).
    stride : int
        Only read every `stride`-th frame while sampling.

    Returns
    -------
    q : np.ndarray, shape (<= n_frames,)
    n_native_contacts : int
    """
    if isinstance(traj_files, (list, tuple)):
        traj_files = traj_files[0]

    native = md.load(native_pdb) if native_pdb else md.load(topology_pdb)
    pairs = native_contact_pairs(native)

    chunk = next(md.iterload(traj_files, top=topology_pdb, chunk=n_frames, stride=stride))
    q = best_hummer_q(chunk, native, pairs=pairs)
    return q, len(pairs)
