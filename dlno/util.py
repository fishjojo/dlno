from functools import reduce, partial
import numpy as np
try:
    from opt_einsum import contract
except ImportError:
    contract = partial(np.einsum, optimize=True)
from pyscf import lib

def project_mo(mo1, s21, s22):
    """Project ``mo1`` to basis ``2``.
    """
    return lib.cho_solve(s22, s21 @ mo1, strict_sym_pos=False)

def orthogonalize(mo1, mo2, s):
    """Project ``mo1`` out of ``mo2``.

    Notes
    -----
    ``mo1`` must be orthonormal.
    """
    s12 = mo1.conj().T @ s @ mo2
    mo2 = mo2 - mo1 @ s12
    return mo2

def ao_index_by_atom(mol, atmlst):
    aoslices = mol.aoslice_by_atom()[:,2:]
    ao_idx_lst = map(lambda x: np.arange(*x), aoslices[atmlst].reshape(-1,2))
    ao_idx = reduce(np.union1d, ao_idx_lst)
    return ao_idx

def shell_index_by_atom(mol, atmlst):
    shlslices = mol.aoslice_by_atom()[:,:2]
    shls_lst = map(lambda x: np.arange(*x), shlslices[atmlst].reshape(-1,2))
    shls = reduce(np.union1d, shls_lst)
    return shls

def fake_mol_by_atom(mol, atmlst=None):
    if atmlst is not None:
        shls = shell_index_by_atom(mol, atmlst)
        fake_mol = mol.copy()
        fake_mol._bas = fake_mol._bas[shls]
    else:
        fake_mol = mol
    return fake_mol

def unique(a):
    unique_arr = {}
    for i, arr in enumerate(a):
        arr_tuple = tuple(arr)
        if arr_tuple not in unique_arr:
            unique_arr[arr_tuple] = [i]
        else:
            unique_arr[arr_tuple].append(i)
    return unique_arr

def list_to_array(a):
    out = np.empty(len(a), dtype=object)
    out[:] = a
    return out

def einsum(expr, *args):
    return contract(expr, *args)

