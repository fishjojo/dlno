from functools import reduce
import numpy as np
from pyscf.gto.mole import inter_distance
from dlno import util

def get_bp_domain(mol, mos, s1e=None, bp_thr=0.9999,
                  q_thr=None, atmlst=None):
    """BP domains based on partial Mulliken charges.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5*(1-bp_thr))
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1,1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    # TODO project MOs onto smaller basis
    assert nao == mol.nao

    rr = atom_distance(mol, atmlst)
    aoslices = mol.aoslice_by_atom()[:,2:]
    bp_atmlst = []

    for i in range(nmo):
        orbi = mos[:,i]
        PS = np.outer(orbi, orbi) * s1e
        GOP = np.sum(PS, axis=1)

        # FIXME Mulliken charge can be negative
        q = abs(np.asarray([GOP[slice(*aoslices[a])].sum() for a in atmlst]))

        _atms = atmlst[q > q_thr]
        av = _compute_av(mol, orbi, s1e, _atms)

        if av < bp_thr:
            center_id = np.argsort(-q)[0]
            _sorted_atm_idx = np.argsort(rr[center_id])

            for iatm in _sorted_atm_idx[1:]:
                a = atmlst[iatm]
                if a not in _atms:
                    _atms = np.append(_atms, a)
                    av = _compute_av(mol, orbi, s1e, _atms)
                    if av >= bp_thr:
                        break
        bp_atmlst.append(_atms)

    return util.list_to_array(bp_atmlst)


def get_primary_domain(mol, lmo_bp_domain, pao_bp_domain, ao2pao_map=None):
    """Extend LMO BP domain by PAO BP domains.
    """
    if ao2pao_map is None:
        ao2pao_map = np.arange(mol.nao)

    nocc = len(lmo_bp_domain)
    aoslices = mol.aoslice_by_atom()[:,2:]
    pd_atmlst = []

    for i in range(nocc):
        _atms = np.empty((0,), dtype=np.int32)
        for a in lmo_bp_domain[i]:
            _tmp = ao2pao_map[slice(*aoslices[a])]
            pao_idx = _tmp[_tmp >= 0]
            _atms = np.union1d(_atms, reduce(np.union1d, pao_bp_domain[pao_idx]))
        pd_atmlst.append(np.union1d(lmo_bp_domain[i], _atms))

    return util.list_to_array(pd_atmlst)


def _compute_av(mol, mo, s1e=None, atmlst=None):
    """Compute BP value
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mo = np.asarray(mo)

    ao_idx = util.ao_index_by_atom(mol, atmlst)
    v = s1e[ao_idx] @ mo
    a = np.linalg.solve(s1e[np.ix_(ao_idx, ao_idx)], v)
    av = np.sum(a * v, axis=0)
    return av


def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    if atmlst is None:
        atmlst = np.arange(mol.natm)
    coords = mol.atom_coords()[atmlst].reshape(-1,3)
    return inter_distance(mol, coords=coords)

