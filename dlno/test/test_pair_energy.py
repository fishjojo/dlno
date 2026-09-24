"""Tests for the multipole estimate of distant-pair MP2 energies.

``mp2.pair_energy_multipole`` returns, for every pair of LMOs i != j, the
MP2 energy of the unordered pair {i, j}, opposite-spin plus same-spin with
exchange neglected: E_ij = -4 S_ij with

    S_ij = sum_{a in [i], b in [j]} (ai|bj)^2 / (e_a + e_b - F_ii - F_jj).

The kernel adds 0.5 * sum(E) over distant entries, i.e. each distant pair once.
"""
import numpy as np
from pyscf import gto, scf, mp, df, lib
from pyscf.mp import dfmp2
from dlno import dlno, util
from dlno.mp2 import pair_energy_multipole


def _tight_scf(mf):
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    return mf


def _atom_localized_lmo(mf):
    """1s orbitals on each He of He2. Pipek-Mezey stalls at the symmetric sigma_g/sigma_u start."""
    return mf.mo_coeff[:, :2] @ (np.array([[1, 1], [1, -1]]) / np.sqrt(2))


def test_pair_energy_he2_supermolecular():
    """E_ij equals the MP2 pair energy E(He2) - 2 E(He), which fixes the prefactor.

    He has one occupied orbital, so the dimer's only inter-atomic pair is {1s_A, 1s_B}.
    """
    R = 6.0
    mf = _tight_scf(scf.RHF(gto.M(atom=f'He 0 0 0; He 0 0 {R}', basis='ccpvdz', verbose=0)))
    e_dimer = mp.MP2(mf).kernel()[0]
    mf_atom = _tight_scf(scf.RHF(gto.M(atom=f'He 0 0 0; ghost-He 0 0 {R}', basis='ccpvdz', verbose=0)))
    e_atom = mp.MP2(mf_atom).kernel()[0]
    e_pair = e_dimer - 2 * e_atom

    mydlno = dlno.DLNO(mf)
    mydlno.lmo = _atom_localized_lmo(mf)
    assert sorted(list(d) for d in mydlno.lmo_bp_domain) == [[0], [1]]
    (eo, vo), (ev, vv) = mydlno.canonicalize(mydlno.build_domain_pao())
    E = pair_energy_multipole(mf.mol, eo, vo, ev, vv, mydlno.lmo_primary_domain, 4)

    assert abs(E[0, 1] - E[1, 0]) < 1e-16
    assert abs(E[0, 1] / e_pair - 1) < 1e-3


def _water_dimer(R):
    """Asymmetric water of test_lmp2.py, plus a copy rotated 90 degrees about y and shifted by R along x."""
    water = np.array([[0.0, 0.0, 0.0], [-0.38326, 0.87702, 0.0], [0.95133, 0.15523, 0.0]])
    rot_y = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    coords = np.vstack([water, water @ rot_y.T + np.array([R, 0.0, 0.0])])
    atoms = [(s, tuple(x)) for s, x in zip(['O', 'H', 'H'] * 2, coords)]
    return gto.M(atom=atoms, basis='ccpvdz', verbose=0)


def test_pair_energy_multipole_vs_exact_integrals():
    """The multipole expansion reproduces exact DF integrals in the same domain orbitals.

    This is the check that is sensitive to the dipole-quadrupole (R^-4) term: with that
    term's sign reversed, the sum below is off by about 5% at this distance.
    """
    mol = _water_dimer(8.0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()

    mydlno = dlno.DLNO(mf)
    bp = mydlno.lmo_bp_domain
    nocc = len(bp)
    on_a = [i for i in range(nocc) if max(bp[i]) <= 2]
    on_b = [i for i in range(nocc) if min(bp[i]) >= 3]
    assert len(on_a) == len(on_b) == nocc // 2

    (eo, vo), (ev, vv) = mydlno.canonicalize(mydlno.build_domain_pao())
    pd = mydlno.lmo_primary_domain
    E = pair_energy_multipole(mol, eo, vo, ev, vv, pd, 4)

    Lpq = lib.unpack_tril(np.asarray(mf.with_df._cderi))
    Lai = []
    for i in range(nocc):
        ao = util.ao_index_by_atom(mol, pd[i])
        Lai.append(np.einsum('Luv,ua,v->La', Lpq[:, ao][:, :, ao], vv[i], vo[i].ravel()))

    e_multipole = e_exact_int = 0.0
    for i in on_a:
        for j in on_b:
            aibj = Lai[i].T @ Lai[j]
            denom = (ev[i] - eo[i])[:, None] + (ev[j] - eo[j])[None, :]
            e_exact_int += -4 * np.sum(aibj**2 / denom)
            e_multipole += E[i, j]

    assert abs(e_multipole / e_exact_int - 1) < 5e-3


def test_distant_pair_in_kernel_he2():
    """With the He-He pair forced distant, the kernel reproduces canonical DF-MP2.

    Each fragment then holds one He atom, and the whole inter-atomic energy, about
    -2.7e-7 Eh at 5 A, comes from the multipole term. Counting that pair twice or
    half would be off by the full pair energy.
    """
    aux = df.autoaux(gto.M(atom='He 0 0 0', basis='ccpvdz', verbose=0))   # cc-pvdz-jkfit has no He
    mol = gto.M(atom='He 0 0 0; He 0 0 5.0', basis='ccpvdz', verbose=0)
    mf = _tight_scf(scf.RHF(mol).density_fit(auxbasis=aux))
    e_canonical = dfmp2.DFMP2(mf).kernel()[0]

    mydlno = dlno.DLNO(mf)
    mydlno.lmo = _atom_localized_lmo(mf)
    mydlno.pair_energy_thr = 0.5              # every off-diagonal pair is distant
    e_local = mydlno.kernel(auxbasis=aux)

    assert abs(e_local - e_canonical) < 1e-8
