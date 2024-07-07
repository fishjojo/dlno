from functools import partial
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import dfmp2
from dlno.multipole import *
from dlno.util import einsum

WITH_T2 = True

def pair_energy_multipole(
        mol,
        e_occ,
        mo_occ,
        e_vir,
        mo_vir,
        atmlst=None,
        order=3,
    ):
    """Multipole approximation to the OS-MP2 pair energy.

    Parameters
    ----------
    e_occ : list
        Occupied orbital energies.
    mo_occ : list
        Occupied orbital coefficients.
    e_vir : list
        Virtual orbital energies for each occupied orbital.
    mo_vir : list
        Virtual orbital coefficients for each occupied orbital.
    atmlst : array
        Atoms on which the basis functions are used
        to compute the operators, for each occupied orbital.
    order : int, default=3
        Multipole expansion orders, can be 2, 3, and 4.

    Returns
    -------
    e_mp2_pair : array
        OS-MP2 pair energy.
    """
    nocc = len(e_occ)
    if atmlst is None:
        atmlst = [None,] * nocc
    e_mp2_pair = np.zeros((nocc, nocc))

    Rs = []
    mu_vo = []
    theta_vo = []
    omega_vo = []
    e_vo = []

    for i in range(nocc):
        lmo_i = mo_occ[i].ravel()
        atmlst_i = atmlst[i]
        Di = dipole_op(mol, atmlst=atmlst_i)
        Ri = einsum('u,xuv,v->x', lmo_i.conj(), Di, lmo_i)
        Rs.append(Ri)

        pao_i = mo_vir[i]
        mu_ai = einsum('ua,xuv,v->xa', pao_i.conj(), Di, lmo_i)
        mu_vo.append(mu_ai)

        e_ai = e_vir[i] - e_occ[i]
        e_vo.append(e_ai)

        if order > 2:
            Qi = quadrupole_op(mol, R=Ri, atmlst=atmlst_i)
            theta_ai = einsum('ua,xyuv,v->xya', pao_i.conj(), Qi, lmo_i)
            theta_vo.append(theta_ai)

        if order > 3:
            Oi = octupole_op(mol, R=Ri, atmlst=atmlst_i)
            omega_ai = einsum('ua,xyzuv,v->xyza', pao_i.conj(), Oi, lmo_i)
            omega_vo.append(omega_ai)

        for j in range(i):
            Rj = Rs[j]
            R = np.linalg.norm(Rj - Ri)
            R_bar = (Rj - Ri) / R

            mu_bj = mu_vo[j]
            aibj_2 = mu_ai.T @ mu_bj
            tmp_ai = R_bar @ mu_ai
            tmp_bj = R_bar @ mu_bj
            aibj_2 -= np.outer(tmp_ai, tmp_bj * 3)
            aibj_2 /= R**3

            if order > 2:
                theta_bj = theta_vo[j]
                RR = np.outer(R_bar, R_bar)

                tmp1_ai = RR.ravel() @ theta_ai.reshape(9,-1)
                tmp1_bj = RR.ravel() @ theta_bj.reshape(9,-1)
                aibj_3  = np.outer(tmp1_ai, tmp_bj * 5)
                aibj_3 -= np.outer(tmp_ai, tmp1_bj * 5)

                mu_R_ai = einsum('xa,y->xya', mu_ai, R_bar).reshape(9,-1)
                mu_R_bj = einsum('xb,y->xyb', mu_bj, R_bar).reshape(9,-1)
                aibj_3 += (2 * mu_R_ai.T) @ theta_bj.reshape(9,-1)
                aibj_3 -= theta_ai.reshape(9,-1).T @ (mu_R_bj * 2)

                aibj_3 /= R**4

            if order > 3:
                omega_bj = omega_vo[j]
                RRR = einsum('x,y,z->xyz', R_bar, R_bar, R_bar)

                aibj_4  = einsum('xa,xyzb,yz->ab', mu_ai, omega_bj, RR * 9)
                aibj_4 += einsum('xy,xyza,zb->ab', RR * 9, omega_ai, mu_bj)

                omega_R3_ai = RRR.ravel() @ omega_ai.reshape(27,-1)
                omega_R3_bj = RRR.ravel() @ omega_bj.reshape(27,-1)
                aibj_4 -= np.outer(tmp_ai, omega_R3_bj * 21)
                aibj_4 -= np.outer(omega_R3_ai, tmp_bj * 21)

                aibj_4 += np.outer(tmp1_ai, tmp1_bj * 35)
                
                tmp2_ai = einsum('xya,y->xa', theta_ai, R_bar)
                tmp2_bj = einsum('xyb,y->xb', theta_bj, R_bar)
                aibj_4 -= tmp2_ai.T @ (tmp2_bj * 20)

                aibj_4 += theta_ai.reshape(9,-1).T @ (theta_bj.reshape(9,-1) * 2)

                aibj_4 /= (3 * R**5)

            aibj = aibj_2
            if order > 2:
                aibj += aibj_3
            if order > 3:
                aibj += aibj_4

            e_bj = e_vo[j]
            aibj2 = aibj * aibj / (e_ai[:,None] + e_bj[None,:])
            e_mp2_pair[i,j] = -8 * np.sum(aibj2)

    e_mp2_pair += e_mp2_pair.T
    return e_mp2_pair


def kernel(mp, prj, mo_energy=None, mo_coeff=None, eris=None,
           with_t2=WITH_T2):
    if mo_energy is not None or mo_coeff is not None:
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:      eris = mp.ao2mo(mo_coeff)
    if mo_energy is None: mo_energy = eris.mo_energy
    if mo_coeff is None:  mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    naux = mp.with_df.get_naoaux()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    Lov = np.empty((naux, nocc*nvir))
    p1 = 0
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        p0, p1 = p1, p1 + qov.shape[0]
        Lov[p0:p1] = qov

    ovov = (Lov.T @ Lov).reshape(nocc,nvir,nocc,nvir)
    oovv = ovov.transpose(0,2,1,3)
    t2 = oovv / lib.direct_sum('jb+ia->ijba', eia, eia)

    ed_ij = einsum('pjab,qjab', t2, oovv) * 2
    ex_ij = -einsum('pjab,qjba', t2, oovv)

    if not with_t2:
        t2 = None

    m = prj.T.conj() @ prj
    ed = einsum('ij,ji', ed_ij, m).real
    ex = einsum('ij,ji', ex_ij, m).real

    emp2_ss = ed*0.5 + ex
    emp2_os = ed*0.5
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2, t2

class DFMP2(dfmp2.DFMP2):
    def kernel(self, prj, mo_energy=None, mo_coeff=None, eris=None,
               with_t2=WITH_T2):
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = 0.

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = kernel(self, prj, mo_energy, mo_coeff, eris, with_t2)
        else:
            raise NotImplementedError

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()
        return self.e_corr, self.t2
