from functools import partial
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import dfmp2
from multipole import *

WITH_T2 = True

einsum = partial(np.einsum, optimize=True)

def pair_energy_multipole(mol,
                          e_occ, mo_occ,
                          e_vir, mo_vir,
                          atmlst=None):
    nocc = len(e_occ)
    if atmlst is None:
        atmlst = [None,] * nocc
    E = np.zeros((nocc,nocc))

    Rs = []
    mu_vo = []
    theta_vo = []
    omega_vo = []
    e_vo = []
    for i in range(nocc):
        lmo_i = mo_occ[i]
        atmlst_i = atmlst[i]
        Di = dipole_op(mol, atmlst=atmlst_i)
        Ri = einsum('ui,xuv,vi->x', lmo_i.conj(), Di, lmo_i)
        #Qi = quadrupole_op(mol, R=Ri, atmlst=atmlst_i)
        #Oi = octupole_op(mol, R=Ri, atmlst=atmlst_i)
        Rs.append(Ri)

        pao_i = mo_vir[i]
        mu_ai = einsum('ua,xuv,vi->xai', pao_i.conj(), Di, lmo_i)
        #theta_ai = einsum('ua,xyuv,vi->xyai', pao_i.conj(), Qi, lmo_i)
        #omega_ai = einsum('ua,xyzuv,vi->xyzai', pao_i.conj(), Oi, lmo_i)
        mu_vo.append(mu_ai)
        #theta_vo.append(theta_ai)
        #omega_vo.append(omega_ai)

        e_i = np.asarray(e_occ[i]).ravel()
        e_a = e_vir[i]
        e_ai = e_a[:,None] - e_i[None,:]
        e_vo.append(e_ai)

    for i in range(nocc):
        Ri = Rs[i]
        mu_ai = mu_vo[i]
        #theta_ai = theta_vo[i]
        #omega_ai = omega_vo[i]
        e_ai = e_vo[i]

        for j in range(i):
            Rj = Rs[j]
            R = np.linalg.norm(Ri - Rj)
            R_bar = (Ri-Rj)/R

            mu_bj = mu_vo[j]
            aibj_2 = einsum('xai,xbj->aibj', mu_ai, mu_bj)
            tmp_ai = einsum('x,xai->ai', R_bar, mu_ai)
            tmp_bj = einsum('x,xai->ai', R_bar, mu_bj)
            aibj_2 -= einsum('ai,bj->aibj', tmp_ai, tmp_bj) * 3
            aibj_2 /= R**3

            #theta_bj = theta_vo[j]

            #tmp1_bj = einsum('x,xybj,y->bj', R_bar, theta_bj, R_bar)
            #aibj_3  = einsum('ai,bj->aibj', tmp_ai, tmp1_bj) * 15
            #aibj_3 -= einsum('xai,xybj,y->aibj', mu_ai, theta_bj, R_bar) * 3
            #aibj_3 -= einsum('yai,xybj,x->aibj', mu_ai, theta_bj, R_bar) * 3
            #aibj_3 -= einsum('xai,yybj,x->aibj', mu_ai, theta_bj, R_bar) * 3

            #tmp1_ai = einsum('x,xyai,y->ai', R_bar, theta_ai, R_bar)
            #aibj_3 += einsum('bj,ai->aibj', tmp_bj, tmp1_ai) * 15
            #aibj_3 -= einsum('xbj,xyai,y->aibj', mu_bj, theta_ai, R_bar) * 3
            #aibj_3 -= einsum('ybj,xyai,x->aibj', mu_bj, theta_ai, R_bar) * 3
            #aibj_3 -= einsum('xbj,yyai,x->aibj', mu_bj, theta_ai, R_bar) * 3

            #aibj_3 /= R**4

            #R_bar_2 = np.outer(R_bar, R_bar)
            #R_bar_3 = np.einsum('x,y,z->xyz', R_bar, R_bar, R_bar)

            #omega_bj = omega_vo[j]

            #aibj_4  = einsum('xai,xyzbj,yz->aibj', mu_ai, omega_bj, R_bar_2) * 9
            #aibj_4 -= einsum('ai,xyzbj,xyz->aibj', tmp_ai, omega_bj, R_bar_3) * 21

            #aibj_4 += einsum('xbj,xyzai,yz->aibj', mu_bj, omega_ai, R_bar_2) * 9
            #aibj_4 -= einsum('bj,xyzai,xyz->aibj', tmp_bj, omega_ai, R_bar_3) * 21

            #aibj_4 += einsum('ai,bj->aibj', tmp1_ai, tmp1_bj) * 35
            #
            #tmp2_ai = einsum('xyai,y->xai', theta_ai, R_bar)
            #tmp2_bj = einsum('xybj,y->xbj', theta_bj, R_bar)
            #aibj_4 -= einsum('xai,xbj->aibj', tmp2_ai, tmp2_bj) * 20

            #aibj_4 += einsum('xyai,yxbj->aibj', theta_ai, theta_bj) * 2

            #aibj_4 /= (R**5 * 3)

            aibj = aibj_2 #+ aibj_3 #+ aibj_4            

            e_bj = e_vo[j]
            aibj2 = aibj * aibj / (e_ai[:,:,None,None] + e_bj[None,None,:,:])
            E[i,j] = -8 * einsum('aibj->ij', aibj2)[0,0]

    E += E.T
    return E


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

    ovov = np.dot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    oovv = ovov.transpose(0,2,1,3)
    t2 = oovv / lib.direct_sum('jb+ia->ijba', eia, eia)

    ed_ij = np.einsum('pjab,qjab', t2, oovv) * 2
    ex_ij = -np.einsum('pjab,qjba', t2, oovv)

    if not with_t2:
        t2 = None

    m = np.dot(prj.T.conj(), prj)
    ed = np.einsum('ij,ji', ed_ij, m).real
    ex = np.einsum('ij,ji', ex_ij, m).real

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
