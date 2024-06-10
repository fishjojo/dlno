from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import scf, df, lo
from domain import get_bp_domain, get_primary_domain, _compute_av
import pao as mod_pao 
import util
import mp2
from lno.ad.ccsd import LNOCCSD

def kernel(mydlno, pair_energy_thr=None):
    if pair_energy_thr is None:
        pair_energy_thr = mydlno.pair_energy_thr

    mol = mydlno.mol
    s1e=mydlno.s1e
    fock = mydlno.fock
    nocc = mydlno.nocc
    lmo_bp_domain = mydlno.lmo_bp_domain
    lmo_primary_domain = mydlno.lmo_primary_domain

    domain_pao = mydlno.build_domain_pao()
    (eo,vo), (ev,vv) = mydlno.canonicalize(domain_pao)
    E = mp2.pair_energy_multipole(mol, eo, vo, ev, vv, lmo_primary_domain)
    E1 = E + np.eye(nocc)

    e_domains = []
    ep_domains = []
    for i in range(nocc):
        idx = np.where(abs(E1[i]) > pair_energy_thr)[0] #strong pairs
        e_domains.append(reduce(np.union1d, lmo_bp_domain[idx]))
        ep_domains.append(reduce(np.union1d, lmo_primary_domain[idx]))

    e_domains = util.list_to_array(e_domains)
    unique_ep_domains = util.unique(ep_domains)

    e_corr = np.sum(E[abs(E) < pair_energy_thr])*.5
    lmo = mydlno.lmo
    for atmlst, lmo_idx in unique_ep_domains.items():
        atmlst = list(atmlst)
        fake_mol = util.fake_mol_by_atom(mol, atmlst)
        auxmol = df.addons.make_auxmol(mol)
        fake_auxmol = util.fake_mol_by_atom(auxmol, atmlst)
        _df = df.DF(fake_mol)
        _df.auxmol = fake_auxmol
        _df.build()

        lmo_idx = np.asarray(lmo_idx)
        sub_e_domains = e_domains[lmo_idx]
        unique_sub_e_domains = util.unique(sub_e_domains)
        for k, v in unique_sub_e_domains.items():
            unique_sub_e_domains[k] = lmo_idx[v]

        ao_idx = util.ao_index_by_atom(mol, atmlst)
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]
        fock22 = fock[np.ix_(ao_idx, ao_idx)]

        fake_mf = scf.RHF(fake_mol)
        fake_mf.with_df = _df
        fake_mf.converged = True

        for atms1, idx1 in unique_sub_e_domains.items():
            pao = mod_pao.pao_overlap_with_domain(
                        mol, mydlno.pao, list(atms1), atmlst,
                        ao2pao_map=mydlno.ao2pao_map, s1e=s1e, ovlp_thr=mydlno.domain_pao_thr)

            pao_prj = util.project_mo(pao, s21, s22)

            for i in idx1:
                idx = np.where(abs(E[i]) > pair_energy_thr)[0]
                lmo_i = lmo[:,i].reshape(-1,1)
                lmo_j = lmo[:,idx]

                lmo_i_prj = util.project_mo(lmo_i, s21, s22)
                lmo_i_prj = lo.orth.vec_lowdin(lmo_i_prj, s=s22)

                lmo_j_prj = util.project_mo(lmo_j, s21, s22)
                lmo_j_prj = util.orthogonalize(lmo_i_prj, lmo_j_prj, s22)
                lmo_j_prj = lo.orth.vec_lowdin(lmo_j_prj, s=s22)            

                lmo_ij_prj = np.concatenate([lmo_i_prj, lmo_j_prj], axis=-1)
                eij, lmo_ij_canon = semicanonicalize(mol, lmo_ij_prj, fock, atmlst)

                pao_a = util.orthogonalize(lmo_ij_canon, pao_prj, s22)
                pao_a = lo.orth.vec_lowdin(pao_a, s=s22)
                ea, pao_a_canon = semicanonicalize(mol, pao_a, fock, atmlst)

                prjlo = reduce(np.dot, (lmo_i_prj.T.conj(), s22, lmo_ij_canon))
                #MP2
                _mo_coeff = np.concatenate([lmo_ij_canon, pao_a_canon], axis=-1)
                _mo_energy = np.concatenate([eij, ea],axis=None)
                _mo_occ = np.zeros((_mo_energy.size,), dtype=np.int32)
                _mo_occ[np.arange(eij.size)] = 2
                fake_mf.mo_coeff = _mo_coeff
                fake_mf.mo_energy = _mo_energy
                fake_mf.mo_occ = _mo_occ
                _mp2 = mp2.DFMP2(fake_mf)
                emp2 = _mp2.kernel(prjlo, with_t2=False)[0]

                mylno = LNOCCSD(fake_mf, thresh=1e-4, fock=fock22, s1e=s22)
                mylno.kernel(frag_lolist='1o', orbloc=lmo_i_prj)
                e_corr += mylno.e_corr - emp2
    return e_corr

def build_domain_pao(mydlno, domain_pao_thr=1e-6):
    mol = mydlno.mol
    lmo_bp_domain = mydlno.lmo_bp_domain
    lmo_p_domain = mydlno.lmo_primary_domain
    ao2pao_map = mydlno.ao2pao_map
    s1e = mydlno.s1e
    pao = mydlno.pao

    domain_pao = []
    for i in range(mydlno.nocc):
        pao_i = mod_pao.pao_overlap_with_domain(
                    mol, pao, lmo_bp_domain[i], lmo_p_domain[i],
                    ao2pao_map=ao2pao_map, s1e=s1e, ovlp_thr=domain_pao_thr)
        domain_pao.append(pao_i)

    out = np.empty(len(domain_pao), dtype=object)
    out[:] = domain_pao
    return out


def canonicalize(mydlno, domain_pao, fock=None, lmo=None,
                 project=False, orth_ov=False):
    if fock is None:
        fock = mydlno.fock
    if lmo is None:
        lmo = mydlno.lmo

    if project:
        mol = mydlno.mol
        lmo_p_domain = mydlno.lmo_primary_domain
        s1e = mydlno.s1e
        av_thr = mydlno.pao_bp_domain_thr

    e_occ = []
    v_occ = []
    e_vir = []
    v_vir = []
    for i in range(mydlno.nocc):
        lmo_i = lmo[:,i].reshape(-1,1)
        pao_i = domain_pao[i]
        fock_i = fock

        if project:
            ao_idx = util.ao_index_by_atom(mol, lmo_p_domain[i])
            fock_i = fock[np.ix_(ao_idx, ao_idx)]
            s21 = s1e[ao_idx]
            s22 = s1e[np.ix_(ao_idx, ao_idx)]

            lmo_i = util.project_mo(lmo_i, s21, s22)
            lmo_i = lo.orth.vec_lowdin(lmo_i, s=s22)

            av = _compute_av(mol, pao_i, s1e=s1e, atmlst=lmo_p_domain[i])
            pao_i = pao_i[:,av>av_thr]
            pao_i = util.project_mo(pao_i, s21, s22)
            pao_i = lo.orth.vec_lowdin(pao_i, s=s22)

            if orth_ov:
                pao_i = util.orthogonalize(lmo_i, pao_i, s22)
                pao_i = lo.orth.vec_lowdin(pao_i, s=s22)

        foo = reduce(np.dot, (lmo_i.T.conj(), fock_i, lmo_i))
        e_occ.append(foo[0,0])
        v_occ.append(lmo_i)

        fvv = reduce(np.dot, (pao_i.T.conj(), fock_i, pao_i))
        w, v = np.linalg.eigh(fvv)
        logger.debug1(mydlno, f"canonical domain PAO energy {i}:\n{w}")
        e_vir.append(w)
        v_vir.append(np.dot(pao_i, v))
    return (e_occ, v_occ), (e_vir, v_vir)


class DLNO(lib.StreamObject):
    lmo_method = 'boys'
    lmo_kwargs = None

    lmo_bp_domain_thr = 0.9999
    pao_bp_domain_thr = 0.98

    pao_norm_thr = 1e-4

    domain_pao_thr = 1e-4
    domain_project = True
    domain_orth_ov = True

    pair_energy_thr = 1e-4

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.mo_coeff = mf.mo_coeff
        self.nocc = self.mol.nelectron//2
        self.verbose = self.mol.verbose

        # private
        self._s1e = None
        self._fock = None
        self._lmo = None
        self._lmo_bp_domain = None
        self._lmo_primary_domain = None
        self._pao = None
        self._ao2pao_map = None
        self._pao_bp_domain = None

    @property
    def s1e(self):
        if self._s1e is None:
            self._s1e = self.mol.intor_symmetric('int1e_ovlp')
        return self._s1e
    @s1e.setter
    def s1e(self, value):
        self._s1e = value

    @property
    def fock(self):
        if self._fock is None:
            self._fock = self._scf.get_fock()
        return self._fock
    @fock.setter
    def fock(self, value):
        self._fock = value

    @property
    def lmo(self):
        if self._lmo is None:
            self._lmo = self.build_lmo()
        return self._lmo
    @lmo.setter
    def lmo(self, value):
        self._lmo = value

    @property
    def lmo_bp_domain(self):
        if self._lmo_bp_domain is None:
            self._lmo_bp_domain = self.build_lmo_bp_domain()
        return self._lmo_bp_domain
    @lmo_bp_domain.setter
    def lmo_bp_domain(self, value):
        self._lmo_bp_domain = value

    @property
    def lmo_primary_domain(self):
        if self._lmo_primary_domain is None:
            self._lmo_primary_domain = self.build_lmo_primary_domain()
        return self._lmo_primary_domain
    @lmo_primary_domain.setter
    def lmo_primary_domain(self, value):
        self._lmo_primary_domain = value

    @property
    def pao(self):
        if self._pao is None:
            # update map as well
            self._pao, self._ao2pao_map = self.build_pao()
        return self._pao
    @pao.setter
    def pao(self, value):
        self._pao = value

    @property
    def ao2pao_map(self):
        if self._ao2pao_map is None:
            # update pao as well
            self._pao, self._ao2pao_map = self.build_pao()
        return self._ao2pao_map
    @ao2pao_map.setter
    def ao2pao_map(self, value):
        self._ao2pao_map = value

    @property
    def pao_bp_domain(self):
        if self._pao_bp_domain is None:
            self._pao_bp_domain = self.build_pao_bp_domain()
        return self._pao_bp_domain
    @pao_bp_domain.setter
    def pao_bp_domain(self, value):
        self._pao_bp_domain = value

    def build_lmo(self, lmo_method=None, lmo_kwargs=None):
        if lmo_method is None:
            lmo_method = self.lmo_method
        if lmo_kwargs is None:
            lmo_kwargs = self.lmo_kwargs
        if lmo_kwargs is None:
            lmo_kwargs = {}

        mol = self.mol
        orbo = self.mo_coeff[:,:self.nocc]
        if lmo_method.lower() == 'boys':
            lmo = lo.Boys(mol, mo_coeff=orbo).kernel(**lmo_kwargs)
        else:
            raise NotImplementedError
        return lmo

    def build_lmo_bp_domain(self, lmo_bp_domain_thr=None):
        if lmo_bp_domain_thr is None:
            lmo_bp_domain_thr = self.lmo_bp_domain_thr
        return get_bp_domain(self.mol, self.lmo, self.s1e, lmo_bp_domain_thr)

    def build_pao(self, pao_norm_thr=None):
        if pao_norm_thr is None:
            pao_norm_thr = self.pao_norm_thr
        return mod_pao.pao(self.mol, self.lmo, self.s1e, pao_norm_thr)

    def build_pao_bp_domain(self, pao_bp_domain_thr=None):
        if pao_bp_domain_thr is None:
            pao_bp_domain_thr = self.pao_bp_domain_thr
        return get_bp_domain(self.mol, self.pao, self.s1e, pao_bp_domain_thr)

    def build_lmo_primary_domain(self):
        return get_primary_domain(self.mol, self.lmo_bp_domain,
                                  self.pao_bp_domain, self.ao2pao_map)

    def build_domain_pao(self, domain_pao_thr=None):
        if domain_pao_thr is None:
            domain_pao_thr = self.domain_pao_thr
        return build_domain_pao(self, domain_pao_thr)

    def canonicalize(self, domain_pao, fock=None, lmo=None,
                     project=None, orth_ov=None):
        if fock is None:
            fock = self.fock
        if lmo is None:
            lmo = self.lmo
        if project is None:
            project = self.domain_project
        if orth_ov is None:
            orth_ov = self.domain_orth_ov
        return canonicalize(self, domain_pao, fock=fock, lmo=lmo,
                            project=project, orth_ov=orth_ov)

    def reset(self):
        self._s1e = None
        self._fock = None
        self._lmo = None
        self._lmo_bp_domain = None
        self._lmo_primary_domain = None
        self._pao = None
        self._ao2pao_map = None
        self._pao_bp_domain = None

    kernel = kernel

def semicanonicalize(mol, mos, fock, atmlst):
    ao_idx = util.ao_index_by_atom(mol, atmlst)
    fock = fock[np.ix_(ao_idx, ao_idx)]

    f = reduce(np.dot, (mos.T.conj(), fock, mos))
    w, v = np.linalg.eigh(f)
    logger.debug1(mol, f"semicanonical orbital energy :\n{w}")
    return w, np.dot(mos, v)


if __name__ == "__main__":
    from pyscf import gto, scf, lo
    from pyscf.mp import dfmp2
    
    mol = gto.Mole()
    mol.atom = '''
        O         -1.48516       -0.11472        0.00000
        H         -1.86842        0.76230        0.00000
        H         -0.53383        0.04051        0.00000
        O          1.41647        0.11126        0.00000
        H          1.74624       -0.37395       -0.75856
        H          1.74624       -0.37395        0.75856
        H        -17.01061        0.77828        0.00081
        O        -17.45593        0.85616       -0.83572
        H        -18.39143        0.81791       -0.66982
    '''
    mol.atom = 'h2o_10.xyz'
    mol.basis = 'ccpvdz'
    mol.verbose = 4
    mol.max_memory = 12000
    mol.build()

    mf = scf.RHF(mol).density_fit()
    e_hf = mf.kernel()

    mymp = dfmp2.DFMP2(mf)
    mymp.kernel()

    mylno = DLNO(mf)
    mylno.lmo_bp_domain_thr = 0.9999
    mylno.pao_bp_domain_thr = 0.98
    mylno.domain_pao_thr = 1e-4
    mylno.pair_energy_thr = 1e-4

    emp2 = mylno.kernel()

    print(emp2, mymp.e_corr, emp2-mymp.e_corr)
