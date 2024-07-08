from functools import reduce
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import scf, df, lo
from pyscf.mp.mp2 import MP2, _mo_splitter

from dlno import pao as mod_pao 
from dlno import util
from dlno import mp2
from dlno.domain import (
    get_bp_domain,
    get_primary_domain,
    _compute_av,
)


def kernel(mydlno, auxbasis=None,
           lno_solver=None, lno_solver_kwargs=None, lno_solver_kernel_kwargs=None,
           lno_solver_mp2_correct=True):
    """Driver function for DLNO calculations.

    Parameters
    ----------
    auxbasis : str, optional
        Auxiliary basis set for density fitting.
    lno_solver : class, optional
        The LNO solver class. Its ``__init__`` function
        should take ``mf``, ``fock``, and ``s1e`` as the arguments
        representing the mean-field object, the Fock matrix,
        and the overlap matrix, respectively.
        Its ``kernel`` function should take ``orbloc`` as the argument
        for the localized occupied orbitals.
    lno_solver_kwargs : dict, optional
        Additional keyword arguments passed to ``lno_solver`` for instantiation.
    lno_solver_kernel_kwargs : dict, optional
        Additional keyword arguments passed to ``lno_solver``'s kernel function.
    lno_solver_mp2_correct : bool, default=True
        Whether to perform a composite MP2 correction within the LNO calculation.
        If set to ``True``, the ``lno_solver`` class should provide the attribute
        ``e_corr_pt2`` as the total fragment MP2 correlation energy being subtracted from 
        ``e_corr``, which is the total fragment high-level correlation energy.
    """
    if lno_solver is not None:
        if lno_solver_kwargs is None:
            lno_solver_kwargs ={}
        if lno_solver_kernel_kwargs is None:
            lno_solver_kernel_kwargs = {}

    mol = mydlno.mol
    s1e = mydlno.s1e
    fock = mydlno.fock
    nocc = mydlno.nocc
    pair_energy_thr = mydlno.pair_energy_thr
    av_thr = mydlno.pao_bp_domain_thr

    lmo_bp_domain = mydlno.lmo_bp_domain
    lmo_primary_domain = mydlno.lmo_primary_domain

    domain_pao = mydlno.build_domain_pao()
    (eo, vo), (ev, vv) = mydlno.canonicalize(domain_pao)
    E = mp2.pair_energy_multipole(mol, eo, vo, ev, vv,
                                  lmo_primary_domain,
                                  mydlno.multipole_order)
    E1 = E + np.eye(nocc)

    e_domains = []
    ep_domains = []
    for i in range(nocc):
        idx = np.where(abs(E1[i]) > pair_energy_thr)[0] #strong pairs
        e_domains.append(reduce(np.union1d, lmo_bp_domain[idx]))
        ep_domains.append(reduce(np.union1d, lmo_primary_domain[idx]))

    e_domains = util.list_to_array(e_domains)
    unique_ep_domains = util.unique(ep_domains)

    e_corr = np.sum(E[abs(E) < pair_energy_thr]) * .5 #distant pairs
    lmo = mydlno.lmo
    for atmlst, lmo_idx in unique_ep_domains.items():
        atmlst = list(atmlst)
        lmo_idx = np.asarray(lmo_idx)
        logger.debug(mydlno, f"extended primary domain:\n{atmlst}")

        fake_mol = util.fake_mol_by_atom(mol, atmlst)
        _df = df.DF(fake_mol, auxbasis=auxbasis)
        _df.build()

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
            logger.debug(mydlno, f"extended domain:\n{atms1}")
            pao = mod_pao.pao_overlap_with_domain(
                        mol, mydlno.pao, list(atms1),
                        ao2pao_map=mydlno.ao2pao_map, s1e=s1e,
                        ovlp_thr=mydlno.domain_pao_thr)

            av = _compute_av(mol, pao, s1e=s1e, atmlst=atmlst)
            pao = pao[:,av>av_thr]
            pao_prj = util.project_mo(pao, s21, s22)

            ij = []
            for i in idx1:
                ij.append(np.where(abs(E1[i]) > pair_energy_thr)[0])
            unique_strong_pairs = util.unique(ij)
            for k, v in unique_strong_pairs.items():
                unique_strong_pairs[k] = idx1[v]

            for ij_idx, i_idx in unique_strong_pairs.items():
                orbloc = []
                for i in i_idx:
                    lmo_i = lmo[:,i].reshape(-1,1)
                    lmo_i_prj = util.project_mo(lmo_i, s21, s22)
                    lmo_i_prj = lo.orth.vec_lowdin(lmo_i_prj, s=s22)
                    orbloc.append(lmo_i_prj)
                orbloc = np.hstack(orbloc)

                lmo_ij = lmo[:,ij_idx]
                lmo_ij_prj = util.project_mo(lmo_ij, s21, s22)
                lmo_ij_prj = lo.orth.vec_lowdin(lmo_ij_prj, s=s22)
                eij, lmo_ij_canon = semicanonicalize(mol, lmo_ij_prj, fock, atmlst)

                pao_a = util.orthogonalize(lmo_ij_canon, pao_prj, s22)
                pao_a = lo.orth.vec_lowdin(pao_a, s=s22)
                ea, pao_a_canon = semicanonicalize(mol, pao_a, fock, atmlst)

                # MP2
                _mo_coeff = np.concatenate([lmo_ij_canon, pao_a_canon], axis=-1)
                _mo_energy = np.concatenate([eij, ea], axis=None)
                _mo_occ = np.zeros((_mo_energy.size,), dtype=np.int32)
                _mo_occ[np.arange(eij.size)] = 2 #restricted closed-shell
                fake_mf.mo_coeff = _mo_coeff
                fake_mf.mo_energy = _mo_energy
                fake_mf.mo_occ = _mo_occ
                _mp2 = mp2.DFMP2(fake_mf)
                prjlo = orbloc.T.conj() @ s22 @ lmo_ij_canon
                emp2 = _mp2.kernel(prjlo, with_t2=False)[0]
                _mp2 = None

                if lno_solver is not None:
                    _lno = lno_solver(fake_mf, fock=fock22, s1e=s22, **lno_solver_kwargs)
                    _lno.kernel(orbloc=orbloc, **lno_solver_kernel_kwargs)
                    e_corr += _lno.e_corr
                    if lno_solver_mp2_correct:
                        e_corr += emp2 - _lno.e_corr_pt2
                    _lno = None
                else:
                    e_corr += emp2

        _df = None
        fake_mol = None
        fake_auxmol = None
        fake_mf = None
        s21 = s22 = None
        fock22 = None
    return e_corr

def build_domain_pao(mydlno, domain_pao_thr=1e-4):
    mol = mydlno.mol
    lmo_bp_domain = mydlno.lmo_bp_domain
    lmo_p_domain = mydlno.lmo_primary_domain
    s1e = mydlno.s1e
    pao = mydlno.pao
    ao2pao_map = mydlno.ao2pao_map

    domain_pao = []
    for i in range(mydlno.nocc):
        pao_i = mod_pao.pao_overlap_with_domain(
                    mol, pao, lmo_bp_domain[i],
                    ao2pao_map=ao2pao_map, s1e=s1e, ovlp_thr=domain_pao_thr)
        domain_pao.append(pao_i)

    return util.list_to_array(domain_pao)


def canonicalize(mydlno, domain_pao, fock=None, lmo=None,
                 project=True, orth_ov=True):
    if fock is None:
        fock = mydlno.fock
    if lmo is None:
        lmo = mydlno.lmo

    if project:
        mol = mydlno.mol
        s1e = mydlno.s1e
        lmo_p_domain = mydlno.lmo_primary_domain
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
            lmo_i = lo.vec_lowdin(lmo_i, s=s22)

            av = _compute_av(mol, pao_i, s1e=s1e, atmlst=lmo_p_domain[i])
            pao_i = pao_i[:,av>av_thr]
            pao_i = util.project_mo(pao_i, s21, s22)
            pao_i = lo.vec_lowdin(pao_i, s=s22)

            if orth_ov:
                pao_i = util.orthogonalize(lmo_i, pao_i, s22)
                pao_i = lo.vec_lowdin(pao_i, s=s22)

        foo = lmo_i.conj().T @ fock_i @ lmo_i
        e_occ.append(foo[0,0])
        v_occ.append(lmo_i)

        fvv = pao_i.conj().T @ fock_i @ pao_i
        w, v = np.linalg.eigh(fvv)
        logger.debug1(mydlno, f"canonical domain PAO energy {i}:\n{w}")
        e_vir.append(w)
        v_vir.append(pao_i @ v)
    return (e_occ, v_occ), (e_vir, v_vir)


class DLNO(MP2):
    """DLNO base class.

    Attributes
    ----------
    lmo_method : str, default="pm"
        Localization method for occupied MOs.
        The occupied local orbitals can also be supplied
        through the ``lmo`` attribute.
    lmo_kwargs : dict
        Options for MO localizer.
    lmo_bp_domain_thr : float, default=0.999
        Threshold for defining the LMO BP domain.
    pao_bp_domain_thr : float, default=0.98
        Threshold for defining the PAO BP domain.
    pao_norm_thr : float, default=1e-4
        PAOs with norm smaller than ``pao_norm_thr`` are discarded.
    domain_pao_thr : float, default=1e-4
        Threshold for selecting PAOs in the larger domain that
        overlap with the smaller domain.
    pair_energy_thr : float, default=1e-4
        Energy cutoff for distinguishing strong and distant pairs
        using OS-MP2 pair correlation energy.
    multipole_order : int, default=4
        Multipole expansion order of OS-MP2 pair correlation energy.
    """
    lmo_method = "pm"
    lmo_kwargs = None

    lmo_bp_domain_thr = 0.999
    pao_bp_domain_thr = 0.98

    pao_norm_thr = 1e-4

    domain_pao_thr = 1e-4
    domain_project = True
    domain_orth_ov = True

    pair_energy_thr = 1e-4
    multipole_order = 4

    def __init__(self, mf, *,
                 frozen=None, mo_coeff=None, mo_occ=None):
        MP2.__init__(self, mf,
                     frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)

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
    def s1e(self, s):
        self._s1e = s

    @property
    def fock(self):
        if self._fock is None:
            self._fock = self._scf.get_fock()
        return self._fock
    @fock.setter
    def fock(self, f):
        self._fock = f

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

    def build_lmo(self, orbocc=None, lmo_method=None, lmo_kwargs=None):
        if lmo_method is None:
            lmo_method = self.lmo_method
        if lmo_kwargs is None:
            lmo_kwargs = self.lmo_kwargs
        if lmo_kwargs is None:
            lmo_kwargs = {}
        if orbocc is None:
            orbocc = self.mo_coeff[:, _mo_splitter(self)[1]]

        mol = self.mol
        if lmo_method.lower() == 'boys':
            lmo = lo.Boys(mol, mo_coeff=orbocc).kernel(**lmo_kwargs)
        elif lmo_method.lower() == 'pm':
            lmo = lo.PM(mol, mo_coeff=orbocc).kernel(**lmo_kwargs)
        elif lmo_method.lower() == 'er':
            lmo = lo.ER(mol, mo_coeff=orbocc).kernel(**lmo_kwargs)
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

        mos = np.hstack((self.mo_coeff[:,_mo_splitter(self)[0]],
                         self.mo_coeff[:,_mo_splitter(self)[1]],
                         self.mo_coeff[:,_mo_splitter(self)[3]]))
        return mod_pao.pao(self.mol, mos, self.s1e, pao_norm_thr)

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

    f = mos.T.conj() @ fock @ mos
    w, v = np.linalg.eigh(f)
    logger.debug1(mol, f"semicanonical orbital energy :\n{w}")
    return w, mos @ v

