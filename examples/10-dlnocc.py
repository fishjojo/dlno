"""
Domain truncated LNO-CCSD(T)
"""
from pyscf import gto, scf
from pyscf.mp import dfmp2
from pyscfad.cc import dfccsd
from dlno import dlno
from pyscfad.lno import LNOCCSD_T

mol = gto.Mole()
mol.atom = """
    O         -1.48516       -0.11472        0.00000
    H         -1.86842        0.76230        0.00000
    H         -0.53383        0.04051        0.00000
    O          1.41647        0.11126        0.00000
    H          1.74624       -0.37395       -0.75856
    H          1.74624       -0.37395        0.75856
    H        -17.01061        0.77828        0.00081
    O        -17.45593        0.85616       -0.83572
    H        -18.39143        0.81791       -0.66982
"""
mol.basis = "ccpvdz"
mol.verbose = 4
mol.max_memory = 16000
mol.build()

mf = scf.RHF(mol).density_fit()
mf.kernel()

# Reference DF-CCSD(T)
mycc = dfccsd.RCCSD(mf, frozen=None)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)
et = mycc.ccsd_t(eris=eris)
e_corr_ref = mycc.e_corr+et

# DLNO-CCSD(T)
mylno = dlno.DLNO(mf)
mylno.lmo_method="pm"
mylno.lmo_bp_domain_thr = 0.999
mylno.pao_bp_domain_thr = 0.98
mylno.domain_pao_thr = 1e-4
mylno.pair_energy_thr = 1e-4
mylno.multipole_order = 4
e_corr = mylno.kernel(
    lno_solver=LNOCCSD_T,
    lno_solver_kwargs={"thresh": 1e-5},
    lno_solver_kernel_kwargs={"frag_lolist": "1o"},
    lno_solver_mp2_correct=True,
)

print(f"DLNO-CCSD(T) correlation energy: {e_corr}\n"
      f"Canonical DF-CCSD(T) correlation energy: {e_corr_ref}\n"
      f"Error: {e_corr-e_corr_ref}\n"
      f"Percentage of correlation energy recovered: {e_corr/e_corr_ref*100}%")
