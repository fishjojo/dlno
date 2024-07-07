from pyscf import gto, scf
from pyscf.mp import dfmp2
from dlno import dlno

def test_lmp2_energy():
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
    mol.basis = 'ccpvdz'
    mol.verbose = 0
    mol.max_memory = 8000
    mol.build()

    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    mycc = dfmp2.DFMP2(mf)
    mycc.kernel()

    mylno = dlno.DLNO(mf)
    mylno.lmo_method='pm'
    mylno.lmo_bp_domain_thr = 0.999
    mylno.pao_bp_domain_thr = 0.98
    mylno.domain_pao_thr = 1e-4
    mylno.pair_energy_thr = 1e-4
    mylno.multipole_order = 4

    e_corr = mylno.kernel()
    assert abs(e_corr-mycc.e_corr) < 2e-5
