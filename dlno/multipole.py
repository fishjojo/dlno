import numpy as np
from dlno.util import fake_mol_by_atom

__all__ = ['dipole_op', 'quadrupole_op', 'octupole_op']

def dipole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        r = fake_mol.intor('int1e_r').reshape(3,nao,nao)
    return r

def quadrupole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        rr = fake_mol.intor('int1e_rr').reshape(3,3,nao,nao)
    r2 = np.trace(rr)

    rr *= 3
    for x in range(3):
        rr[x,x] -= r2
    rr *= .5
    return rr

def octupole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_common_origin(R):
        rrr = fake_mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)

    r2r_0 = np.trace(rrr, axis1=1, axis2=2)
    r2r_1 = np.trace(rrr, axis1=2, axis2=0)
    r2r_2 = np.trace(rrr, axis1=0, axis2=1)

    rrr *= 5
    for x in range(3):
        rrr[:,x,x] -= r2r_0
        rrr[x,:,x] -= r2r_1
        rrr[x,x,:] -= r2r_2
    rrr *= .5
    return rrr

