from functools import reduce
import numpy as np
import util

def dipole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = util.fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_rinv_origin(R):
        r = fake_mol.intor('int1e_r').reshape(3,nao,nao)
    return r

def quadrupole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = util.fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_rinv_origin(R):
        rr = fake_mol.intor('int1e_rr').reshape(3,3,nao,nao)
        #r2 = fake_mol.intor('int1e_r2')
    r2 = np.trace(rr)

    out = 3 * rr
    for x in range(3):
        out[x,x] -= r2
    out *= .5
    return out

def octupole_op(mol, R=np.zeros((3,)), atmlst=None):
    fake_mol = util.fake_mol_by_atom(mol, atmlst)
    nao = fake_mol.nao
    with fake_mol.with_rinv_origin(R):
        rrr = fake_mol.intor('int1e_rrr').reshape(3,3,3,nao,nao)
        r2r = fake_mol.intor('int1e_r2r').reshape(3,nao,nao)

    out = 5 * rrr
    for x in range(3):
        out[:,x,x] -= r2r
        out[x,:,x] -= r2r
        out[x,x,:] -= r2r
    out *= .5
    return out

