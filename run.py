import pyscf

mol = pyscf.gto.Mole()
mol.build(
    atom = '''
        O         0.4183272099    0.1671038379    0.1010361156
        H         0.8784893276   -0.0368266484    0.9330933285
        H        -0.3195928737    0.7774121014    0.3045311682
        ''',
    basis = '6-31g*',
    verbose=0,
    charge=0
    )
mf = pyscf.scf.RKS(mol)
mf.verbose = 4
mf.max_cycle = 200
mf.conv_tol  = 1e-8
mf.conv_tol_grad = 1e-8
mf.xc = 'b3lyp'
mf.kernel()

from rks import CasidaTDDFT as tddft
mf.verbose = 10
tdobj = tddft(mf)
result = tdobj.kernel(nstates=3)
tddft.analyze(tdobj, verbose=10)