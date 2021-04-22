"""The file shows that pyamg's preconditioner works well even for A, although A is not
symmetric: Number of GMRES iterations more or less indepent of the number of unknowns.
"""
import matplotlib.pyplot as plt
import meshzoo
import npx
import numpy as np
import pyamg
import scipy.linalg
import skfem as fem
from skfem.helpers import dot
from skfem.models.poisson import laplace

rng = np.random.default_rng(0)

for n in range(5, 100, 5):
    points, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), n)
    # points, cells = meshzoo.disk(6, 25)
    print(f"{n = }, {len(points) = }")

    @fem.BilinearForm
    def flux(u, v, w):
        return dot(w.n, u.grad) * v

    # copy to avoid warnings; better would be to get the transposed arrays from meshzoo
    # directly
    mesh = fem.MeshTri(points.T.copy(), cells.T.copy())
    basis = fem.InteriorBasis(mesh, fem.ElementTriP1())
    facet_basis = fem.FacetBasis(basis.mesh, basis.elem)

    lap = fem.asm(laplace, basis)
    boundary_terms = fem.asm(flux, facet_basis)

    A = lap - boundary_terms

    b = rng.random(A.shape[1])
    # make the system consistent by removing A's left nullspace components from the
    # right-hand side
    lns = scipy.linalg.null_space(A.T.toarray()).T
    for n in lns:
        b -= np.dot(b, n) / np.dot(n, n) * n

    ml = pyamg.smoothed_aggregation_solver(
        A,
        coarse_solver="jacobi",
        symmetry="nonsymmetric",
        max_coarse=100,
        # B=scipy.linalg.null_space(A.toarray())
    )
    M = ml.aspreconditioner(cycle="V")

    # M is generally not symmetric:
    # print(np.dot(x, M @ y))
    # print(np.dot(M @ x, y))

    x = rng.random(M.shape[1])
    y = rng.random(M.shape[1])

    x, info = npx.gmres(A, b, tol=1e-12, M=M, maxiter=20)
    # res = b - A @ info.xk

    plt.semilogy(np.arange(len(info.resnorms)), info.resnorms)
    plt.xlabel("step")
    plt.ylabel("residual")
    plt.show()
    # plt.savefig("out.png", transparent=True, bbox_inches="tight")
