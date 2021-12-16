"""The file shows that pyamg's preconditioner works well even for A, although A is not
symmetric: Number of GMRES iterations more or less indepent of the number of unknowns.
"""
import matplotlib.pyplot as plt
import meshzoo
import numpy as np
import pyamg
import scipy.linalg
import scipyx
import skfem as fem
from skfem.helpers import dot
from skfem.models.poisson import laplace

rng = np.random.default_rng(0)

tol = 1.0e-8

for n in range(5, 41, 5):
    # points, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), n)
    points, cells = meshzoo.disk(6, n)
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
    )
    M = ml.aspreconditioner(cycle="V")

    # M is generally not symmetric:
    # print(np.dot(x, M @ y))
    # print(np.dot(M @ x, y))

    _, info = scipyx.gmres(A, b, tol=tol, M=M, maxiter=20)
    # res = b - A @ info.xk

    num_unknowns = A.shape[1]
    plt.semilogy(
        np.arange(len(info.resnorms)), info.resnorms, label=f"N={num_unknowns}"
    )

plt.xlabel("step")
plt.ylabel("residual")
plt.legend()
plt.savefig("out.png", bbox_inches="tight")
plt.show()
