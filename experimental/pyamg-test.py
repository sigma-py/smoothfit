import matplotlib.pyplot as plt
import meshzoo
import npx
import numpy as np
import pyamg
import scipy.linalg
import skfem as fem
from skfem.helpers import dot
from skfem.models.poisson import laplace

points, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), 35)
# points, cells = meshzoo.disk(6, 25)


@fem.BilinearForm
def flux(u, v, w):
    return dot(w.n, u.grad) * v


mesh = fem.MeshTri(points.T, cells.T)
basis = fem.InteriorBasis(mesh, fem.ElementTriP1())
facet_basis = fem.FacetBasis(basis.mesh, basis.elem)

lap = fem.asm(laplace, basis)
boundary_terms = fem.asm(flux, facet_basis)

A = lap - boundary_terms

# np.random.seed(0)
b = np.random.rand(A.shape[1])
# make the system consistent
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

print(type(M))

x = np.random.rand(M.shape[1])
y = np.random.rand(M.shape[1])

print(np.dot(x, M @ y))
print(np.dot(M @ x, y))

exit(1)

x, info = npx.gmres(A, b, tol=1e-12, M=M, maxiter=20)
# res = b - A @ info.xk

plt.semilogy(np.arange(len(info.resnorms)), info.resnorms)
plt.savefig("out.png", transparent=True, bbox_inches="tight")
# plt.show()
