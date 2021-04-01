import meshzoo
import numpy as np
import scipy.linalg
import skfem as fem
from skfem.models.poisson import laplace

# points, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), 20)
points, cells = meshzoo.disk(6, 10)


@fem.BilinearForm
def flux(u, v, w):
    from skfem.helpers import dot

    return dot(w.n, u.grad) * v


mesh = fem.MeshTri(points.T, cells.T)
basis = fem.InteriorBasis(mesh, fem.ElementTriP1())
facet_basis = fem.FacetBasis(basis.mesh, basis.elem)

lap = fem.asm(laplace, basis)
boundary_terms = fem.asm(flux, facet_basis)

A = lap - boundary_terms

# the right null space are the affine-linear functions
# rns = scipy.linalg.null_space(A.toarray()).T
# mesh.save("out.vtk", point_data={"k0": rns[0], "k1": rns[1], "k2": rns[2]})

# the left null space is a bit weird; something around the boundaries
# lns = scipy.linalg.null_space(A.T.toarray()).T
# mesh.save("out.vtk", point_data={"k0": lns[0], "k1": lns[1], "k2": lns[2]})
