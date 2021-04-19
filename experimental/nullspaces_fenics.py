import meshzoo
import numpy as np
from dolfin import (
    EigenMatrix,
    Expression,
    FacetNormal,
    FunctionSpace,
    Mesh,
    MeshEditor,
    TestFunction,
    TrialFunction,
    assemble,
    dot,
    ds,
    dx,
    grad,
    project,
)

np.random.seed(0)


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


def create_A_fenics(points, cells):
    # Convert points, cells to dolfin mesh
    editor = MeshEditor()
    mesh = Mesh()
    # topological and geometrical dimension 2
    editor.open(mesh, "triangle", 2, 2, 1)
    editor.init_vertices(len(points))
    editor.init_cells(len(cells))
    for k, point in enumerate(points):
        editor.add_vertex(k, point[:2])
    for k, cell in enumerate(cells.astype(np.uintp)):
        editor.add_cell(k, cell)
    editor.close()

    degree = 1
    V = FunctionSpace(mesh, "CG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(mesh)

    # A = _assemble_eigen(dot(grad(u), grad(v)) * dx).sparray()
    # A = _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()
    A = _assemble_eigen(
        dot(grad(u), grad(v)) * dx
        - dot(n, grad(u)) * v * ds
        # - dot(n, grad(v)) * u * ds
    ).sparray()

    return A, V


def create_A_scikit(points, cells):
    import skfem as fem
    from skfem.models.poisson import laplace

    @fem.BilinearForm
    def flux(u, v, w):
        from skfem.helpers import dot

        return dot(w.n, u.grad) * v

    mesh = fem.MeshTri(points.T, cells.T)
    basis = fem.InteriorBasis(mesh, fem.ElementTriP1())
    facet_basis = fem.FacetBasis(basis.mesh, basis.elem)

    print(basis)
    exit(1)

    lap = fem.asm(laplace, basis)
    boundary_terms = fem.asm(flux, facet_basis)

    return lap - boundary_terms


k = 20

# mesh = UnitIntervalMesh(n)
# mesh = UnitSquareMesh(k, k)

# points, cells = meshzoo.rectangle_tri((0.0, 0.0), (1.0, 1.0), k)
points, cells = meshzoo.disk(6, 10)

A1, V = create_A_fenics(points, cells)
# A2 = create_A_scikit(points, cells)


# the right null space are the affine-linear functions
f0 = Expression("1.0", element=V.ufl_element())
e0 = project(f0, V).vector().get_local()
f1 = Expression("x[0]", element=V.ufl_element())
e1 = project(f1, V).vector().get_local()
f2 = Expression("x[1]", element=V.ufl_element())
e2 = project(f2, V).vector().get_local()

nullspace = [e0, e1, e2]
for n in nullspace:
    An = A1 @ n
    print(np.dot(An, An))

# keep an eye on for how to store functions
# <https://fenicsproject.discourse.group/t/store-function-with-meshio/5473/2>
# import meshio
#
# meshio.Mesh(
#     V.mesh().coordinates(),
#     {"triangle": V.mesh().cells()},
#     point_data={"k0": nullspace[0], "k1": nullspace[1], "k2": nullspace[2]},
#     # point_data={"vx": points[:, 0], "vy": points[:, 1]}
# ).write("out.vtk")
