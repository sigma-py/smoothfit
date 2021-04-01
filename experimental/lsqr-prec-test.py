import krypy
import meshzoo
import numpy as np
import matplotlib.pyplot as plt
import perfplot
import pyamg
import pykry
import scipy.optimize
from dolfin import (
    BoundingBoxTree,
    Cell,
    DirichletBC,
    EigenMatrix,
    FacetNormal,
    Function,
    FunctionSpace,
    IntervalMesh,
    Mesh,
    MeshEditor,
    Point,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    assemble,
    dot,
    ds,
    dx,
    grad,
)
from scipy import sparse
from scipy.sparse.linalg import spsolve

np.random.seed(123)


def _build_eval_matrix(V, points):
    """Build the sparse m-by-n matrix that maps a coefficient set for a function in V to
    the values of that function at m given points."""
    # See <https://www.allanswered.com/post/lkbkm/#zxqgk>
    mesh = V.mesh()

    bbt = BoundingBoxTree()
    bbt.build(mesh)
    dofmap = V.dofmap()
    el = V.element()
    sdim = el.space_dimension()

    rows = []
    cols = []
    data = []
    for i, x in enumerate(points):
        cell_id = bbt.compute_first_entity_collision(Point(*x))
        cell = Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        rows.append(np.full(sdim, i))
        cols.append(dofmap.cell_dofs(cell_id))

        v = el.evaluate_basis_all(x, coordinate_dofs, cell_id)
        data.append(v)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    m = len(points)
    n = V.dim()
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    return matrix


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


def setup(n):
    x0 = np.random.rand(n, 2) - 0.5
    # y0 = np.ones(n)
    # y0 = x0[:, 0]
    # y0 = x0[:, 0]**2
    # y0 = np.cos(np.pi*x0.T[0])
    # y0 = np.cos(np.pi*x0.T[0]) * np.cos(np.pi*x0.T[1])
    y0 = np.cos(np.pi * np.sqrt(x0.T[0] ** 2 + x0.T[1] ** 2))

    points, cells = meshzoo.rectangle_tri((-1.0, -1.0), (1.0, 1.0), n)

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
    mesh = V.mesh()
    n = FacetNormal(mesh)

    A = _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()
    # lmbda = 1.0
    # A *= lmbda

    E = _build_eval_matrix(V, x0)

    # mass matrix
    M = _assemble_eigen(u * v * dx).sparray()

    # Neumann preconditioner
    An = _assemble_eigen(dot(grad(u), grad(v)) * dx).sparray()

    # Dirichlet preconditioner
    Ad = _assemble_eigen(dot(grad(u), grad(v)) * dx)
    bc = DirichletBC(V, 0.0, "on_boundary")
    bc.apply(Ad)
    Ad = Ad.sparray()

    # Aq = _assemble_eigen(
    #     dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds - dot(n, grad(v)) * u * ds
    # ).sparray()

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=100)
    # mln = pyamg.smoothed_aggregation_solver(An, coarse_solver="jacobi", max_coarse=100)
    # mld = pyamg.smoothed_aggregation_solver(Ad, coarse_solver="jacobi", max_coarse=100)
    # mlq = pyamg.smoothed_aggregation_solver(Aq, coarse_solver="jacobi", max_coarse=100)

    ml = pyamg.smoothed_aggregation_solver(
        A, coarse_solver="jacobi", symmetry="nonsymmetric", max_coarse=100
    )
    mlT = pyamg.smoothed_aggregation_solver(
        A.T, coarse_solver="jacobi", symmetry="nonsymmetric", max_coarse=100
    )

    P = scipy.sparse.linalg.LinearOperator(
        A.shape,
        matvec=lambda x: ml.solve(x, tol=1.0e-15),
        rmatvec=lambda x: mlT.solve(x, tol=1.0e-15),
    )

    x = np.random.rand(A.shape[1])
    y = np.random.rand(A.shape[1])

    # problem: P.T isn't even really the transpose of P. This messes up lsqr.
    print()
    print("a")
    print(np.dot(x, P @ y))
    print(np.dot(P.T @ x, y))
    print()

    # not working well:
    # P = (ml.aspreconditioner(), mlT.aspreconditioner())
    return A, E, M, P, y0


def scipy_lsqr_without_m(data):
    A, E, _, _, y0 = data

    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=lambda x: np.concatenate([A @ x, E @ x]),
        rmatvec=lambda y: A.T @ y[: A.shape[0]] + E.T @ y[A.shape[0] :],
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    x = out[0]
    # num_iter = out[2]
    return x


def a_identity(A, E, y0):
    def matvec(x):
        return np.concatenate([x, E @ x])

    def rmatvec(y):
        return y[: A.shape[0]] + E.T @ y[A.shape[0] :]

    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=matvec,
        rmatvec=rmatvec,
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    return out[2]


def _rprec(A, E, P, y0):
    def matvec(x):
        Px = P @ x
        return np.concatenate([A @ Px, E @ Px])

    def rmatvec(y):
        return P.T @ ((A.T @ y[: A.shape[0]]) + E.T @ y[A.shape[0] :])

    m = A.shape[0] + E.shape[0]
    n = A.shape[1]

    x = np.random.rand(n)
    y = np.random.rand(m)

    print(np.dot(matvec(x), y))
    print(np.dot(rmatvec(y), x))

    lop = scipy.sparse.linalg.LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec)

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    return out[2]


def _main():
    n_range = range(3, 101)
    num_steps = []
    for n in n_range:
        A, E, _, P, y0 = setup(n)
        n_steps = _rprec(A, E, P, y0)
        # n_steps = a_identity(A, E, y0)
        num_steps.append(n_steps)
        print(n, n_steps)

    plt.plot(n_range, num_steps)
    plt.show()


if __name__ == "__main__":
    _main()
