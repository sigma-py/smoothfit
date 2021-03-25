import meshzoo
import numpy as np
import perfplot
import pykry
import krypy
import scipy.optimize
from dolfin import (
    BoundingBoxTree,
    Cell,
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

    lmbda = 1.0

    degree = 1
    V = FunctionSpace(mesh, "CG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)
    mesh = V.mesh()
    n = FacetNormal(mesh)

    A = _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()
    A *= lmbda

    E = _build_eval_matrix(V, x0)

    # mass matrix
    M = _assemble_eigen(u * v * dx).sparray()

    return A, E, M, y0


def dense_direct(data):
    A, E, M, y0 = data
    # Minv is dense, yikes!
    a = A.toarray()
    m = M.toarray()
    e = E.toarray()
    AT_Minv_A = np.dot(a.T, np.linalg.solve(m, a)) + np.dot(e.T, e)
    ET_b = np.dot(e.T, y0)
    x = np.linalg.solve(AT_Minv_A, ET_b)
    return x


def minimize(data):
    A, E, M, y0 = data

    def f(x):
        Ax = A.dot(x)
        Exy = E.dot(x) - y0
        return np.dot(Ax, spsolve(M, Ax)) + np.dot(Exy, Exy)

    # Set x0 to be the average of y0
    x0 = np.full(A.shape[0], np.sum(y0) / y0.shape[0])
    out = scipy.optimize.minimize(f, x0, method="bfgs")
    x = out.x
    return x


def sparse_cg(data):
    A, E, M, y0 = data

    def matvec(x):
        Ax = A.dot(x)
        return A.T.dot(sparse.linalg.spsolve(M, Ax)) + E.T.dot(E.dot(x))

    lop = pykry.LinearOperator((E.shape[1], E.shape[1]), float, dot=matvec)

    ET_b = E.T.dot(y0)
    try:
        out = pykry.cg(lop, ET_b, tol=1.0e-10, maxiter=10000)
        x = out.xk
    except krypy.utils.ConvergenceError:
        x = np.nan
    return x


def scipy_cg(data):
    A, E, M, y0 = data

    def matvec(x):
        Ax = A.dot(x)
        return A.T.dot(sparse.linalg.spsolve(M, Ax)) + E.T.dot(E.dot(x))

    lop = scipy.sparse.linalg.LinearOperator((E.shape[1], E.shape[1]), matvec=matvec)

    ET_b = E.T.dot(y0)
    x, info = scipy.sparse.linalg.cg(lop, ET_b, tol=1.0e-10, maxiter=10000)

    if info != 0:
        x = np.nan
    return x


def scipy_cg_without_m(data):
    A, E, _, y0 = data

    def matvec(x):
        return A.T.dot(A.dot(x)) + E.T.dot(E.dot(x))

    lop = scipy.sparse.linalg.LinearOperator((E.shape[1], E.shape[1]), matvec=matvec)

    ET_b = E.T.dot(y0)
    x, info = scipy.sparse.linalg.cg(lop, ET_b, tol=1.0e-10, maxiter=10000)

    if info != 0:
        x = np.nan
    return x


def scipy_lsqr_without_m(data):
    A, E, _, y0 = data

    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=lambda x: np.concatenate([A @ x, E @ x]),
        rmatvec=lambda y: A.T @ y[:A.shape[0]] + E.T @ y[A.shape[0]:]
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b)

    x = out[0]
    # num_iter = out[2]
    return x


pb = perfplot.bench(
    setup=setup,
    kernels=[
        dense_direct,
        # minimize,
        # sparse_cg,
        # scipy_cg,
        scipy_cg_without_m,
        scipy_lsqr_without_m,
    ],
    n_range=range(5, 61, 5),
    equality_check=None,
    xlabel="n",
)
pb.show()
