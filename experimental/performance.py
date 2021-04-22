import krypy
import meshzoo
import numpy as np
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
    FunctionSpace,
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

    # # Neumann preconditioner
    # An = _assemble_eigen(dot(grad(u), grad(v)) * dx).sparray()

    # Dirichlet preconditioner
    Ad = _assemble_eigen(dot(grad(u), grad(v)) * dx)
    bc = DirichletBC(V, 0.0, "on_boundary")
    bc.apply(Ad)
    # Ad = Ad.sparray()

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

    P = ml.aspreconditioner()
    # PT = mlT.aspreconditioner()

    # x = np.random.rand(A.shape[1])
    # y = np.random.rand(A.shape[1])

    P = [
        (
            scipy.sparse.linalg.LinearOperator(
                A.shape, matvec=lambda x: ml.solve(x, tol=1.0e-8)
            ),
            scipy.sparse.linalg.LinearOperator(
                A.shape, matvec=lambda x: mlT.solve(x, tol=1.0e-8)
            ),
        ),
        # not working well:
        # (ml.aspreconditioner(), mlT.aspreconditioner())
    ]
    # P.append(scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=2)
    # ))
    # P.append(scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=3)
    # ))
    # P.append(scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=4)
    # ))
    # P.append(scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=5)
    # ))
    # P.append(scipy.sparse.linalg.LinearOperator(
    #     An.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=10)
    # ))

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi")
    # P1 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=1)
    # )

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi")
    # P2 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=2)
    # )

    # ml = pyamg.smoothed_aggregation_solver(
    #     A, coarse_solver="gauss_seidel", max_coarse=100
    # )
    # P2 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=1)
    # )

    # accel="cg" is a really bad idea
    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=100)
    # P3 = scipy.sparse.linalg.LinearOperator(
    #     A.shape,
    #     matvec=lambda x: ml.solve(x, tol=1.0e-10, accel="cg")
    # )

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=100)
    # P3 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=2)
    # )

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=1000)
    # P4 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=2)
    # )

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=100)
    # P5 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=0.0, maxiter=10)
    # )

    # ml = pyamg.smoothed_aggregation_solver(A, coarse_solver="jacobi", max_coarse=100)
    # P6 = scipy.sparse.linalg.LinearOperator(
    #     A.shape, matvec=lambda x: ml.solve(x, tol=1.0e-10)
    # )

    return A, E, M, P, y0


def dense_direct(data):
    A, E, M, _, y0 = data
    # Minv is dense, yikes!
    a = A.toarray()
    m = M.toarray()
    e = E.toarray()
    AT_Minv_A = np.dot(a.T, np.linalg.solve(m, a)) + np.dot(e.T, e)
    ET_b = np.dot(e.T, y0)
    x = np.linalg.solve(AT_Minv_A, ET_b)
    return x


def dense_ls(data):
    A, E, _, _, y0 = data
    a = A.toarray()
    e = E.toarray()
    AE = np.vstack([a, e])
    b = np.concatenate([np.zeros(A.shape[0]), y0])
    return np.linalg.lstsq(AE, b)


def minimize(data):
    A, E, M, _, y0 = data

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
    A, E, M, _, y0 = data

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
    A, E, M, _, y0 = data

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
    A, E, _, _, y0 = data

    def matvec(x):
        return A.T.dot(A.dot(x)) + E.T.dot(E.dot(x))

    lop = scipy.sparse.linalg.LinearOperator((E.shape[1], E.shape[1]), matvec=matvec)

    ET_b = E.T.dot(y0)
    x, info = scipy.sparse.linalg.cg(lop, ET_b, tol=1.0e-10, maxiter=10000)

    if info != 0:
        x = np.nan
    return x


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


def scipy_lsmr_without_m(data):
    A, E, _, _, y0 = data

    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=lambda x: np.concatenate([A @ x, E @ x]),
        rmatvec=lambda y: A.T @ y[: A.shape[0]] + E.T @ y[A.shape[0] :],
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsmr(lop, b, atol=1.0e-10)

    x = out[0]
    # num_iter = out[2]
    # conda = out[6]
    return x


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


def a_identity(data):
    """Super simple: A is the identity matrix. For testing purposes. Always uses around 10
    LSQR iterations, independent of n and m."""
    A, E, _, _, y0 = data

    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == E.shape[1]
    n = A.shape[0]
    m = E.shape[0]

    lop = scipy.sparse.linalg.LinearOperator(
        (n + m, n),
        matvec=lambda x: np.concatenate([x, E @ x]),
        rmatvec=lambda y: y[:n] + E.T @ y[n:],
    )

    b = np.concatenate([np.zeros(n), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    num_iter = out[2]
    print(f"{n = }, {m = }, {num_iter = }")

    x = out[0]
    return x


def _lprec(A, E, P_PT, y0):
    P, PT = P_PT
    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=lambda x: np.concatenate([P @ (A @ x), E @ x]),
        rmatvec=lambda y: A.T @ (PT @ y[: A.shape[0]]) + E.T @ y[A.shape[0] :],
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    x = out[0]
    return x


def _rprec(A, E, P, y0):
    lop = scipy.sparse.linalg.LinearOperator(
        (A.shape[0] + E.shape[0], A.shape[1]),
        matvec=lambda x: np.concatenate([A @ (P @ x), E @ x]),
        rmatvec=lambda y: P.T @ (A.T @ y[: A.shape[0]]) + E.T @ y[A.shape[0] :],
    )

    b = np.concatenate([np.zeros(A.shape[0]), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    x = out[0]
    x = P @ x
    return x


def lsqr_prec0(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[0], y0)


def lsqr_prec1(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[1], y0)


def lsqr_prec2(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[2], y0)


def lsqr_prec3(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[3], y0)


def lsqr_prec4(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[4], y0)


def lsqr_prec5(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[5], y0)


pb = perfplot.live(
    setup=setup,
    kernels=[
        # dense_direct,
        # dense_ls,
        # # minimize,
        # # sparse_cg,
        # # scipy_cg,
        # scipy_cg_without_m,
        # scipy_lsqr_without_m,
        # scipy_lsmr_without_m,
        a_identity,
        # lsqr_prec0,
        # lsqr_prec1,
        # lsqr_prec2,
        # lsqr_prec3,
        # lsqr_prec3,
        # lsqr_prec4,
        # lsqr_prec5,
    ],
    n_range=range(10, 1001, 10),
    equality_check=None,
    max_time=4.0,
    xlabel="n",
)
