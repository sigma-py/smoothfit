"""Play around with preconditioners for LSQR.
"""
import krylov
import krypy
import meshzoo
import numpy as np
import perfplot
import pyamg
import scipy.optimize
import skfem
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, spsolve
from skfem.helpers import dot
from skfem.models.poisson import laplace

rng = np.random.default_rng(0)


def setup(n):
    x0 = rng.random((n, 2)) - 0.5
    # y0 = np.ones(n)
    # y0 = x0[:, 0]
    # y0 = x0[:, 0]**2
    # y0 = np.cos(np.pi*x0.T[0])
    # y0 = np.cos(np.pi*x0.T[0]) * np.cos(np.pi*x0.T[1])
    y0 = np.cos(np.pi * np.sqrt(x0.T[0] ** 2 + x0.T[1] ** 2))

    points, cells = meshzoo.rectangle_tri((-1.0, -1.0), (1.0, 1.0), n)

    mesh = skfem.MeshTri(points.T.copy(), cells.T.copy())
    element = skfem.ElementTriP1()

    @skfem.BilinearForm
    def mass(u, v, _):
        return u * v

    @skfem.BilinearForm
    def flux(u, v, w):
        return dot(w.n, u.grad) * v

    basis = skfem.InteriorBasis(mesh, element)
    facet_basis = skfem.FacetBasis(basis.mesh, basis.elem)

    lap = skfem.asm(laplace, basis)
    boundary_terms = skfem.asm(flux, facet_basis)

    A = lap - boundary_terms
    # A *= lmbda

    # get the evaluation matrix
    E = basis.probes(x0.T)

    # mass matrix
    M = skfem.asm(mass, basis)

    # x = _solve(A, M, E, y0, solver)

    # # Neumann preconditioner
    # An = _assemble_eigen(dot(grad(u), grad(v)) * dx).sparray()

    # # Dirichlet preconditioner
    # Ad = _assemble_eigen(dot(grad(u), grad(v)) * dx)
    # bc = DirichletBC(V, 0.0, "on_boundary")
    # bc.apply(Ad)
    # # Ad = Ad.sparray()

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
    # mlT = pyamg.smoothed_aggregation_solver(
    #     A.T, coarse_solver="jacobi", symmetry="nonsymmetric", max_coarse=100
    # )

    P = ml.aspreconditioner()
    # PT = mlT.aspreconditioner()

    # construct transpose -- dense, super expensive!
    I = np.eye(P.shape[0])
    PT = (P @ I).T
    PT = scipy.sparse.csr_matrix(PT)

    # # make sure it's really the transpose
    # x = rng.random(A.shape[1])
    # y = rng.random(A.shape[1])
    # print(np.dot(x, P @ y))
    # print(np.dot(PT @ x, y))

    def matvec(x):
        return P @ x

    def rmatvec(y):
        return PT @ y

    precs = [
        scipy.sparse.linalg.LinearOperator(A.shape, matvec=matvec, rmatvec=rmatvec)
        # not working well:
        # (ml.aspreconditioner(), mlT.aspreconditioner())
    ]

    return A, E, M, precs, y0


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

    lop = LinearOperator((E.shape[1], E.shape[1]), dtype=float, matvec=matvec)

    ET_b = E.T.dot(y0)
    try:
        out = krylov.cg(lop, ET_b, tol=1.0e-10, maxiter=10000)
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


def a_identity(data):
    """Super simple: A is the identity matrix. For testing purposes. Always uses around
    10 LSQR iterations, independent of n and m."""
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

    # num_iter = out[2]
    # print(f"{n = }, {m = }, {num_iter = }")

    x = out[0]
    return x


def _lprec(A, E, P, y0):
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == E.shape[1]
    n = A.shape[0]
    m = E.shape[0]

    lop = scipy.sparse.linalg.LinearOperator(
        (n + m, n),
        matvec=lambda x: np.concatenate([P @ (A @ x), E @ x]),
        rmatvec=lambda y: A.T @ (P.T @ y[:n]) + E.T @ y[n:],
    )

    b = np.concatenate([np.zeros(n), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)
    x = out[0]
    return x


def _rprec(A, E, P, y0):
    assert A.shape[0] == A.shape[1]
    assert A.shape[1] == E.shape[1]
    n = A.shape[0]
    m = E.shape[0]

    lop = scipy.sparse.linalg.LinearOperator(
        (n + m, n),
        matvec=lambda x: np.concatenate([A @ (P @ x), E @ x]),
        rmatvec=lambda y: P.T @ (A.T @ y[:n]) + E.T @ y[n:],
    )

    b = np.concatenate([np.zeros(n), y0])
    out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)

    x = out[0]
    x = P @ x
    return x


def lsqr_prec0(data):
    A, E, _, P, y0 = data
    return _lprec(A, E, P[0], y0)


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
        lsqr_prec0,
    ],
    n_range=range(10, 1001, 10),
    equality_check=None,
    max_time=4.0,
    xlabel="n",
)
