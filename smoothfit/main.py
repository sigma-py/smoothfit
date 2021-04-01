import warnings

import numpy as np
import pykry
import scipy.optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve


def fit1d(
    x0,
    y0,
    a: float,
    b: float,
    n: int,
    lmbda: float,
    degree: int = 1,
    solver: str = "dense-direct",
    variant: str = "skfem",
):
    x0 = np.asarray(x0)
    if np.any(x0 < a) or np.any(x0 > b):
        raise ValueError("Interval (a, b) must contain all x.")

    cells = np.array([np.arange(0, n), np.arange(1, n + 1)]).T
    points = np.linspace(a, b, n + 1, endpoint=True).reshape(-1, 1)
    return fit(
        x0[:, np.newaxis],
        y0,
        points,
        cells,
        lmbda,
        degree,
        solver=solver,
        variant=variant,
    )


def fit(*args, variant="skfem", **kwargs):
    """We're trying to minimize

       1/2 sum_i (f(xi) - yi)^2  +  ||lmbda Delta f||^2_{L^2(Omega)}

    over all functions f from V with weights w_i, lmbda. The discretization of this is

       1/2 ||E(f) - y||_2^2 + ||lmbda Delta_h f_h||^2_{M^{-1}}

    where E is the (small and fat) evaluation operator at coordinates x_i, Delta_h is
    the discretization of Delta, and M is the mass matrix. One can either try and
    minimize this equation with a generic method or solve the linear equation

      lmbda^2 A.T M^{-1} A x + E.T E x = E.T y0

    for the extremum x. Unfortunately, solving the linear equation is not
    straightforward. M is spd, A is nonsymmetric and rank-deficient but
    positive-semidefinite. So far, we simply use sparse CG, but a good idea for a
    preconditioner is highly welcome.
    """
    if variant == "dolfin":
        return _fit_dolfin(*args, **kwargs)

    assert variant == "skfem"
    return _fit_skfem(*args, **kwargs)


def _fit_skfem(
    x0, y0, points, cells, lmbda: float, degree: int = 1, solver: str = "lsqr"
):
    import skfem
    from skfem.helpers import dot
    from skfem.models.poisson import laplace

    assert degree == 1

    if cells.shape[1] == 2:
        mesh = skfem.MeshLine(points.T, cells.T)
        element = skfem.ElementLineP1()
    else:
        assert cells.shape[1] == 3
        mesh = skfem.MeshTri(points.T, cells.T)
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
    A *= lmbda

    # get the evaluation matrix
    E = basis.probes(x0.T)

    # mass matrix
    M = skfem.asm(mass, basis)

    x = _solve(A, M, E, y0, solver)
    return basis, x


def _fit_dolfin(
    x0, y0, points, cells, lmbda: float, degree: int = 1, solver: str = "lsqr"
):
    from dolfin import (
        BoundingBoxTree,
        Cell,
        EigenMatrix,
        FacetNormal,
        Function,
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

    def _assemble_eigen(form):
        L = EigenMatrix()
        assemble(form, tensor=L)
        return L

    def _build_eval_matrix(V, points):
        """Build the sparse m-by-n matrix that maps a coefficient set for a function in
        V to the values of that function at m given points."""
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

    editor = MeshEditor()
    mesh = Mesh()

    # Convert points, cells to dolfin mesh
    if cells.shape[1] == 2:
        editor.open(mesh, "interval", 1, 1, 1)
    else:
        # can only handle triangles for now
        assert cells.shape[1] == 3
        # topological and geometrical dimension 2
        editor.open(mesh, "triangle", 2, 2, 1)

    editor.init_vertices(len(points))
    editor.init_cells(len(cells))
    for k, point in enumerate(points):
        editor.add_vertex(k, point)
    for k, cell in enumerate(cells.astype(np.uintp)):
        editor.add_cell(k, cell)
    editor.close()

    V = FunctionSpace(mesh, "CG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)

    mesh = V.mesh()
    n = FacetNormal(mesh)

    # omega = assemble(1 * dx(mesh))

    A = _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()
    A *= lmbda

    E = _build_eval_matrix(V, x0)

    # mass matrix
    M = _assemble_eigen(u * v * dx).sparray()

    x = _solve(A, M, E, y0, solver)
    u = Function(V)
    u.vector().set_local(x)
    return u


def _solve(A, M, E, y0, solver):
    if solver == "dense-direct":
        # Minv is dense, yikes!
        a = A.toarray()
        m = M.toarray()
        e = E.toarray()
        AT_Minv_A = np.dot(a.T, np.linalg.solve(m, a)) + np.dot(e.T, e)
        ET_b = np.dot(e.T, y0)
        x = np.linalg.solve(AT_Minv_A, ET_b)

    elif solver == "sparse-cg":

        def matvec(x):
            Ax = A.dot(x)
            return A.T.dot(sparse.linalg.spsolve(M, Ax)) + E.T.dot(E.dot(x))

        lop = pykry.LinearOperator((E.shape[1], E.shape[1]), float, dot=matvec)

        ET_b = E.T.dot(y0)
        out = pykry.cg(lop, ET_b, tol=1.0e-10, maxiter=1000)
        x = out.xk

        # import matplotlib.pyplot as plt
        # plt.semilogy(out.resnorms)
        # plt.grid()
        # plt.show()
    elif solver in ["lsqr", "lsmr"]:
        # Scipy implementations of both LSQR and LSMR can only be used with the standard
        # l_2 inner product. Let's do this here, but keep in mind that the factor M^{-1}
        # is not considered here, and lambda needs to be adapted for each different n.
        # The discrete residual is not an approximation to the inner product of the
        # continuous problem here.
        # Keep an eye on <https://scicomp.stackexchange.com/q/37115/3980>, perhaps
        # there'll be a good idea for a preconditioner one day.
        lop = scipy.sparse.linalg.LinearOperator(
            (A.shape[0] + E.shape[0], A.shape[1]),
            matvec=lambda x: np.concatenate([A @ x, E @ x]),
            rmatvec=lambda y: A.T @ y[: A.shape[0]] + E.T @ y[A.shape[0] :],
        )
        b = np.concatenate([np.zeros(A.shape[0]), y0])
        if solver == "lsqr":
            out = scipy.sparse.linalg.lsqr(lop, b, atol=1.0e-10)
        else:
            assert solver == "lsmr"
            out = scipy.sparse.linalg.lsmr(lop, b, atol=1.0e-10)
        x = out[0]
        if out[4] > 1.0e-10:
            warnings.warn(f"{solver} residual 2-norm is {out[4]}.")
    else:

        def f(x):
            Ax = A.dot(x)
            Exy = E.dot(x) - y0
            return np.dot(Ax, spsolve(M, Ax)) + np.dot(Exy, Exy)

        # Set x0 to be the average of y0
        x0 = np.full(A.shape[0], np.sum(y0) / y0.shape[0])
        out = scipy.optimize.minimize(f, x0, method=solver)
        x = out.x

    return x
