import numpy
import pykry
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


def _build_eval_matrix(V, points):
    """Build the sparse m-by-n matrix that maps a coefficient set for a
    function in V to the values of that function at m given points.
    """
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

        rows.append(numpy.full(sdim, i))
        cols.append(dofmap.cell_dofs(cell_id))

        v = el.evaluate_basis_all(x, coordinate_dofs, cell_id)
        data.append(v)

    rows = numpy.concatenate(rows)
    cols = numpy.concatenate(cols)
    data = numpy.concatenate(data)

    m = len(points)
    n = V.dim()
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    return matrix


def fit1d(x0, y0, a, b, n, lmbda, solver="dense-direct", degree=1):
    mesh = IntervalMesh(n, a, b)
    V = FunctionSpace(mesh, "CG", degree)
    return fit(x0[:, numpy.newaxis], y0, V, lmbda, solver=solver)


def fit2d(x0, y0, points, cells, lmbda, degree=1, solver="sparse"):
    # Convert points, cells to dolfin mesh
    editor = MeshEditor()
    mesh = Mesh()
    # topological and geometrical dimension 2
    editor.open(mesh, "triangle", 2, 2, 1)
    editor.init_vertices(len(points))
    editor.init_cells(len(cells))
    for k, point in enumerate(points):
        editor.add_vertex(k, point[:2])
    for k, cell in enumerate(cells.astype(numpy.uintp)):
        editor.add_cell(k, cell)
    editor.close()

    V = FunctionSpace(mesh, "CG", degree)
    return fit(x0, y0, V, lmbda, solver=solver)


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


def fit(x0, y0, V, lmbda, solver, prec_dirichlet_indices=None):
    """We're trying to minimize

       sum_i w_i (f(xi) - yi)^2  +  ||lmbda Delta f||^2_{L^2(Omega)}

    over all functions f from V with weights w_i, lmbda. The discretization of this is

       ||W(E(f) - y)||_2^2 + ||lmbda Delta_h f_h||^2_{M^{-1}}

    where E is the (small and fat) evaluation operator at coordinates x_i, Delta_h is
    the discretization of Delta, and M is the mass matrix. One can either try and
    minimize this equation with a generic method or solve the linear equation

      lmbda A.T M^{-1} A x + E.T E x = E.T y0

    for the extremum x. Unfortunately, solving the linear equation is not
    straightforward. M is spd, A is nonsymmetric and rank-deficient but
    positive-semidefinite. So far, we simply use sparse CG, but a good idea for a
    preconditioner is highly welcome.
    """
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

    # Scipy implementations of both LSQR and LSMR can only be used with the standard l_2
    # inner product. This is not sufficient here: We need the M inner product to make
    # sure that the discrete residual is an approximation to the inner product of the
    # continuous problem.
    if solver == "dense-direct":
        # Minv is dense, yikes!
        a = A.toarray()
        m = M.toarray()
        e = E.toarray()
        AT_Minv_A = numpy.dot(a.T, numpy.linalg.solve(m, a)) + numpy.dot(e.T, e)
        ET_b = numpy.dot(e.T, y0)
        x = numpy.linalg.solve(AT_Minv_A, ET_b)

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
    else:

        def f(x):
            Ax = A.dot(x)
            Exy = E.dot(x) - y0
            return numpy.dot(Ax, spsolve(M, Ax)) + numpy.dot(Exy, Exy)

        # Set x0 to be the average of y0
        x0 = numpy.full(A.shape[0], numpy.sum(y0) / y0.shape[0])
        out = scipy.optimize.minimize(f, x0, method=solver)
        x = out.x

    u = Function(V)
    u.vector().set_local(x)
    return u
