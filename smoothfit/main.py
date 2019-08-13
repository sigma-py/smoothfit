import numpy
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

import pykry


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


def fit1d(x0, y0, a, b, n, lmbda, solver="dense", degree=1):
    mesh = IntervalMesh(n, a, b)
    V = FunctionSpace(mesh, "CG", degree)
    return fit(x0[:, numpy.newaxis], y0, V, lmbda, solver=solver)


# def fit_triangle(x0, y0, corners, eps):
#     return


def fit_polygon(x0, y0, eps, corners, char_length):
    # Create the mesh with pygmsh
    import pygmsh

    geom = pygmsh.built_in.Geometry()
    corners3d = numpy.column_stack([corners, numpy.zeros(len(corners))])
    geom.add_polygon(corners3d, lcar=char_length)
    points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    cells = cells["triangle"]

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

    # Only allow degree 1 for now. It's unclear how many Dirichlet points are
    # needed to make the preconditioning operator positive definite.
    V = FunctionSpace(mesh, "CG", 1)

    # Find the indices corresponding to the corners
    gdim = mesh.geometry().dim()
    dofs_x = V.tabulate_dof_coordinates().reshape(-1, gdim)
    i = []
    for corner in corners:
        diff = dofs_x - corner
        norm_diff = numpy.einsum("ij, ij->i", diff, diff)
        i.append(numpy.where(abs(norm_diff) < 1.0e-15)[0][0])

    Eps = numpy.array([[2 * eps, eps], [eps, 2 * eps]])
    return fit(x0, y0, V, Eps, prec_dirichlet_indices=None)


def fit2d(x0, y0, points, cells, eps, degree=1, solver="gmres"):
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

    # Eps = numpy.array([[eps, eps], [eps, eps]])
    # Eps = numpy.array([[eps, 0], [0, eps]])
    Eps = numpy.array([[2 * eps, eps], [eps, 2 * eps]])
    # Eps = numpy.array([[1.0, 1.0], [1.0, 1.0]])

    V = FunctionSpace(mesh, "CG", degree)
    return fit(x0, y0, V, Eps, solver=solver)


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


def fit(x0, y0, V, lmbda, solver, prec_dirichlet_indices=None):
    """We're trying to minimize

       sum_i (f(xi) - yi)^2  +  lmbda ||Delta f||^2_{L^2(Omega)}

    over all functions f from V. The discretization of this is

       ||E(f) - y||_2^2 + lmbda ||Delta_h f_h||^2_{M^{-1}}

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
    A *= numpy.sqrt(lmbda)

    E = _build_eval_matrix(V, x0)

    # mass matrix
    M = _assemble_eigen(u * v * dx).sparray()

    # Scipy implementations of both LSQR and LSMR can only be used with the standard l_2
    # inner product. This is not sufficient here: We need the M inner product to make
    # sure that the discrete residual is an approximation to the inner product of the
    # continuous problem.
    if solver == "minimization":

        def f(x):
            Ax = A.dot(x)
            Exy = E.dot(x) - y0
            return numpy.dot(Ax, sparse.linalg.spsolve(M, Ax)) + numpy.dot(Exy, Exy)

        # Set x0 to be the average of y0
        x0 = numpy.full(A.shape[0], numpy.sum(y0) / y0.shape[0])
        out = scipy.optimize.minimize(f, x0, method="Powell")
        x = out.x

    elif solver == "dense":
        # Minv is dense, yikes!
        a = A.toarray()
        m = M.toarray()
        e = E.toarray()
        AT_Minv_A = numpy.dot(a.T, numpy.linalg.solve(m, a)) + numpy.dot(e.T, e)
        ET_b = numpy.dot(e.T, y0)
        x = numpy.linalg.solve(AT_Minv_A, ET_b)

    else:
        assert solver == "sparse"

        def matvec(x):
            has_extra_dimension = False
            if len(x.shape) == 2:
                assert x.shape[1] == 1
                x = x[:, 0]
                has_extra_dimension = True
            out = A.T.dot(sparse.linalg.spsolve(M, A.dot(x))) + E.T.dot(E.dot(x))
            if has_extra_dimension:
                out = out[:, None]
            return out

        lop = sparse.linalg.LinearOperator((E.shape[1], E.shape[1]), matvec=matvec)

        ET_b = E.T.dot(y0)
        out = pykry.cg(lop, ET_b, tol=1.0e-10, maxiter=1000)
        x = out.xk

        # import matplotlib.pyplot as plt
        # plt.semilogy(out.resnorms)
        # plt.grid()
        # plt.show()

    u = Function(V)
    u.vector().set_local(x)
    return u
