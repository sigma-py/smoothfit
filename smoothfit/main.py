import numpy
import pyamg
from dolfin import (
    BoundingBoxTree,
    Cell,
    Constant,
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
    as_tensor,
    assemble,
    dot,
    ds,
    dx,
    grad,
)
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

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


def fit1d(x0, y0, a, b, n, lmbda, degree=1):
    mesh = IntervalMesh(n, a, b)
    Lmbda = numpy.array([[lmbda]])
    V = FunctionSpace(mesh, "CG", degree)

    # Find the indices corresponding to the end points
    dofs_x = V.tabulate_dof_coordinates()
    i0 = numpy.where(abs(dofs_x - a) < 1.0e-15)[0][0]
    i1 = numpy.where(abs(dofs_x - b) < 1.0e-15)[0][0]

    return fit(x0[:, numpy.newaxis], y0, V, Lmbda, prec_dirichlet_indices=[i0, i1])


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


def fit(x0, y0, V, Eps, solver="dense", prec_dirichlet_indices=None):
    u = TrialFunction(V)
    v = TestFunction(V)

    mesh = V.mesh()
    n = FacetNormal(mesh)

    # vol = assemble(Constant(1) * dx(mesh))

    # gdim = mesh.geometry().dim()
    # A = [
    #     _assemble_eigen(
    #         +Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
    #         - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds
    #     ).sparray()
    #     for i in range(gdim)
    #     for j in range(gdim)
    # ]

    A = Eps[0, 0] * _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()

    E = _build_eval_matrix(V, x0)

    # omega = assemble(1 * dx(mesh))

    # mass matrix
    M = _assemble_eigen(u * v * dx).sparray()

    # Scipy implementations of both LSQR and LSMR can only be used with the
    # standard l_2 inner product. This is not sufficient here: We need the M
    # inner product to make sure that the discrete residual is an approximation
    # to the inner product of the continuous problem.
    if solver == "dense":
        # Minv is dense, yikes!
        M = M.toarray()
        BTMinvB = numpy.dot(A.toarray().T, numpy.linalg.solve(M, A.toarray())) + E.T.dot(E)
        # BTMinvB = sum(a.T.dot(a) for a in A) + E.T.dot(E)
        BTb = E.T.dot(y0)
        x = numpy.linalg.solve(BTMinvB, BTb)

        # compute residual
        # r0 = E * x - y0
        # alpha = numpy.dot(r0, r0)
        # Ax = numpy.dot(A.toarray(), x)
        # beta = numpy.dot(Ax, numpy.linalg.solve(M, Ax))
        # print(alpha, beta, alpha + beta)
        # exit(1)

        # # Minv is dense, yikes!
        # M = M.toarray()
        # BTMinvB = sum(
        #     numpy.dot(a.toarray().T, numpy.linalg.solve(M, a.toarray())) for a in A
        # ) + E.T.dot(E)
        # # BTMinvB = sum(a.T.dot(a) for a in A) + E.T.dot(E)
        # BTb = E.T.dot(y0)
        # x = sparse.linalg.spsolve(BTMinvB, BTb)

    else:
        assert solver == "gmres", "Unknown solver '{}'.".format(solver)

        def matvec(x):
            # M^{-1} can be computed in O(n) with CG + diagonal preconditioning
            # or algebraic multigrid.
            # Reshape for <https://github.com/scipy/scipy/issues/8772>.
            s = sum([a.T.dot(sparse.linalg.spsolve(M, a.dot(x))) for a in A]).reshape(
                x.shape
            )
            return s + E.T.dot(E.dot(x))

        matrix = sparse.linalg.LinearOperator(
            (E.shape[1], E.shape[1]),
            # matvec=lambda x: B.T.dot(B.dot(x))
            matvec=matvec,
        )

        if prec_dirichlet_indices:
            # As preconditioner for `A^T M^{-1} A`, `ML(B) M ML(B.T)` where ML
            # is a multigrid solve and B is the weak form of `-\Delta u` with
            # Dirichlet conditions in only a few points, chosen such that B is
            # nonsingular. (Exactly how these points have to be chosen is still
            # a matter of research, see
            # <https://scicomp.stackexchange.com/q/29403/3980>.)
            # B approximates A quite well, especially if the domain is
            # polygonal and the indices are the corners of the polygon.

            # Weak form of `-Delta u` without boundary conditions.
            Aprec = _assemble_eigen(
                +dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
                - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds
            ).sparray()
            # Add Dirichlet conditions at a few points
            Aprec = Aprec.tolil()
            for k in prec_dirichlet_indices:
                Aprec[k] = 0
                Aprec[k, k] = 1
            n = Aprec.shape[0]
            Aprec = Aprec.tocsr()

            ml = pyamg.smoothed_aggregation_solver(Aprec)
            mlT = pyamg.smoothed_aggregation_solver(Aprec.T.tocsr())

            def prec_matvec(b):
                x0 = numpy.zeros(n)
                b1 = mlT.solve(b, x0, tol=1.0e-12)
                b2 = M.dot(b1)
                x = ml.solve(b2, x0, tol=1.0e-12)
                return x

            prec = LinearOperator((n, n), matvec=prec_matvec)

        else:
            prec = None

        # b = numpy.concatenate([numpy.zeros(sum(a.shape[0] for a in A)), y0])
        BTb = E.T.dot(y0)

        # Scipy's own GMRES.
        # x, info = sparse.linalg.gmres(matrix, BTb, tol=1.0e-10)
        # assert info == 0, \
        #     'sparse.linalg.gmres not successful (error code {})'.format(info)

        out = pykry.gmres(matrix, BTb, M=prec, tol=1.0e-10)
        print(len(out.resnorms))
        x = out.xk

    u = Function(V)
    u.vector().set_local(x)
    return u
