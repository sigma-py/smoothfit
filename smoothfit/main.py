# -*- coding: utf-8 -*-
#
import itertools
from dolfin import (
    IntervalMesh, FunctionSpace, TrialFunction, TestFunction, assemble, dot,
    grad, dx, as_backend_type, BoundingBoxTree, Point, Cell, MeshEditor, Mesh,
    Function, PETScMatrix, FacetNormal, ds, Expression, XDMFFile, Constant,
    as_tensor
    )
import numpy
from scipy import sparse
from scipy.optimize import minimize


def _build_eval_matrix1d(V, points):
    '''Build the sparse m-by-n matrix that maps a coefficient set for a
    function in V to the values of that function at m given points.
    '''
    # See <https://www.allanswered.com/post/lkbkm/#zxqgk>
    mesh = V.mesh()

    bbt = BoundingBoxTree()
    bbt.build(mesh)
    dofmap = V.dofmap()
    el = V.element()
    rows = []
    cols = []
    data = []
    for i, x in enumerate(points):
        cell_id = bbt.compute_first_entity_collision(Point(x))
        cell = Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        rows.append([i, i])
        cols.append(dofmap.cell_dofs(cell_id))

        v = numpy.empty(2, dtype=float)
        el.evaluate_basis_all(v, numpy.array(x), coordinate_dofs, cell_id)
        data.append(v)

    rows = numpy.concatenate(rows)
    cols = numpy.concatenate(cols)
    data = numpy.concatenate(data)

    m = len(points)
    n = V.dim()
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    return matrix


def _build_eval_matrix2d(V, points):
    mesh = V.mesh()

    bbt = BoundingBoxTree()
    bbt.build(mesh)
    dofmap = V.dofmap()
    el = V.element()
    rows = []
    cols = []
    data = []
    for i, x in enumerate(points):
        cell_id = bbt.compute_first_entity_collision(Point(*x))
        cell = Cell(mesh, cell_id)
        coordinate_dofs = cell.get_vertex_coordinates()

        rows.append([i, i, i])
        cols.append(dofmap.cell_dofs(cell_id))

        v = numpy.empty(3, dtype=float)
        el.evaluate_basis_all(v, x, coordinate_dofs, cell_id)
        data.append(v)

    rows = numpy.concatenate(rows)
    cols = numpy.concatenate(cols)
    data = numpy.concatenate(data)

    m = len(points)
    n = V.dim()
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(m, n))
    return matrix


def fit1d(x0, y0, a, b, n, eps, verbose=False):
    mesh = IntervalMesh(n, a, b)
    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(V.mesh())
    L = assemble(
        dot(eps * grad(u), grad(v)) * dx
        # Adding this term effectively removed the boundary conditions
        - dot(eps*grad(u), n) * v * ds
        )

    # convert to scipy matrix
    Lmat = as_backend_type(L).mat()
    indptr, indices, data = Lmat.getValuesCSR()
    size = Lmat.getSize()
    A = sparse.csr_matrix((data, indices, indptr), shape=size)

    AT = A.getH()

    E = _build_eval_matrix1d(V, x0)
    ET = E.getH()

    def f(alpha):
        A_alpha = A.dot(alpha)
        d = E.dot(alpha) - y0
        return (
            0.5 * numpy.dot(A_alpha, A_alpha) + 0.5 * numpy.dot(d, d),
            AT.dot(A_alpha) + ET.dot(d)
            )
        # return (
        #     0.5 * numpy.dot(alpha, A_alpha) + 0.5 * numpy.dot(d, d),
        #     A_alpha + E.T.dot(d)
        #     )

    alpha0 = numpy.zeros(V.dim())
    out = minimize(
        f,
        alpha0,
        jac=True,
        method='L-BFGS-B',
        )
    assert out.success, 'Optimization not successful.'
    if verbose:
        print(out.nfev)
        print(out.fun)

    # The least-squares solution is actually less accurate than the minimization
    # from scipy.optimize import lsq_linear
    # out = lsq_linear(
    #     sparse.vstack([A, E]),
    #     numpy.concatenate([numpy.zeros(A.shape[0]), y0]),
    #     tol=1e-13
    #     )
    # print(out.cost)

    return mesh.coordinates(), out.x[::-1]


def fitfail(x0, y0, points, cells, eps, verbose=False):
    # Convert points, cells to dolfin mesh
    editor = MeshEditor()
    mesh = Mesh()
    # topological and geometrical dimension 2
    editor.open(mesh, 'triangle', 2, 2, 1)
    editor.init_vertices(len(points))
    editor.init_cells(len(cells))
    for k, point in enumerate(points):
        editor.add_vertex(k, point[:2])
    for k, cell in enumerate(cells.astype(numpy.uintp)):
        editor.add_cell(k, cell)
    editor.close()

    # xdmf = XDMFFile('temp.xdmf')
    # xdmf.write(mesh)
    # exit(1)

    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    # n = FacetNormal(mesh)
    # xy = Expression(('x[0]', 'x[1]'), degree=1, domain=mesh)
    # u = Expression('x[0]', degree=1, domain=mesh)
    # out1 = assemble(3 * dot(grad(u), xy) * v * dx)
    # print(out1.get_local())
    # print()

    # # out2 = assemble(-div(grad(u) - xy * dot(grad(u), xy)) * v * dx)
    # out2 = assemble(
    #     dot(grad(u) - xy * dot(grad(u), xy), grad(v)) * dx
    #     - dot(grad(u) - xy * dot(grad(u), xy), n) * v * ds
    #     )
    # print(out2.get_local())
    # print()
    # print((out1.get_local() - out2.get_local()) / out1.get_local())

    # xdmf = XDMFFile('diff.xdmf')
    # diff = Function(V)
    # diff.vector().set_local(abs(out1.get_local() - out2.get_local()) / abs(out1.get_local()))
    # xdmf.write(diff)

    # n = FacetNormal(mesh)
    # xy = Expression(('x[0]', 'x[1]'), degree=1, domain=mesh)
    # u = Expression('x[0]', degree=1, domain=mesh)
    # out1 = assemble(dot(grad(u), n) * v * ds)
    # print(out1.get_local())
    # # print()

    # Perfect if used n->xy:
    # out2 = 1/3 * assemble(-div(grad(u) - xy * dot(grad(u), xy)) * v * ds)
    # Linear error:
    # out2 = - 1/3 * assemble(div(grad(u) - xy * dot(grad(u), n)) * v * ds)
    # out2 = assemble(dot(grad(u) - xy * dot(grad(u), xy), surface_grad_v) * ds)
    # out2 = assemble(dot(grad(u) - xy * dot(grad(u), xy), grad(v)) * ds)
    # out2 = assemble(dot(grad(u) - n * dot(grad(u), n), grad(v)) * ds)
    # Quadratic error:
    # out2 = assemble(dot(grad(u) - xy * dot(grad(u), n), grad(v)) * ds)

    # out2 = assemble(dot(grad(u) - xy * dot(grad(u), n), grad(v)) * ds)

    # print(out2.get_local())
    # print()
    # print(out1.get_local() - out2.get_local())

    # xdmf = XDMFFile('diff.xdmf')
    # diff = Function(V)
    # diff.vector().set_local(abs(out1.get_local() - out2.get_local()))
    # xdmf.write(diff)
    # exit(1)

    n = FacetNormal(mesh)
    L = PETScMatrix()
    # xy = Expression(('x[0]', 'x[1]'), degree=1)
    assemble(
        eps * dot(grad(u), grad(v)) * dx
        # + eps/3 * div(grad(u) - xy * dot(grad(u), n)) * v * ds,
        # - eps * dot(grad(u) - xy * dot(grad(u), n), grad(v)) * ds,
        - eps * dot(grad(u) - n * dot(grad(u), n), grad(v)) * ds,
        # - eps * dot(grad(u), n) * v * ds,
        tensor=L
        )
    row_ptr, col_indices, data = L.mat().getValuesCSR()
    size = L.mat().getSize()
    A = sparse.csr_matrix((data, col_indices, row_ptr), shape=size)

    # print(A.shape)
    # print(numpy.sort(numpy.linalg.eigvals(A.toarray())))
    # exit(1)

    # # Interior Laplacian part.
    # # Refer to
    # # https://www.allanswered.com/post/awlpv/get-test-functions-with-no-support-on-the-boundary/
    # # for a purely Dolfinic version of this.
    # L = PETScMatrix()
    # assemble(eps * dot(grad(u), grad(v)) * dx, tensor=L)

    # def create_dirichlet():
    #     L = PETScMatrix()
    #     assemble(dot(eps * grad(u), grad(v)) * dx, tensor=L)
    #     # Check which rows belong to boundary test functions
    #     bc = DirichletBC(V, 0, lambda x, b: b)
    #     bc.apply(L)
    #     # Convert to scipy sparse matrix
    #     row_ptr, col_indices, data = L.mat().getValuesCSR()
    #     size = L.mat().getSize()
    #     D = sparse.csr_matrix((data, col_indices, row_ptr), shape=size)
    #     D.eliminate_zeros()
    #     return D
    # D = create_dirichlet()
    # print(D)
    # print(numpy.linalg.eigvals(D.todense()))

    # # Check which rows belong to boundary test functions
    # bc = DirichletBC(V, 0, lambda x, b: b)
    # # Convert to scipy sparse matrix
    # row_ptr, col_indices, data = L.mat().getValuesCSR()
    # size = L.mat().getSize()
    # A1 = sparse.csr_matrix((data, col_indices, row_ptr), shape=size)
    # # Remove boundary condition rows
    # bc_dofs = bc.get_boundary_values().keys()
    # offset = V.dofmap().ownership_range()[0]
    # interior_idx = numpy.fromiter(
    #     (i for i in range(*L.local_range(0)) if i - offset not in bc_dofs),
    #     dtype=la_index_dtype()
    #     )
    # A1 = A1[interior_idx]

    # # Surface Laplacian part
    # boundary_idx = numpy.sort(list(set(range(size[0])) - set(interior_idx)))
    # LS = PETScMatrix()
    # n = FacetNormal(mesh)
    # surface_grad_u = grad(u) - dot(n, grad(u)) * n
    # surface_grad_v = grad(v) - dot(n, grad(v)) * n
    # assemble(
    #     # + eps * dot(surface_grad_u, surface_grad_v) * ds
    #     - eps * dot(surface_grad_u, grad(v)) * ds
    #     # + eps * dot(grad(u), grad(v)) * ds
    #     # - eps * dot(n, grad(u)) * dot(n, grad(v)) * ds
    #     + 3 * eps * dot(n, grad(u)) * v * ds
    #     ,
    #     tensor=LS
    #     )
    #
    # row_ptr, col_indices, data = LS.mat().getValuesCSR()
    # size = LS.mat().getSize()
    # A2 = sparse.csr_matrix((data, col_indices, row_ptr), shape=size)
    # #
    # A2 = A2[boundary_idx]

    # # Dirichlet part TODO remove
    # A2 = sparse.lil_matrix((len(boundary_idx), A1.shape[1]))
    # for k, idx in enumerate(boundary_idx):
    #     A2[k, idx] = 1.0
    # A2 = A2.tocsr()

    # A = sparse.vstack([A1, A2])
    # idx = numpy.concatenate([interior_idx, boundary_idx])
    # ridx = numpy.argsort(idx)
    # A = A[ridx]

    # e = numpy.ones(A.shape[1])
    # print(numpy.all(abs(A.dot(e)) < 1.0e-14))
    # exit(1)

    # print(numpy.sort(numpy.linalg.eigvals(A.todense())))

    # Adense = A.toarray()
    # vals, vecs = numpy.linalg.eig(Adense)
    # assert numpy.all(abs(numpy.imag(vals)) < 1.0e-13)
    # vals = numpy.real(vals)
    # ridx_eig = numpy.argsort(vals)
    # vals = vals[ridx_eig]
    # print(vals)
    # vecs = vecs[:, ridx_eig]
    # assert numpy.all(abs(numpy.imag(vecs[:, 0])) < 1.0e-13)
    # v0 = numpy.real(vecs[:, 0])
    # # print(numpy.dot(Adense, v0) - vals[0]*v0)
    # #
    # xdmf = XDMFFile('eig.xdmf')
    # eig0 = Function(V)
    # eig0.vector().set_local(v0)
    # xdmf.write(eig0)
    # exit(1)

    from dolfin import interpolate
    ex = interpolate(Expression('x[0]', degree=1), V)
    # xdmf = XDMFFile('ex.xdmf')
    # xdmf.write(ex)
    v = ex.vector().get_local()
    Av = A.dot(v)
    print(numpy.all(abs(Av) < 1.0e-11), numpy.sqrt(numpy.dot(Av, Av)))
    # A1v = A1.dot(v)
    # print(numpy.all(abs(A1v) < 1.0e-11), numpy.sqrt(numpy.dot(A1v, A1v)))
    # A2v = A2.dot(v)
    # print(numpy.all(abs(A2v) < 1.0e-11), numpy.sqrt(numpy.dot(A2v, A2v)))
    # exit(1)
    # print(A2.dot(v))

    res = Function(V)
    res.vector().set_local(A.dot(v))
    xdmf = XDMFFile('res3.xdmf')
    xdmf.write(res)
    # exit(1)

    AT = A.getH()

    E = _build_eval_matrix2d(V, x0)
    ET = E.getH()

    def f(alpha):
        A_alpha = A.dot(alpha)
        d = E.dot(alpha) - y0
        return (
            0.5 * numpy.dot(A_alpha, A_alpha) + 0.5 * numpy.dot(d, d),
            AT.dot(A_alpha) + ET.dot(d)
            )
        # return (
        #     0.5 * numpy.dot(alpha, A_alpha) + 0.5 * numpy.dot(d, d),
        #     A_alpha + E.T.dot(d)
        #     )

    alpha0 = numpy.zeros(V.dim())
    out = minimize(
        f,
        alpha0,
        jac=True,
        method='L-BFGS-B',
        )
    assert out.success, 'Optimization not successful.'
    if verbose:
        print(out.nfev)
        print(out.fun)

    # The least-squares solution is actually less accurate than the minimization
    # from scipy.optimize import lsq_linear
    # out = lsq_linear(
    #     sparse.vstack([A, E]),
    #     numpy.concatenate([numpy.zeros(A.shape[0]), y0]),
    #     tol=1e-13
    #     )
    # print(out.cost)

    u = Function(V)
    u.vector().set_local(out.x)

    return u


def fit(x0, y0, points, cells, eps, verbose=False):
    # Convert points, cells to dolfin mesh
    editor = MeshEditor()
    mesh = Mesh()
    # topological and geometrical dimension 2
    editor.open(mesh, 'triangle', 2, 2, 1)
    editor.init_vertices(len(points))
    editor.init_cells(len(cells))
    for k, point in enumerate(points):
        editor.add_vertex(k, point[:2])
    for k, cell in enumerate(cells.astype(numpy.uintp)):
        editor.add_cell(k, cell)
    editor.close()

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    A = []
    # No need for itertools.product([0, 1], repeat=2) here. The matrices
    # corresponding to the derivatives xy, yx are equal. TODO perhaps add a
    # weight?
    # for i, j in itertools.combinations_with_replacement([0, 1], 2):
    # Eps = numpy.array([[eps, eps], [eps, eps]])
    Eps = numpy.array([[2*eps, eps], [eps, 2*eps]])
    # Eps = numpy.array([[eps, 0], [0, eps]])
    for i, j in itertools.product([0, 1], repeat=2):
        L0 = PETScMatrix()
        assemble(
            + Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
            # pylint: disable=unsubscriptable-object
            - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds,
            tensor=L0
            )
        row_ptr, col_indices, data = L0.mat().getValuesCSR()
        size = L0.mat().getSize()
        A.append(sparse.csr_matrix((data, col_indices, row_ptr), shape=size))

    AT = [a.getH() for a in A]

    E = _build_eval_matrix2d(V, x0)
    ET = E.getH()

    def f(alpha):
        A_alpha = [a.dot(alpha) for a in A]
        d = E.dot(alpha) - y0
        return (
            + 0.5 * sum([numpy.dot(a_alpha, a_alpha) for a_alpha in A_alpha])
            + 0.5 * numpy.dot(d, d),
            # gradient
            + sum([at.dot(a_alpha) for at, a_alpha in zip(AT, A_alpha)])
            + ET.dot(d)
            )
        # return (
        #     0.5 * numpy.dot(alpha, A_alpha) + 0.5 * numpy.dot(d, d),
        #     A_alpha + ET.dot(d)
        #     )

    assert_equality = False
    if assert_equality:
        # The sum of the `A`s is exactly that:
        Asum = sum(A)
        L = PETScMatrix()
        n = FacetNormal(V.mesh())
        assemble(
            + dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
            - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds,
            tensor=L
            )
        row_ptr, col_indices, data = L.mat().getValuesCSR()
        size = L.mat().getSize()
        AA = sparse.csr_matrix((data, col_indices, row_ptr), shape=size)
        assert numpy.all(Asum.indices == AA.indices)
        assert numpy.all(Asum.indptr == AA.indptr)
        assert numpy.all(abs(Asum.data - AA.data) < 1.0e-14)

    # vals, vecs = numpy.linalg.eig(Asum.todense())
    # print()
    # print('eigs Asum')
    # print(numpy.sort(vals))
    # # print(vecs[:, 0].T)
    # vals, vecs = numpy.linalg.eig(AA.todense())
    # print()
    # print('eigs AA')
    # print(numpy.sort(vals))
    # exit(1)

    alpha0 = numpy.zeros(V.dim())
    out = minimize(
        f,
        alpha0,
        jac=True,
        method='L-BFGS-B',
        )
    assert out.success, 'Optimization not successful.'
    if verbose:
        print(out.nfev)
        print(out.fun)

    # The least-squares solution is actually less accurate than the minimization
    # from scipy.optimize import lsq_linear
    # out = lsq_linear(
    #     sparse.vstack([A, E]),
    #     numpy.concatenate([numpy.zeros(A.shape[0]), y0]),
    #     tol=1e-13
    #     )
    # print(out.cost)

    u = Function(V)
    u.vector().set_local(out.x)

    return u
