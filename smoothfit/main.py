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


def _build_eval_matrix(V, points):
    '''Build the sparse m-by-n matrix that maps a coefficient set for a
    function in V to the values of that function at m given points.
    '''
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

        v = numpy.empty(sdim, dtype=float)
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
    Eps = numpy.array([[eps]])
    return fit(x0[:, numpy.newaxis], y0, mesh, Eps, verbose=verbose)


def fit2d(x0, y0, points, cells, eps, verbose=False):
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

    # Eps = numpy.array([[eps, eps], [eps, eps]])
    # Eps = numpy.array([[eps, 0], [0, eps]])
    Eps = numpy.array([[2*eps, eps], [eps, 2*eps]])

    return fit(x0, y0, mesh, Eps, verbose=verbose)


def fit(x0, y0, mesh, Eps, verbose=False):
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    dim = mesh.geometry().dim()

    A = []
    # No need for itertools.product([0, 1], repeat=2) here. The matrices
    # corresponding to the derivatives xy, yx are equal. TODO perhaps add a
    # weight?
    # for i, j in itertools.combinations_with_replacement([0, 1], 2):
    for i in range(dim):
        for j in range(dim):
            L0 = PETScMatrix()
            assemble(
                + Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
                # pylint: disable=unsubscriptable-object
                - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds,
                tensor=L0
                )
            row_ptr, col_indices, data = L0.mat().getValuesCSR()
            size = L0.mat().getSize()
            A.append(
                sparse.csr_matrix((data, col_indices, row_ptr), shape=size)
                )

    AT = [a.getH() for a in A]

    E = _build_eval_matrix(V, x0)
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
