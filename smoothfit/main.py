# -*- coding: utf-8 -*-
#
from dolfin import (
    IntervalMesh, FunctionSpace, TrialFunction, TestFunction, assemble,
    dx, BoundingBoxTree, Point, Cell, MeshEditor, Mesh, Function,
    FacetNormal, ds, Constant, EigenMatrix
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


def fit1d(x0, y0, a, b, n, eps, degree=1, verbose=False):
    mesh = IntervalMesh(n, a, b)
    Eps = numpy.array([[eps]])
    return fit(x0[:, numpy.newaxis], y0, mesh, Eps, degree=degree, verbose=verbose)


def fit2d(x0, y0, points, cells, eps,
          degree=1, verbose=False, solver='spsolve'):
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
    # Eps = numpy.array([[1.0, 1.0], [1.0, 1.0]])

    return fit(x0, y0, mesh, Eps, degree=degree, verbose=verbose, solver=solver)


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


def fit(x0, y0, mesh, Eps, degree=1, verbose=False, solver='spsolve'):
    V = FunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    dim = mesh.geometry().dim()

    A = [
        _assemble_eigen(
            + Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
            # pylint: disable=unsubscriptable-object
            - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds
            ).sparray()
        for i in range(dim)
        for j in range(dim)
        ]

    E = _build_eval_matrix(V, x0)

    M = sparse.vstack(A + [E])
    b = numpy.concatenate([numpy.zeros(sum(a.shape[0] for a in A)), y0])

    if solver == 'spsolve':
        MTM = M.T.dot(M)
        x = sparse.linalg.spsolve(MTM, M.T.dot(b))
    elif solver == 'lsqr':
        x, istop, *_ = sparse.linalg.lsqr(
            M, b, show=verbose,
            atol=1.0e-10, btol=1.0e-10,
            )
        assert istop == 2, \
            'sparse.linalg.lsqr not successful (error code {})'.format(istop)
    elif solver == 'lsmr':
        x, istop, *_ = sparse.linalg.lsmr(
            M, b, show=verbose,
            atol=1.0e-10, btol=1.0e-10,
            # min(M.shape) is the default
            maxiter=max(min(M.shape), 10000)
            )
        assert istop == 2, \
            'sparse.linalg.lsmr not successful (error code {})'.format(istop)
    else:
        assert solver == 'gmres', 'Unknown solver \'{}\'.'.format(solver)
        A = sparse.linalg.LinearOperator(
            (M.shape[1], M.shape[1]),
            matvec=lambda x: M.T.dot(M.dot(x))
            )
        x, info = sparse.linalg.gmres(A, M.T.dot(b), tol=1.0e-12)
        assert info == 0, \
            'sparse.linalg.gmres not successful (error code {})'.format(info)

    u = Function(V)
    u.vector().set_local(x)
    return u
