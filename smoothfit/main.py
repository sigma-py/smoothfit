# -*- coding: utf-8 -*-
#
from dolfin import (
    IntervalMesh, FunctionSpace, TrialFunction, TestFunction, assemble,
    dot, grad, dx, as_backend_type, BoundingBoxTree, Point, Cell
    )
from scipy import sparse
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy


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


def fit(x0, y0, a, b, n, eps, verbose=False):
    mesh = IntervalMesh(50, a, b)
    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    L = assemble(dot(eps * grad(u), grad(v)) * dx)

    # convert to scipy matrix
    Lmat = as_backend_type(L).mat()
    indptr, indices, data = Lmat.getValuesCSR()
    size = Lmat.getSize()
    A = sparse.csr_matrix((data, indices, indptr), shape=size)

    # delete first and last row
    # https://stackoverflow.com/a/13084858/353337
    mask = numpy.ones(A.shape[0], dtype=bool)
    mask[0] = False
    mask[-1] = False
    # unfortunatly I think boolean indexing does not work:
    w = numpy.flatnonzero(mask)
    A = A[w, :]

    AT = A.getH()

    E = _build_eval_matrix(V, x0)
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

    return mesh.coordinates(), out.x[::-1]
