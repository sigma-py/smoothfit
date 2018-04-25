# -*- coding: utf-8 -*-
#
from dolfin import (
    IntervalMesh, FunctionSpace, TrialFunction, TestFunction, assemble, dot,
    grad, dx, BoundingBoxTree, Point, Cell, MeshEditor, Mesh, Function,
    FacetNormal, ds, Constant, as_tensor, EigenMatrix, DirichletBC
    )
import matplotlib.pyplot as plt
import krypy
import numpy
import pyamg
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator



def _assemble_eigen(form, bcs=None):
    if bcs is None:
        bcs = []

    L = EigenMatrix()
    assemble(form, tensor=L)
    for bc in bcs:
        bc.apply(L)
    return L


def _spsolve(A, b):
    # Reshape for <https://github.com/scipy/scipy/issues/8772>.
    return sparse.linalg.spsolve(A, b).reshape(b.shape)


def solve(mesh, Eps, degree):
    V = FunctionSpace(mesh, 'CG', degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    dim = mesh.geometry().dim()

    A = [
        _assemble_eigen(
            + Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
            - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds
            ).sparray()
        for j in range(dim)
        for i in range(dim)
        ]

    assert_equality = True
    if assert_equality:
        # The sum of the `A`s is exactly that:
        n = FacetNormal(V.mesh())
        AA = _assemble_eigen(
            + dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
            - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds
            ).sparray()
        diff = AA - sum(A)
        assert numpy.all(abs(diff.data) < 1.0e-14)
        #
        # ATAsum = sum(a.T.dot(a) for a in A)
        # diff = AA.T.dot(AA) - ATAsum
        # # import betterspy
        # # betterspy.show(ATAsum)
        # # betterspy.show(AA.T.dot(AA))
        # # betterspy.show(ATAsum - AA.T.dot(AA))
        # print(diff.data)
        # assert numpy.all(abs(diff.data) < 1.0e-14)

    tol=1.0e-10

    def lower(x, on_boundary):
        return on_boundary and abs(x[1] + 1.0) < tol

    def upper(x, on_boundary):
        return on_boundary and abs(x[1] - 1.0) < tol

    def left(x, on_boundary):
        return on_boundary and abs(x[0] + 1.0) < tol

    def right(x, on_boundary):
        return on_boundary and abs(x[0] - 1.0) < tol

    def upper_left(x, on_boundary):
        return on_boundary and x[1] > +1.0 - tol and x[0] < -0.8

    def lower_right(x, on_boundary):
        return on_boundary and x[1] < -1.0 + tol and x[0] > 0.8

    bcs = [
        DirichletBC(V, Constant(0.0), lower_right),
        DirichletBC(V, Constant(0.0), upper_left),
        # DirichletBC(V, Constant(0.0), left),
        # DirichletBC(V, Constant(0.0), right),
        # DirichletBC(V, Constant(0.0), upper_left, method='pointwise'),
        # DirichletBC(V, Constant(0.0), lower_left, method='pointwise'),
        # DirichletBC(V, Constant(0.0), lower_right, method='pointwise'),
        ]

    M = _assemble_eigen(u*v*dx).sparray()

    ATMinvAsum = sum(
        numpy.dot(a.toarray().T, numpy.linalg.solve(M.toarray(), a.toarray()))
        for a in A
        )

    AA2 = _assemble_eigen(
        + dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
        - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds,
        bcs=[DirichletBC(V, Constant(0.0), 'on_boundary')]
        # bcs=bcs
        # bcs=[
        #     DirichletBC(V, Constant(0.0), lower),
        #     DirichletBC(V, Constant(0.0), right),
        #     ]
        ).sparray()

    ATA2 = AA2.T.dot(numpy.linalg.solve(M.toarray(), AA2.toarray()))

    # Eigenvalues of the operators
    if False:
        ATMinvAsum_eigs = numpy.sort(numpy.linalg.eigvalsh(ATMinvAsum))
        ATA2_eigs = numpy.sort(numpy.linalg.eigvals(ATA2))
        # print(ATA2_eigs[:20])
        # exit(1)
        plt.semilogy(ATMinvAsum_eigs, '.', label='ATMinvAsum')
        plt.semilogy(ATA2_eigs, '.', label='ATA2')
        plt.legend()
        plt.show()

    # Preconditioned eigenvalues
    if False:
        IATA_eigs = numpy.sort(scipy.linalg.eigvalsh(ATMinvAsum, ATA2))
        plt.semilogy(IATA_eigs, '.', label='precond eigenvalues')
        plt.legend()
        plt.show()

    # # Test with A only
    # numpy.random.seed(123)
    # b = numpy.random.rand(sum(a.shape[0] for a in A))
    # MTM = M.T.dot(M)
    # MTb = M.T.dot(b)
    # sol = _gmres(
    #     MTM,
    #     # TODO linear operator
    #     # lambda x: M.T.dot(M.dot(x)),
    #     MTb,
    #     M=prec
    #     )
    # plt.semilogy(sol.resnorms)
    # plt.show()
    # exit(1)

    n = AA2.shape[0]

    # define the operator
    def matvec(x):
        # M^{-1} can be computed in O(n) with CG + diagonal preconditioning
        # or algebraic multigrid.
        # return sum([a.T.dot(a.dot(x)) for a in A])
        return numpy.sum([a.T.dot(_spsolve(M, a.dot(x))) for a in A], axis=0)
    op = sparse.linalg.LinearOperator((n, n), matvec=matvec)

    # pick a random solution and a consistent rhs
    x = numpy.random.rand(n)
    b = op.dot(x)

    linear_system = krypy.linsys.LinearSystem(op, b)
    print('unpreconditioned solve...')
    out = krypy.linsys.Gmres(linear_system, tol=1.0e-12)
    print('done.')

    # preconditioned solver
    ml = pyamg.smoothed_aggregation_solver(AA2)
    # res = []
    # b = numpy.random.rand(AA2.shape[0])
    # x0 = numpy.zeros(AA2.shape[1])
    # x = ml.solve(b, x0, residuals=res, tol=1.0e-12)
    # print(res)
    # plt.semilogy(res)
    # plt.show()

    mlT = pyamg.smoothed_aggregation_solver(AA2.T.tocsr())
    # res = []
    # b = numpy.random.rand(AA2.shape[0])
    # x0 = numpy.zeros(AA2.shape[1])
    # x = mlT.solve(b, x0, residuals=res, tol=1.0e-12)

    # print(res)
    def prec_matvec(b):
        x0 = numpy.zeros(n)
        b1 = mlT.solve(b, x0, tol=1.0e-12)
        b2 = M.dot(b1)
        x = ml.solve(b2, x0, tol=1.0e-12)
        return x
    prec = LinearOperator((n, n), matvec=prec_matvec)

    # TODO assert this in a test
    # x = prec_matvec(b)
    # print(b - AA2.T.dot(AA2.dot(x)))

    linear_system = krypy.linsys.LinearSystem(op, b, M=prec)
    print('preconditioned solve...')
    try:
        out_prec = krypy.linsys.Gmres(linear_system, tol=1.0e-12, maxiter=100)
    except krypy.utils.ConvergenceError:
        print('prec not converged!')
        pass
    print('done.')

    plt.semilogy(out.resnorms, label='original')
    plt.semilogy(out_prec.resnorms, label='preconditioned')
    plt.legend()
    plt.show()

    return out.xk


if __name__ == '__main__':
    # 1d mesh
    mesh = IntervalMesh(300, -1.0, +1.0)
    Eps = numpy.array([[1.0]])

    # # 2d mesh
    # import meshzoo
    # points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 20, 20)
    # editor = MeshEditor()
    # mesh = Mesh()
    # # topological and geometrical dimension 2
    # editor.open(mesh, 'triangle', 2, 2, 1)
    # editor.init_vertices(len(points))
    # editor.init_cells(len(cells))
    # for k, point in enumerate(points):
    #     editor.add_vertex(k, point[:2])
    # for k, cell in enumerate(cells.astype(numpy.uintp)):
    #     editor.add_cell(k, cell)
    # editor.close()
    # Eps = numpy.array([[2.0, 1.0], [1.0, 2.0]])

    solve(mesh, Eps, degree=1)
