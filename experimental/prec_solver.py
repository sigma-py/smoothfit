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


def solve(M, b, mesh, Eps):
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    dim = mesh.geometry().dim()

    A = [[
        _assemble_eigen(
            + Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
            # pylint: disable=unsubscriptable-object
            - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds
            ).sparray()
        for j in range(dim)
        ]
        for i in range(dim)
        ]
    Aflat = [item for sublist in A for item in sublist]

    assert_equality = False
    if assert_equality:
        # The sum of the `A`s is exactly that:
        n = FacetNormal(V.mesh())
        AA = _assemble_eigen(
            + dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
            - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds
            ).sparray()
        diff = AA - sum(Aflat)
        assert numpy.all(abs(diff.data) < 1.0e-14)
        #
        # ATAsum = sum(a.T.dot(a) for a in Aflat)
        # diff = AA.T.dot(AA) - ATAsum
        # # import betterspy
        # # betterspy.show(ATAsum)
        # # betterspy.show(AA.T.dot(AA))
        # # betterspy.show(ATAsum - AA.T.dot(AA))
        # print(diff.data)
        # assert numpy.all(abs(diff.data) < 1.0e-14)

    tol=1.0e-13

    def lower(x, on_boundary):
        return on_boundary and abs(x[1] + 1.0) < tol

    def right(x, on_boundary):
        return on_boundary and abs(x[0] - 1.0) < tol

    def upper_left(x, on_boundary):
        return on_boundary and (abs(x[0] + 1.0) < tol) and (abs(x[1] - 1.0)<tol)

    def lower_left(x, on_boundary):
        return on_boundary and (abs(x[0] + 1.0) < tol) and (abs(x[1] + 1.0)<tol)

    def lower_right(x, on_boundary):
        return on_boundary and (abs(x[0] - 1.0) < tol) and (abs(x[1] + 1.0)<tol)

    bcs = [
        DirichletBC(V, Constant(0.0), upper_left, method='pointwise'),
        DirichletBC(V, Constant(0.0), lower_left, method='pointwise'),
        DirichletBC(V, Constant(0.0), lower_right, method='pointwise'),
        ]

    AA2 = _assemble_eigen(
        + dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
        - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds,
        # bcs=[DirichletBC(V, Constant(0.0), 'on_boundary')]
        # bcs=bcs
        bcs=[
            DirichletBC(V, Constant(0.0), lower),
            DirichletBC(V, Constant(0.0), right),
            ]
        ).sparray()

    ATA2 = AA2.dot(AA2)
    # ATAsum = sum(a.T.dot(a) for a in Aflat)

    # ATAsum_eigs = numpy.sort(numpy.linalg.eigvalsh(ATAsum.todense()))
    # print(ATAsum_eigs)
    # print()
    ATA2_eigs = numpy.sort(numpy.linalg.eigvalsh(ATA2.todense()))
    print(ATA2_eigs)
    exit(1)
    # plt.semilogy(range(len(ATAsum_eigs)), ATAsum_eigs, '.', label='ATAsum')
    # plt.semilogy(range(len(ATA2_eigs)), ATA2_eigs, '.', label='ATA2')
    # plt.legend()
    # plt.show()

    # # invsqrtATA2 = numpy.linalg.inv(scipy.linalg.sqrtm(ATA2.todense()))
    # # IATA = numpy.dot(numpy.dot(invsqrtATA2, ATAsum), invsqrtATA2)
    # # IATA_eigs = numpy.sort(numpy.linalg.eigvals(IATA))
    # IATA_eigs = numpy.sort(scipy.linalg.eigvals(ATAsum.todense(), ATA2.todense()))
    # plt.semilogy(range(len(IATA_eigs)), IATA_eigs, '.', label='IATA')
    # # plt.plot(IATA_eigs, numpy.zeros(len(IATA_eigs)), 'x', label='IATA')
    # plt.legend()
    # plt.show()
    # # exit(1)

    # # Test with A only
    # M = sparse.vstack(Aflat)
    # numpy.random.seed(123)
    # b = numpy.random.rand(sum(a.shape[0] for a in Aflat))
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
        n = len(b)
        x0 = numpy.zeros(n)
        b1 = mlT.solve(b, x0, tol=1.0e-12)
        x = ml.solve(b1, x0, tol=1.0e-12)
        return x
    n = AA2.shape[0]
    prec = LinearOperator((n, n), matvec=prec_matvec)

    # TODO assert this in a test
    # x = prec_matvec(b)
    # print(b - AA2.T.dot(AA2.dot(x)))

    MTM = sparse.linalg.LinearOperator(
        (M.shape[1], M.shape[1]),
        matvec=lambda x: M.T.dot(M.dot(x))
        )

    linear_system = krypy.linsys.LinearSystem(MTM, M.T.dot(b), M=prec)
    out = krypy.linsys.Gmres(linear_system, tol=1.0e-12)

    # plt.semilogy(out.resnorms)
    # plt.show()

    return out.xk


if __name__ == '__main__':
    dla
