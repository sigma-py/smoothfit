# -*- coding: utf-8 -*-
#
import itertools
import time

import matplotlib.pyplot as plt
import numpy
import pyamg
import scipy
from dolfin import (
    BoundingBoxTree,
    Cell,
    Constant,
    DirichletBC,
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
    XDMFFile,
    as_tensor,
    assemble,
    dot,
    ds,
    dx,
    grad,
)
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator

import krypy
import meshzoo


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
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)

    gdim = mesh.geometry().dim()

    A = [
        _assemble_eigen(
            +Constant(Eps[i, j]) * u.dx(i) * v.dx(j) * dx
            - Constant(Eps[i, j]) * u.dx(i) * n[j] * v * ds
        ).sparray()
        for j in range(gdim)
        for i in range(gdim)
    ]

    assert_equality = True
    if assert_equality:
        # The sum of the `A`s is exactly that:
        n = FacetNormal(V.mesh())
        AA = _assemble_eigen(
            +dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
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

    tol = 1.0e-10

    def lower(x, on_boundary):
        return on_boundary and x[1] < -1.0 + tol

    def upper(x, on_boundary):
        return on_boundary and x[1] > 1.0 - tol

    def left(x, on_boundary):
        return on_boundary and abs(x[0] + 1.0) < tol

    def right(x, on_boundary):
        return on_boundary and abs(x[0] - 1.0) < tol

    def upper_left(x, on_boundary):
        return on_boundary and x[1] > +1.0 - tol and x[0] < -0.8

    def lower_right(x, on_boundary):
        return on_boundary and x[1] < -1.0 + tol and x[0] > 0.8

    bcs = [
        # DirichletBC(V, Constant(0.0), lower_right),
        # DirichletBC(V, Constant(0.0), upper_left),
        DirichletBC(V, Constant(0.0), lower),
        DirichletBC(V, Constant(0.0), upper),
        # DirichletBC(V, Constant(0.0), upper_left, method='pointwise'),
        # DirichletBC(V, Constant(0.0), lower_left, method='pointwise'),
        # DirichletBC(V, Constant(0.0), lower_right, method='pointwise'),
    ]

    M = _assemble_eigen(u * v * dx).sparray()

    ATMinvAsum = sum(
        numpy.dot(a.toarray().T, numpy.linalg.solve(M.toarray(), a.toarray()))
        for a in A
    )

    AA2 = _assemble_eigen(
        +dot(dot(as_tensor(Eps), grad(u)), grad(v)) * dx
        - dot(dot(as_tensor(Eps), grad(u)), n) * v * ds,
        # bcs=[DirichletBC(V, Constant(0.0), 'on_boundary')]
        # bcs=bcs
        # bcs=[
        #     DirichletBC(V, Constant(0.0), lower),
        #     DirichletBC(V, Constant(0.0), right),
        #     ]
    ).sparray()

    ATA2 = numpy.dot(AA2.toarray().T, numpy.linalg.solve(M.toarray(), AA2.toarray()))

    # Find combination of Dirichlet points:
    if False:
        # min_val = 1.0
        max_val = 0.0
        # min_combi = []
        max_combi = []
        is_updated = False
        it = 0
        # get boundary indices
        d = DirichletBC(V, Constant(0.0), "on_boundary")
        boundary_idx = numpy.sort(list(d.get_boundary_values().keys()))
        # boundary_idx = numpy.arange(V.dim())
        # print(boundary_idx)
        # pick some at random
        # idx = numpy.sort(numpy.random.choice(boundary_idx, size=3, replace=False))
        for idx in itertools.combinations(boundary_idx, 3):
            it += 1
            print()
            print(it)

            # Replace the rows corresponding to test functions living in one cell
            # (deliberately chosen as 0) by Dirichlet rows.
            AA3 = AA2.tolil()
            for k in idx:
                AA3[k] = 0
                AA3[k, k] = 1
            n = AA3.shape[0]
            AA3 = AA3.tocsr()

            ATA2 = numpy.dot(
                AA3.toarray().T, numpy.linalg.solve(M.toarray(), AA3.toarray())
            )
            vals = numpy.sort(numpy.linalg.eigvalsh(ATA2))
            if vals[0] < 0.0:
                continue

            # op = sparse.linalg.LinearOperator(
            #     (n, n),
            #     matvec=lambda x: _spsolve(AA3, M.dot(_spsolve(AA3.T.tocsr(), x)))
            #     )
            # vals, _ = scipy.sparse.linalg.eigsh(op, k=3, which='LM')
            # vals = numpy.sort(1/vals[::-1])
            # print(vals)

            print(idx)

            # if min_val > vals[0]:
            #     min_val = vals[0]
            #     min_combi = idx
            #     is_updated = True

            if max_val < vals[0]:
                max_val = vals[0]
                max_combi = idx
                is_updated = True

            if is_updated:
                # vals, _ = scipy.sparse.linalg.eigsh(op, k=10, which='LM')
                # vals = numpy.sort(1/vals[::-1])
                # print(vals)
                is_updated = False
                # print(min_val, min_combi)
                print(max_val, max_combi)
                # plt.plot(dofs_x[:, 0], dofs_x[:, 1], 'x')
                meshzoo.plot2d(mesh.coordinates(), mesh.cells())
                dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))
                # plt.plot(dofs_x[min_combi, 0], dofs_x[min_combi, 1], 'or')
                plt.plot(dofs_x[max_combi, 0], dofs_x[max_combi, 1], "ob")
                plt.gca().set_aspect("equal")
                plt.title("smallest eigenvalue: {}".format(max_val))
                plt.show()

            # # if True:
            # if abs(vals[0]) < 1.0e-8:
            #     gdim = mesh.geometry().dim()
            #     # plt.plot(dofs_x[:, 0], dofs_x[:, 1], 'x')
            #     X = mesh.coordinates()
            #     meshzoo.plot2d(mesh.coordinates(), mesh.cells())
            #     dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))
            #     plt.plot(dofs_x[idx, 0], dofs_x[idx, 1], 'or')
            #     plt.gca().set_aspect('equal')
            #     plt.show()

        meshzoo.plot2d(mesh.coordinates(), mesh.cells())
        dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))
        # plt.plot(dofs_x[min_combi, 0], dofs_x[min_combi, 1], 'or')
        plt.plot(dofs_x[max_combi, 0], dofs_x[max_combi, 1], "ob")
        plt.gca().set_aspect("equal")
        plt.title("final smallest eigenvalue: {}".format(max_val))
        plt.show()
        exit(1)

    # Eigenvalues of the operators
    if True:
        ATMinvAsum_eigs = numpy.sort(numpy.linalg.eigvalsh(ATMinvAsum))
        ATA2_eigs = numpy.sort(numpy.linalg.eigvalsh(ATA2))
        print(ATA2_eigs[:20])
        plt.semilogy(ATMinvAsum_eigs, ".", label="ATMinvAsum")
        plt.semilogy(ATA2_eigs, ".", label="ATA2")
        plt.legend()
        plt.show()
        # Preconditioned eigenvalues
        IATA_eigs = numpy.sort(scipy.linalg.eigvalsh(ATMinvAsum, ATA2))
        plt.semilogy(IATA_eigs, ".", label="precond eigenvalues")
        plt.legend()
        plt.show()
        exit(1)

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
    print("unpreconditioned solve...")
    t = time.time()
    out = krypy.linsys.Gmres(linear_system, tol=1.0e-12, explicit_residual=True)
    out.xk = out.xk.reshape(b.shape)
    print("done.")
    print("  res: {}".format(out.resnorms[-1]))
    print(
        "  unprec res: {}".format(
            numpy.linalg.norm(b - op.dot(out.xk)) / numpy.linalg.norm(b)
        )
    )
    # The error isn't useful here; only with the nullspace removed
    # print('  error: {}'.format(numpy.linalg.norm(out.xk - x)))
    print("  its: {}".format(len(out.resnorms)))
    print("  duration: {}s".format(time.time() - t))

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

    linear_system = krypy.linsys.LinearSystem(op, b, Ml=prec)
    print()
    print("preconditioned solve...")
    t = time.time()
    try:
        out_prec = krypy.linsys.Gmres(
            linear_system, tol=1.0e-14, maxiter=1000, explicit_residual=True
        )
    except krypy.utils.ConvergenceError:
        print("prec not converged!")
        pass
    out_prec.xk = out_prec.xk.reshape(b.shape)
    print("done.")
    print("  res: {}".format(out_prec.resnorms[-1]))
    print(
        "  unprec res: {}".format(
            numpy.linalg.norm(b - op.dot(out_prec.xk)) / numpy.linalg.norm(b)
        )
    )
    print("  its: {}".format(len(out_prec.resnorms)))
    print("  duration: {}s".format(time.time() - t))

    plt.semilogy(out.resnorms, label="original")
    plt.semilogy(out_prec.resnorms, label="preconditioned")
    plt.legend()
    plt.show()

    return out.xk


if __name__ == "__main__":
    # # 1d mesh
    # mesh = IntervalMesh(300, -1.0, +1.0)
    # Eps = numpy.array([[1.0]])

    # 2d mesh
    import meshzoo

    # Triangle:
    # Dirichlet points _must_ be the corners of the triangle
    points, cells = meshzoo.triangle(8, corners=[[0, 0], [1, 0], [0, 1]])

    # Triangle (unstructured).
    # Two nodes must be in a corner, the third as close as possible to the
    # third corner.
    # Degree 2: At least 5 nodes, two in two corners, one in the middle
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_polygon([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     ], lcar=0.125
    #     )
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    # Rectangle:
    # If n is even, then the top left and bottom right points must be Dirichlet
    # points. Third point doesn't matter. If n is odd, all triplets make the
    # system nonsingular as long as they don't sit on a line.
    # If 4 points are used, and they are put in the corners of the rectangle,
    # the minimum eigenvalue is largest.
    # points, cells = meshzoo.rectangle(-1.0, 1.0, -1.0, 1.0, 6, 6)

    # Rectangle (unstructured):
    # Three points:
    # The max value is taken if one points is in the corner, the two others on
    # the opposite sides of the rectangle.
    # Four points:
    # The max value is taken for a rectangle slightly smaller and tilted.
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_polygon([
    #     [0.0, 0.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [1.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     ], lcar=0.15
    #     )
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    # Hexagon: Dirichlet points cannot sit on a line, otherwise everything is
    # permitted.
    # points, cells = meshzoo.hexagon(3)

    # Circle:
    # Any three vertices will do as long as they don't sit on a line.
    # Best for k vertices: Regular k-polygon on the outline of the circle.
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_circle([0.0, 0.0, 0.0], 1.0, 0.2)
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    # L-shape
    # Optimal: Far apart triangle.
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_polygon([
    #     [0.0, 0.0, 0.0],
    #     [+1.0, 0.0, 0.0],
    #     [+1.0, -1.0, 0.0],
    #     [-1.0, -1.0, 0.0],
    #     [-1.0, +1.0, 0.0],
    #     [0.0, +1.0, 0.0],
    #     ], lcar=0.3
    #     )
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

    # T-shape
    # Optimal: Three nodes: In the arms of the T.
    #          Four nodes:
    # import pygmsh
    # geom = pygmsh.built_in.Geometry()
    # geom.add_polygon([
    #     [+1.0, +1.0, 0.0],
    #     [-1.0, +1.0, 0.0],
    #     [-1.0, +0.5, 0.0],
    #     [-0.25, +0.5, 0.0],
    #     [-0.25, -1.0, 0.0],
    #     [+0.25, -1.0, 0.0],
    #     [+0.25, +0.5, 0.0],
    #     [+1.0, +0.5, 0.0],
    #     ], lcar=0.2
    #     )
    # points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    # cells = cells['triangle']

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
    Eps = numpy.array([[2.0, 1.0], [1.0, 2.0]])

    solve(mesh, Eps, degree=1)
