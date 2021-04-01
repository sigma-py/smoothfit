import matplotlib.pyplot as plt
import numpy as np
from dolfin import (
    EigenMatrix,
    FacetNormal,
    FunctionSpace,
    TestFunction,
    TrialFunction,
    UnitIntervalMesh,
    UnitSquareMesh,
    assemble,
    dot,
    ds,
    dx,
    grad,
)

np.random.seed(0)


def _assemble_eigen(form):
    L = EigenMatrix()
    assemble(form, tensor=L)
    return L


for k in range(3, 100):
    # mesh = UnitIntervalMesh(n)
    mesh = UnitSquareMesh(k, k)

    degree = 1
    V = FunctionSpace(mesh, "CG", degree)

    u = TrialFunction(V)
    v = TestFunction(V)
    mesh = V.mesh()
    n = FacetNormal(mesh)

    # A = _assemble_eigen(dot(grad(u), grad(v)) * dx).sparray()
    A = _assemble_eigen(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds).sparray()
    Aq = _assemble_eigen(
        dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds - dot(n, grad(v)) * u * ds
    ).sparray()

    # out = np.linalg.eigvals(A.toarray())
    # print(np.sort(np.abs(out)))
    #
    # plt.plot(out.real, out.imag, "o")
    # plt.show()
    # # plt.savefig("out.png", transparent=True, bbox_inches="tight")
    _, ax = plt.subplots(1, 2)

    out = np.linalg.eigvals(A.toarray())
    ax[0].plot(out.real, out.imag, "o")

    AqA = np.linalg.inv(Aq.toarray()) @ A.toarray()
    print(AqA.shape)
    out = np.linalg.eigvals(AqA)
    # print(np.sort(np.abs(out)))
    ax[1].plot(out.real, out.imag, "o")
    plt.show()
