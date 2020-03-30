# See <https://scicomp.stackexchange.com/q/33221/3980>
import sympy
from dolfin import (
    Expression,
    FacetNormal,
    Function,
    FunctionSpace,
    Mesh,
    MeshEditor,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    assemble,
    dot,
    ds,
    dx,
    grad,
    project,
    solve,
)


def create_dolfin_mesh(points, cells):
    # https://bitbucket.org/fenics-project/dolfin/issues/845/initialize-mesh-from-vertices
    editor = MeshEditor()
    mesh = Mesh()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(points.shape[0])
    editor.init_cells(cells.shape[0])
    for k, point in enumerate(points):
        editor.add_vertex(k, point)
    for k, cell in enumerate(cells):
        editor.add_cell(k, cell)
    editor.close()
    return mesh


mesh = UnitSquareMesh(200, 200)
# mesh = create_dolfin_mesh(*meshzoo.triangle(1500, corners=[[0, 0], [1, 0], [0, 1]]))


V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

n = FacetNormal(mesh)
# A = assemble(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds)
A = assemble(dot(grad(u), grad(v)) * dx - dot(n, grad(u)) * v * ds)
M = assemble(u * v * dx)

f = Expression("sin(pi * x[0]) * sin(pi * x[1])", element=V.ufl_element())
x = project(f, V)

Ax = A * x.vector()
Minv_Ax = Function(V).vector()
solve(M, Minv_Ax, Ax)
val = Ax.inner(Minv_Ax)

print(val)


# Exact value
x = sympy.Symbol("x")
y = sympy.Symbol("y")
f = sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)
f2 = sympy.diff(f, x, x) + sympy.diff(f, y, y)
val2 = sympy.integrate(sympy.integrate(f2 ** 2, (x, 0, 1)), (y, 0, 1))
# val2 = sympy.integrate(sympy.integrate(f2 ** 2, (y, 0, 1 - x)), (x, 0, 1))
print(sympy.N(val2))
