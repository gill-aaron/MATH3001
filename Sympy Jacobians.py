# -*- coding: utf-8 -*-

from sympy import (Symbol, Function, diff, pprint, init_printing, simplify, solveset, S, ConditionSet, real_roots, 
                                                                                                        N, Eq)
from sympy.matrices import Matrix

init_printing(use_unicode=True)

h = Symbol("h", nonzero=True, real=True)

t = Symbol("t", positive=True, real=True)
GM = Symbol("GM", positive=True, real=True)

r = Function("r", real=True)(t)
v = diff(r)

# Setting fixed values of h and GM can give interesting results about the relationships
# between those variables and phase-space volume

#h = 0
#GM = 1


# The norm of the position vector is specified only as a generic function. Its exact value is not needed for the proofs below.
norm = Function("n", real=True, positive=True)(t)
Fr = -GM * r / norm**3


# FORWARD EULER METHOD
pprint("FORWARD EULER METHOD")
pprint("")

r_2 = r + h * v
v_2 = v + h * Fr


J_mat = Matrix(2,2,[simplify(diff(r_2, r)),
                simplify(diff(r_2, v)),
                simplify(diff(v_2, r)),
                simplify(diff(v_2, v))]
                )

pprint("")
pprint("Jacobian matrix of the Forward Euler Method mapping:")

pprint(simplify(J_mat))

pprint("")
pprint("Jacobian determinant of the Forward Euler Method mapping:")

pprint(simplify(J_mat.det()))

# The solve function is chosen over the (generally preferred) solveset function so that h = 0 may be excluded as a solution.
# Only the solve function takes given assumptions into account when giving solutions.
# We can now demonstrate that there are no solutions for which the Jacobian equals 1.
pprint("")
pprint("Set of (nonzero, real) step sizes where phase-space volume is conserved:")
pprint(solveset(J_mat.det()-1, h, domain=ConditionSet(h, (h > 0) | (h < 0), S.Reals)))
pprint("\n\n")


# MODIFIED EULER METHOD
pprint("MODIFIED EULER METHOD")
pprint("")

r_2 = r + h * v
norm2 = Function("n2", real=True, positive=True)(t)
Fr2 = -GM * r_2 / norm2**3
v_2 = v + h * Fr2


J_mat = Matrix(2,2,[simplify(diff(r_2, r)),
                    simplify(diff(r_2, v)),
                    simplify(diff(v_2, r)),
                    simplify(diff(v_2, v))]
                    )


pprint("Jacobian matrix of the Modified Euler Method mapping:")

pprint(simplify(J_mat))

pprint("")
pprint("Jacobian determinant of the Modified Euler Method mapping:")

pprint(simplify(J_mat.det()))

pprint("")
pprint("Set of (nonzero, real) step sizes where phase-space volume is conserved:")
pprint(solveset(J_mat.det()-1, h, domain=ConditionSet(h, (h > 0) | (h < 0), S.Reals)))
pprint("\n\n")


# LEAPFROG METHOD
pprint("LEAPFROG METHOD")
pprint("")

r_2 = r + h / 2 * v
norm2 = Function("n2", real=True, positive=True)(t)
Fr2 = -GM * r_2 / norm2**3
v_2 = v + h * Fr2
r_2 += h / 2 * v_2


J_mat = Matrix(2,2,[simplify(diff(r_2, r)),
                    simplify(diff(r_2, v)),
                    simplify(diff(v_2, r)),
                    simplify(diff(v_2, v))]
                    )


pprint("Jacobian matrix of the Leapfrog Method mapping:")

pprint(simplify(J_mat))

pprint("")
pprint("Jacobian determinant of the Leapfrog Method mapping:")

pprint(simplify(J_mat.det()))

pprint("")
pprint("Set of (nonzero, real) step sizes where phase-space volume is conserved:")
pprint(solveset(J_mat.det()-1, h, domain=ConditionSet(h, (h > 0) | (h < 0), S.Reals)))
pprint("\n\n")


# RUNGE-KUTTA 4 METHOD
pprint("RUNGE-KUTTA 4 METHOD")
pprint("")

rk1 = h * v
vk1 = h * Fr

k_1 = h * Matrix(1,2,[rk1, vk1])

norm2 = Function("n2", real=True, positive=True)(t)

rk2 = h * (v + k_1[0]/ 2) #(r + k_1[0] / 2) + h * (v + k_1[0]/ 2)
vk2 = h * -GM * (r + k_1[1] / 2) / norm2**3 #(v + k_1[1] / 2) + h * -GM * (r + k_1[1] / 2) / normr2**3

k_2 = h * Matrix(1,2,[rk2, vk2])

norm3 = Function("n3", real=True, positive=True)(t)

rk3 = h * (v + k_2[0] / 2) #(r + k_2[0] / 2) + h * (v + k_2[0] / 2)
vk3 = h * -GM * (r + k_2[1] / 2) / norm3**3 #(v + k_2[1] / 2) + h * -GM * (r + k_2[1] / 2) / normr3**3

k_3 = h * Matrix(1,2,[rk3, vk3])

norm4 = Function("n4", real=True, positive=True)(t)

rk4 = h * (v + k_3[0] / 2) #(r + k_3[0] / 2) + h * (v + k_3[0] / 2)
vk4 = h * -GM * (r + k_3[1] / 2) / norm4**3 #(v + k_3[1] / 2) + h * -GM * (r + k_3[1] / 2) / normr4**3

k_4 = h * Matrix(1,2,[rk4, vk4])

r_2 = r + (rk1 + 2 * rk2 + 2 * rk3 + rk4) / 6
v_2 = v + (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6




J_mat = Matrix(2,2,[simplify(diff(r_2, r)),
                    simplify(diff(r_2, v)),
                    simplify(diff(v_2, r)),
                    simplify(diff(v_2, v))]
                    )

pprint("Jacobian matrix of the RK4 Method mapping:")

print("Not printed for brevity, can print by removing comments in code")
#pprint(simplify(J_mat))

pprint("")
pprint("Jacobian determinant of the RK4 Method mapping:")

print("Not printed for brevity, can print by removing comments in code")
#pprint(simplify(J_mat.det()))

pprint("")
pprint("Set of (nonzero, real) step sizes where phase-space volume is conserved:")

print("Not printed for brevity, can print by removing comments in code")
#pprint(solveset(J_mat.det()-1, h, domain=ConditionSet(h, (h > 0) | (h < 0), S.Reals)))
