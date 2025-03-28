# -*- coding: utf-8 -*-

from sympy import Symbol, Function, diff, pprint, init_printing, sqrt, simplify

init_printing(use_unicode=True)

t = Symbol("t", positive=True, real=True)
GM = Symbol("GM", positive=True, real=True)

r_x = Function("r_1", real=True)(t)
r_y = Function("r_2", real=True)(t)
v_x = diff(r_x)
v_y = diff(r_y)


norm = sqrt(r_x**2 + r_y**2)
Fx =  -GM * r_x / norm**3
Fy = -GM * r_y / norm**3

E = (v_x**2 + v_y**2) / 2 - GM / norm

energy_conservation = diff(E, t)
energy_conservation = energy_conservation.subs(diff(v_x), Fx)
energy_conservation = energy_conservation.subs(diff(v_y), Fy)

L = r_x * v_y - r_y * v_x

angular_conservation = diff(L, t)
angular_conservation = angular_conservation.subs(diff(v_x), Fx)
angular_conservation = angular_conservation.subs(diff(v_y), Fy)

pprint("Rate of Change of Energy:")
pprint(simplify(energy_conservation))

pprint("Rate of Change of Angular Momentum:")
pprint(simplify(angular_conservation))


