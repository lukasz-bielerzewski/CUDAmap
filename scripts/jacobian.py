#!/usr/bin/env python3

import sympy
import numpy as np

def createEmatrix(E, angleSym, angleVal) :
    w =np.array([E[3], E[4], E[5]])
    #print(w)
    v =np.array([E[0], E[1], E[2]])
    #print(v)
    w_ =np.array([[0, -E[5], E[4]],
        [E[5], 0, -E[3]],
        [-E[4], E[3], 0]])
    #print(w_)

    I = np.identity(3)

    ew0=sympy.Matrix(I) + sympy.Matrix(w_)*sympy.sin(angleSym) + sympy.Matrix(w_)*sympy.Matrix(w_)*(1-sympy.cos(angleSym))
    #print(ew0)
    #print(ew0.evalf(subs={alpha0: angle0}));
    T = (sympy.Matrix(I) - ew0)*sympy.Matrix(np.cross(w,v))
    #print(T)
    #print(T.evalf(subs={alpha0: angle0}));
    e=sympy.Matrix([[ew0[0,0], ew0[0,1], ew0[0,2], T[0]],
                    [ew0[1,0], ew0[1,1], ew0[1,2], T[1]],
                    [ew0[2,0], ew0[2,1], ew0[2,2], T[2]],
                    [0,0,0,1]])
    #print(e)

    #print(e.evalf(subs={angleSym: angleVal}));
    return e

def createGmatrix(G) :
    g = np.identity(4)
    gsym = sympy.Matrix(g)
    gsym[0,3]=G[0]
    gsym[1,3]=G[1]
    gsym[2,3]=G[2]
    
    return gsym

alpha0,alpha1,alpha2=sympy.symbols('config[0] config[1] config[2]')
g0,g1,g2=sympy.symbols('g0[0] g0[1] g0[2]')
ksi05,ksi14,ksi22,ksi24=sympy.symbols('ksi[0][5] ksi[1][4] ksi[2][2] ksi[2][4]')
gsym=sympy.Matrix([[g0], [g1], [g2]])

# leg0
angle0=-0.12217
E0=[0, 0, 0, 0, 0, 1];
E0sym=sympy.Matrix([0, 0, 0, 0, 0, ksi05]);
e1 = createEmatrix(E0sym, alpha0, angle0)

angle1=0.57596
E1=[0, 0, 0, 0, -1, 0];
E1sym=sympy.Matrix([0, 0, 0, 0, ksi14, 0]);
e2 = createEmatrix(E1sym, alpha1, angle1)

angle2=-1.0472
E2=[0, 0, -0.25, 0, -1, 0];
E2sym=sympy.Matrix([0, 0, ksi22, 0, ksi24, 0]);
e3 = createEmatrix(E2sym, alpha2, angle2)

g=[0.587, 0.19, 0];
G = createGmatrix(gsym)

fk1 = e1 * e2 *e3 * G
position = sympy.Matrix([[fk1[0,3]],[fk1[1,3]],[fk1[2,3]]])
jacobian = position.jacobian([alpha0,alpha1,alpha2])
print('Jacobian leg 0')
print(jacobian)
#print(fk1)
#print(fk1.evalf(subs={config[0]: angle0, config[1]: angle1, config[2]: angle2}));

# leg1
angle0=0.12217
E0=np.array([0, 0, 0, 0, 0, 1]);
E0sym=sympy.Matrix([0, 0, 0, 0, 0, ksi05]);
e1 = createEmatrix(E0sym, alpha0, angle0)

angle1=0.57596
E1=[0, 0, 0, 0, -1, 0];
E1sym=sympy.Matrix([0, 0, 0, 0, ksi14, 0]);
e2 = createEmatrix(E1sym, alpha1, angle1)

angle2=-1.0472
E2=[0, 0, -0.25, 0, -1, 0];
E2sym=sympy.Matrix([0, 0, ksi22, 0, ksi24, 0]);
e3 = createEmatrix(E2sym, alpha2, angle2)

g=[0.587, -0.19, 0];
G = createGmatrix(gsym)

fk1 = e1 * e2 *e3 * G
position = sympy.Matrix([[fk1[0,3]],[fk1[1,3]],[fk1[2,3]]])
jacobian = position.jacobian([alpha0,alpha1,alpha2])
print('Jacobian leg 1')
print(jacobian)
#print(fk1)
#print(fk1.evalf(subs={config[0]: angle0, config[1]: angle1, config[2]: angle2}));

# leg2
angle0=0.12217
E0=np.array([0, 0, 0, 0, 0, 1]);
E0sym=sympy.Matrix([0, 0, 0, 0, 0, ksi05]);
e1 = createEmatrix(E0sym, alpha0, angle0)

angle1=0.57596
E1=[0, 0, 0, 0, -1, 0];
E1sym=sympy.Matrix([0, 0, 0, 0, ksi14, 0]);
e2 = createEmatrix(E1sym, alpha1, angle1)

angle2=-1.0472
E2=[0, 0, -0.25, 0, -1, 0];
E2sym=sympy.Matrix([0, 0, ksi22, 0, ksi24, 0]);
e3 = createEmatrix(E2sym, alpha2, angle2)

g=[0.587, 0.19, 0];
G = createGmatrix(gsym)

fk1 = e1 * e2 *e3 * G
position = sympy.Matrix([[fk1[0,3]],[fk1[1,3]],[fk1[2,3]]])
jacobian = position.jacobian([alpha0,alpha1,alpha2])
print('Jacobian leg 2')
print(jacobian)
#print(fk1)
#print(fk1.evalf(subs={config[0]: angle0, config[1]: angle1, config[2]: angle2}));

# leg3
angle0=-0.12217
E0=np.array([0, 0, 0, 0, 0, 1]);
E0sym=sympy.Matrix([0, 0, 0, 0, 0, ksi05]);
e1 = createEmatrix(E0sym, alpha0, angle0)

angle1=0.57596
E1=[0, 0, 0, 0, -1, 0];
E1sym=sympy.Matrix([0, 0, 0, 0, ksi14, 0]);
e2 = createEmatrix(E1sym, alpha1, angle1)

angle2=-1.0472
E2=[0, 0, -0.25, 0, -1, 0];
E2sym=sympy.Matrix([0, 0, ksi22, 0, ksi24, 0]);
e3 = createEmatrix(E2sym, alpha2, angle2)

g=[0.587, -0.19, 0];
G = createGmatrix(gsym)

fk1 = e1 * e2 *e3 * G
position = sympy.Matrix([[fk1[0,3]],[fk1[1,3]],[fk1[2,3]]])
jacobian = position.jacobian([alpha0,alpha1,alpha2])
print('Jacobian leg 3')
print(jacobian)
#print(fk1)
#print(fk1.evalf(subs={config[0]: angle0, config[1]: angle1, config[2]: angle2}));
