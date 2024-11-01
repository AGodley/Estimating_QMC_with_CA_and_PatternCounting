from kraus import k, k_dot
from qutip import *
from absorber import ss
import numpy as np

tht_test = 0.2
lmbd_test = 0.8
phi_test = np.pi/4
M = [fock(2, 0), fock(2, 1)]

# Checks Gauge condition
K, *_ = k(tht_test, lmbd_test, phi_test, M)
K_dot, *_ = k_dot(tht_test, lmbd_test, phi_test, M)
A_sum = K_dot[0].dag() * K[0] + K_dot[1].dag() * K[1]
r_ss = ss(tht_test, lmbd_test, phi_test)
print(r_ss)
rho_ss = 1/2 * (qeye(2) + r_ss[0, 0]*sigmax() + r_ss[1, 0]*sigmay() + r_ss[2, 0]*sigmaz())
print(rho_ss)
print(f"Gauge condition: {(rho_ss * A_sum).tr()}")