import numpy as np
import matplotlib.pyplot as plt
import cmath
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# rho - density
# c - phase velocity
# P1 = P-wave incidental angle, S1 = SW-wave reflection angle
# P2 = P-wave trasmission angle, S2 = SW-wave transmission angle

# from the paper, water kg/m3, aluminum kg/m3, water m/s ...
rho_1 = 1000
rho_2 = 2000
c_P1 = 1500
c_S1 = 0.001
c_P2 = 6420
c_S2 = 3040

def snell(th_P1, c_P1, c_P2):
	th_P2 = cmath.asin(np.sin(th_P1)*c_P2/c_P1)
	# asin returns the arc sine value of a complex number.
	return th_P2


def inv_matrix(th_P1, th_S1, th_P2, th_S2):
	mat = np.array(([np.sin(th_P1)/(rho_1*c_P1) , np.cos(th_S1)/(rho_1*c_S1) , -np.sin(th_P2)/(rho_2*c_P2) , np.sin(th_S2)/(rho_2*c_S2)],
					[np.cos(th_P1)/(rho_1*c_P1) , -np.sin(th_S1)/(rho_1*c_P1) , np.cos(th_P2)/(rho_2*c_P2) , np.sin(th_S2)/(rho_2*c_S2)],
					[-np.cos(2*th_S1)			, np.sin(2*th_S1)			  , np.cos(2*th_S2)			   , np.sin(2*th_S2)		   ],
					[np.sin(2*th_P1)/(c_P1**2/c_S1**2) , np.cos(2*th_S1)	  , np.sin(2*th_P2)/(c_P2**2/c_S2**2) , -np.cos(2*th_S2)   ]))
	return np.linalg.inv(mat)

def other_matrix(th_P1, th_S1):
	omat = np.array([-np.sin(th_P1)/(rho_1*c_P1) , np.cos(th_P1)/(rho_1*c_P1) , np.cos(2*th_S1) , np.sin(2*th_P1)/(c_P1**2/c_S1**2)])
	return omat.T # transform from (x,1) to (1,x)

th = np.linspace(0,30,1000)


R_P_Pot = np.zeros(len(th))
R_S_Pot = np.zeros_like(R_P_Pot)
T_P_Pot = np.zeros_like(R_P_Pot)
T_S_Pot = np.zeros_like(R_P_Pot)

for i, th_P1 in enumerate(th):
	# reflected P-wave, reflected S-wave, transmitted P-wave ... coeffs.
	th_P1 = np.deg2rad(th_P1)
	th_S1 = snell(th_P1, c_P1, c_S1)
	th_P2 = snell(th_P1, c_P1, c_P2)
	th_S2 = snell(th_P1, c_P1, c_S2)

	R_P, R_S, T_P, T_S = inv_matrix(th_P1, th_S1, th_P2, th_S2) @ other_matrix(th_P1, th_S1)

	Z_P1 = (c_P1*rho_1)/np.cos(th_P1)
	Z_P2 = (c_P2*rho_2)/np.cos(th_P2)
	Z_S1 = (c_P1*rho_1)/np.cos(th_S1)
	Z_S2 = (c_P2*rho_2)/np.cos(th_S2)

	# Changed the sign on R_P_Pot and T_S_Pot (according to the paper it should be minus)
	R_P_Pot[i] = R_P*np.conj(R_P)
	R_S_Pot[i] = -R_S*np.conj(R_S)*np.real(1/np.conj(Z_S1))/np.real(1/np.conj(Z_P1))
	T_P_Pot[i] = T_P*np.conj(T_P)*np.real(1/np.conj(Z_P2))/np.real(1/np.conj(Z_P1))
	T_S_Pot[i] = T_S*np.conj(T_S)*np.real(1/np.conj(Z_S2))/np.real(1/np.conj(Z_P1))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot()
ax.set_title('Interface results', fontsize=18)
ax.plot(th, R_P_Pot, label = 'R P Pot')
ax.plot(th, R_S_Pot, label = 'R S Pot')
ax.plot(th, T_P_Pot, label = 'T P Pot')
ax.plot(th, T_S_Pot, label = 'T S Pot')
ax.set_xlabel('Angle of incidence (degree)', fontsize=14)
ax.set_ylabel('Power transmission and reflection coefficients', fontsize=14)
ax.legend(fontsize=20)

plt.show()