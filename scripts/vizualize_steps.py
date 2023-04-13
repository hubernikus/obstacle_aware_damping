import numpy as np
import matplotlib.pyplot as plt

from passive_control.controller import smooth_step, smooth_step_neg

# x = np.linspace(0.95,1,1000)


# delta_E_cont = 0.01 #mn.EPSILON
# lambda_1 = 200
# lambda_2 = 20
# lambda_2_p = np.empty(1000)
# for i in range(x.shape[0]):
#     lambda_2_p[i] = lambda_1*smooth_step(1-delta_E_cont, 1, x[i]) +\
#             lambda_2*smooth_step_neg(1-delta_E_cont, 1, x[i])

# fig, ax = plt.subplots()
# ax.plot(x, lambda_2_p)
# ax.set_title(r"Smooth step function to modify $\lambda_2$")
# ax.set_ylim([0,210])
# ax.set_ylabel(r'$\lambda_2 \prime $')
# ax.set_xlabel(r'$e_{1,DS}^T \, e_{2, obs}$')
# fig.show()
# plt.show()


x = np.linspace(0, 0.1, 1000)
delta_w_cont = 0.01  # mn.EPSILON
w = 0.5
w_p = np.empty(1000)
for i in range(x.shape[0]):
    w_p[i] = w * smooth_step(0, delta_w_cont, x[i])
fig, ax = plt.subplots()
ax.plot(x, w_p)
ax.set_title(r"Smooth step function to modify the weight")
# ax.set_ylim([0,210])
ax.set_ylabel("weight")
ax.set_xlabel(r"$\Vert n \Vert$")
fig.show()
plt.show()
