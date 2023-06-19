import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from func.data_gen import surface_1
from vector_spaces.func import sim_meas as sim
import matplotlib.patheffects as pe


x = np.linspace(-4, 4, 500)
y = np.linspace(-4, 4, 500)
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(0, len(x)):
    for j in range(0, len(y)):
        Z[i, j] = surface_1(X[0, i], Y[j, 0])

origin = 'lower'

C1 = 1.4
C2 = 0.2

k=50
eps=1
A = np.array([[-1.8-C1, -3.25+1.8], # D
              [-1.8-C2, -2+1.8]])
B = np.array([[0-C1, -1.4-0, -1.45--1.4], # F
              [-1.5-C2, -3--1.5, -3.5--3]])
C = np.array([[2.45-C1], # A
              [-0.25-C2]])
D = np.array([[-1.2-C1, -1.4--1.2, -1.45--1.4], # E
              [-2-C2, -3--2, -3.5--3]])
A_hat = sim.part(A, k)
B_hat = sim.part(B, k)
C_hat = sim.part(C, k)
D_hat = sim.part(D, k)

w = np.ones(k) * np.linspace(1,0,k)

print(sim.S(A_hat, B_hat, w, eps)[0])
print(sim.S(B_hat, A_hat, w, eps)[0])
print("Branching dist {}, number of vectors is {}".format(eps, k))
print('--------------------------------------')
print('Similarity Score D: {}'.format((sim.S(A_hat, B_hat, w, eps)[0] +
                                      sim.S(A_hat, C_hat, w, eps)[0] +
                                      sim.S(A_hat, D_hat, w, eps)[0]) / 3))
print('D-F branching point at {}% along journey'.format((sim.S(A_hat, B_hat, w, eps)[1]) / k*100))
print('D-A branching point at {}% along journey'.format((sim.S(A_hat, C_hat, w, eps)[1]) / k*100))
print('D-E branching point at {}% along journey'.format((sim.S(A_hat, D_hat, w, eps)[1]) / k*100))
print('--------------------------------------')
print('Similarity Score F: {}'.format((sim.S(B_hat, A_hat, w, eps)[0] +
                                      sim.S(B_hat, C_hat, w, eps)[0] +
                                      sim.S(B_hat, D_hat, w, eps)[0]) / 3))
print('F-D branching point at {}% along journey'.format((sim.S(B_hat, A_hat, w, eps)[1]) / k*100))
print('F-A branching point at {}% along journey'.format((sim.S(B_hat, C_hat, w, eps)[1]) / k*100))
print('F-E branching point at {}% along journey'.format((sim.S(B_hat, D_hat, w, eps)[1]) / k*100))
print('--------------------------------------')
print('Similarity Score A: {}'.format((sim.S(C_hat, A_hat, w, eps)[0] +
                                      sim.S(C_hat, B_hat, w, eps)[0] +
                                      sim.S(C_hat, C_hat, w, eps)[0]) / 3))
print('A-F branching point at {}% along journey'.format((sim.S(C_hat, A_hat, w, eps)[1]) / k*100))
print('A-D branching point at {}% along journey'.format((sim.S(C_hat, B_hat, w, eps)[1]) / k*100))
print('A-E branching point at {}% along journey'.format((sim.S(C_hat, D_hat, w, eps)[1]) / k*100))
print('--------------------------------------')
print('Similarity Score E: {}'.format((sim.S(D_hat, A_hat, w, eps)[0] +
                                      sim.S(D_hat, B_hat, w, eps)[0] +
                                      sim.S(D_hat, C_hat, w, eps)[0]) / 3))
print('E-D branching point at {}% along journey'.format((sim.S(D_hat, A_hat, w, eps)[1]) / k*100))
print('E-F branching point at {}% along journey'.format((sim.S(D_hat, B_hat, w, eps)[1]) / k*100))
print('E-A branching point at {}% along journey'.format((sim.S(D_hat, C_hat, w, eps)[1]) / k*100))


fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X, Y, Z, 5, cmap=mpl.colormaps['seismic'], origin=origin, alpha=0.6, antialiased=True)
ax2.contour(X, Y, Z, levels=[0.7], linestyles='dashed', colors='red', linewidths=1.5, alpha=1)
# CS2 = ax2.contour(CS, levels=[0], colors='k', origin=origin, linewidths=0.5, linestyles='dashed', alpha=0.5)
ax2.xaxis.set_tick_params(labelbottom=False)
ax2.yaxis.set_tick_params(labelleft=False)

# Hide X and Y axes tick marks
ax2.set_xticks([])
ax2.set_yticks([])
ax2.axis('off')
font_size = 25
weight_1 = 4
weight_2 = 3
start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, A.shape[1]):
    start1 = end1
    start2 = end2
    end1 += A[0, i]
    end2 += A[1, i]
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_1, color='black', alpha=.5))
    if i == A.shape[1] - 1:
        plt.text(-3.35, -2, 'D', va='center', ha='center',
                 rotation='horizontal', fontsize=font_size, color='green', alpha=.7,
                 path_effects=[pe.withStroke(linewidth=4, foreground="black")])

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, A_hat.shape[1]):
    start1 = end1
    start2 = end2
    end1 = A_hat[0, i]+C1
    end2 = A_hat[1, i]+C2
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_2, color='green', alpha=.8))

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, B.shape[1]):
    start1 = end1
    start2 = end2
    end1 += B[0, i]
    end2 += B[1, i]
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_1, color='black', alpha=.5))
    if i == B.shape[1] - 1:
        plt.text(-1.45, -3.7, 'F', va='center', ha='center',
                 rotation='horizontal', fontsize=font_size, color='red', alpha=.7,
                 path_effects=[pe.withStroke(linewidth=4, foreground="black")])


start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, B_hat.shape[1]):
    start1 = end1
    start2 = end2
    end1 = B_hat[0, i] + C1
    end2 = B_hat[1, i] + C2
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_2, color='red', alpha=.8))

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, C.shape[1]):
    start1 = end1
    start2 = end2
    end1 += C[0, i]
    end2 += C[1, i]
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_1, color='black', alpha=.5))
    if i == C.shape[1] - 1:
        plt.text(2.45 + 0.15, -0.25, 'A', va='center', ha='center',
                 rotation='horizontal', fontsize=font_size, color='blue', alpha=.7,
                 path_effects=[pe.withStroke(linewidth=4, foreground="black")])

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, C_hat.shape[1]):
    start1 = end1
    start2 = end2
    end1 = C_hat[0, i]+C1
    end2 = C_hat[1, i]+C2
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_2, color='blue', alpha=.8))

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, D.shape[1]):
    start1 = end1
    start2 = end2
    end1 += D[0, i]
    end2 += D[1, i]
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_1, color='black', alpha=.5))
    if i == D.shape[1] - 1:
        plt.text(-1.2 - 0.45, -1.75 - 0.7, r'E$_2$', va='center', ha='center',
                 rotation='horizontal', fontsize=font_size, color='yellow', alpha=.7,
                 path_effects=[pe.withStroke(linewidth=4, foreground="black")])

start1 = 0
start2 = 0
end1 = C1
end2 = C2
for i in range(0, D_hat.shape[1]):
    start1 = end1
    start2 = end2
    end1 = D_hat[0, i]+C1
    end2 = D_hat[1, i]+C2
    plt.annotate("", xy=(end1, end2), xytext=(start1, start2),
                 arrowprops=dict(arrowstyle="->", lw=weight_2, color='yellow', alpha=.8))


circle = plt.Circle((-2, 2), eps, color='black', linestyle='--', alpha=0.2)
plt.hlines(2, -2, -2+eps, colors='black', linestyle='-')
plt.text(-2 + eps+0.1, 2, r'$\epsilon$', va='center', ha='center',
                 rotation='horizontal', fontsize=15, color='black', alpha=.7)
ax2.add_patch(circle)
ax2.plot([C1], [C2], marker="X", markersize=8, markeredgecolor="k", markerfacecolor="white", alpha=1, zorder=50)
plt.savefig('plots_no_distance_weight/CF_paths_kis{}_epsis{}.pdf'.format(k, eps), bbox_inches='tight')
plt.show()
