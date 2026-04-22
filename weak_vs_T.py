import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx

def draw_graph(G):
    K = G.shape[0]
    DG = nx.DiGraph()

    DG.add_nodes_from(range(K))

    for i in range(K):
        for j in range(K):
            if G[i, j]:
                DG.add_edge(i, j)

    pos = nx.spring_layout(DG, seed=42)

    plt.figure(figsize=(5,5))
    nx.draw(DG, pos,
            with_labels=True,
            node_size=500,
            font_size=10,
            arrows=True)

    plt.title("Feedback Graph")
    plt.show()


rng = np.random.default_rng(5)

def random_weak_graph(K):
    G = np.zeros((K, K), dtype=bool)
    for i in range(K):
        choices = [j for j in range(K) if j != i]
        j0 = rng.choice(choices)
        G[j0, i] = True
        for j in choices:
            if j != j0 and rng.random() < 0.5:
                G[j, i] = True
    return G

def exact_delta(G):
    K = G.shape[0]
    W = [i for i in range(K) if not G[i, i]]

    for r in range(1, K+1):
        for D in itertools.combinations(range(K), r):
            ok = True
            for w in W:
                if not any(G[d, w] for d in D):
                    ok = False
                    break
            if ok:
                return r
    return K

def run_exp3g(C, G, eta, gamma):
    T, K = C.shape
    w = np.ones(K)
    u = np.zeros(K)
    u[0] = 1.0
    total = 0.0

    for t in range(T):
        q = w / w.sum()
        p = (1 - gamma) * q + gamma * u
        a = rng.choice(K, p=p)
        total += C[t, a]

        P = p @ G.astype(float)
        seen = G[a]

        hat = np.zeros(K)
        hat[seen] = C[t, seen] / np.maximum(P[seen], 1e-12)

        w *= np.exp(-eta * hat)

    return total - C.sum(axis=0).min()

K = 10
T_vals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
N = 200

G = random_weak_graph(K)
delta_val = exact_delta(G)
c = rng.uniform(0, 1, size=K)

draw_graph(G)

mean_regrets = []

for T in T_vals:
    C = np.tile(c, (T, 1))
    gamma = (delta_val * np.log(K) / T) ** (1/3)
    eta = (np.log(K) / (T * np.sqrt(delta_val))) ** (2/3)

    vals = [run_exp3g(C, G, eta, gamma) for _ in range(N)]
    mean_regrets.append(np.mean(vals))

mean_regrets = np.array(mean_regrets)

theory = (delta_val**(1/3)) * (np.array(T_vals)**(2/3))
theory = theory * (mean_regrets[-1] / theory[-1])

plt.figure(figsize=(6,4))
plt.plot(T_vals, mean_regrets, 'o-', label='mean regret')
plt.plot(T_vals, theory, '--', label=r'$\delta^{1/3} T^{2/3}$ (scaled)')
plt.xlabel("T")
plt.ylabel("regret")
plt.legend()
plt.tight_layout()
plt.show()

print("delta(G) =", delta_val)