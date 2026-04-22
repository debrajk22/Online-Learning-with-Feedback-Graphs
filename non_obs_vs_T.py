import numpy as np
import matplotlib.pyplot as plt
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

def random_non_obs_graph(K):
    G = np.zeros((K, K), dtype=bool)
    bad = rng.integers(K)

    for i in range(K):
        if i == bad:
            continue

        r = rng.random()
        if r < 1/3:
            G[i, i] = True
        elif r < 2/3:
            G[:, i] = True
            G[i, i] = False
        else:
            G[:, i] = True

    G[:, bad] = False
    G[bad, bad] = False

    return G

def run_exp3g(C, G, eta, gamma):
    T, K = C.shape
    w = np.ones(K)
    u = np.ones(K) / K
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

G = random_non_obs_graph(K)
c = rng.uniform(0, 1, size=K)

draw_graph(G)

mean_regrets = []

for T in T_vals:
    C = np.tile(c, (T, 1))
    eta = 1 / np.sqrt(T)
    gamma = eta

    vals = [run_exp3g(C, G, eta, gamma) for _ in range(N)]
    mean_regrets.append(np.mean(vals))

mean_regrets = np.array(mean_regrets)

theory = np.array(T_vals)
theory = theory * (mean_regrets[-1] / theory[-1])

plt.figure(figsize=(6,4))
plt.plot(T_vals, mean_regrets, 'o-', label='mean regret')
plt.plot(T_vals, theory, '--', label=r'$T$ (scaled)')
plt.xlabel("T")
plt.ylabel("regret")
plt.legend()
plt.tight_layout()
plt.show()

