import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTO FICTÍCIO
# Otimização do parâmetro de calibração x de um atuador de precisão.
# Objetivo: maximizar o desempenho f(x) = x·sin(10πx) + 1.0
# x ∈ [−1, 2]  →  representação binária com 20 bits
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
X_MIN = -1.0
X_MAX = 2.0
N_BITS = 20          # precisão: (X_MAX-X_MIN) / (2^20 - 1) ≈ 2.86e-6
N_GERACOES = 100
TX_CROSS = 0.80        # taxa de cruzamento
TX_MUT = 0.01        # taxa de mutação (bit-flip)
ELITISMO = True        # guarda o melhor indivíduo intacto

# ── Função objetivo ───────────────────────────────────────────────────────────


def f(x):
    return x * np.sin(10 * np.pi * x) + 1.0

# ══════════════════════════════════════════════════════════════════════════════
# 1. CODIFICAÇÃO / DECODIFICAÇÃO BINÁRIA
# ══════════════════════════════════════════════════════════════════════════════


def decode(cromossomo):
    """Converte vetor de bits → valor real em [X_MIN, X_MAX]."""
    b = int(''.join(str(g) for g in cromossomo), 2)
    return X_MIN + b * (X_MAX - X_MIN) / (2**N_BITS - 1)


def encode_pop(N, rng):
    """Inicializa população aleatória de N cromossomos binários."""
    return rng.integers(0, 2, size=(N, N_BITS))

# ══════════════════════════════════════════════════════════════════════════════
# 2. SELEÇÃO POR ROLETA (Eq. 5 do enunciado)
# ══════════════════════════════════════════════════════════════════════════════


def selecao_roleta(pop, fitness, rng):
    """Seleciona um indivíduo usando seleção proporcional ao fitness."""
    # Desloca para garantir valores não-negativos
    f_min = fitness.min()
    f_adj = fitness - f_min + 1e-6

    probs = f_adj / f_adj.sum()              # pi = fi / Σfj
    acum = np.cumsum(probs)                 # Ci = Σ pj (j=1..i)

    r = rng.uniform(0, 1)
    for i, c in enumerate(acum):
        if r <= c:
            return pop[i].copy()
    return pop[-1].copy()

# ══════════════════════════════════════════════════════════════════════════════
# 3. CRUZAMENTO DE UM PONTO
# ══════════════════════════════════════════════════════════════════════════════


def crossover_um_ponto(pai1, pai2, rng):
    if rng.uniform() < TX_CROSS:
        ponto = rng.integers(1, N_BITS)
        filho1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
        filho2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
    else:
        filho1, filho2 = pai1.copy(), pai2.copy()
    return filho1, filho2

# ══════════════════════════════════════════════════════════════════════════════
# 4. MUTAÇÃO PARAMÉTRICA (bit-flip com probabilidade TX_MUT por gene)
# ══════════════════════════════════════════════════════════════════════════════


def mutacao(cromossomo, rng):
    for i in range(N_BITS):
        if rng.uniform() < TX_MUT:
            cromossomo[i] ^= 1      # inverte o bit
    return cromossomo

# ══════════════════════════════════════════════════════════════════════════════
# 5. ALGORITMO GENÉTICO COMPLETO (Algoritmo 3 do enunciado)
# ══════════════════════════════════════════════════════════════════════════════


def algoritmo_genetico(N, seed=SEED):
    """
    Executa o AG com população de tamanho N.
    Retorna: histórico do melhor fitness por geração,
             melhor x encontrado, melhor f(x) encontrado.
    """
    rng = np.random.default_rng(seed)
    pop = encode_pop(N, rng)

    melhor_fitness_hist = []
    melhor_x_global = None
    melhor_f_global = -np.inf

    for g in range(N_GERACOES):
        # ── Avaliação ─────────────────────────────────────────────────────
        xs = np.array([decode(c) for c in pop])
        fitness = f(xs)

        idx_melhor = np.argmax(fitness)
        if fitness[idx_melhor] > melhor_f_global:
            melhor_f_global = fitness[idx_melhor]
            melhor_x_global = xs[idx_melhor]
        melhor_fitness_hist.append(melhor_f_global)

        # ── Nova população ────────────────────────────────────────────────
        nova_pop = []

        if ELITISMO:
            nova_pop.append(pop[idx_melhor].copy())   # preserva o melhor

        while len(nova_pop) < N:
            # Seleção por roleta (Eq. 5)
            pai1 = selecao_roleta(pop, fitness, rng)
            pai2 = selecao_roleta(pop, fitness, rng)
            # Cruzamento de um ponto
            f1, f2 = crossover_um_ponto(pai1, pai2, rng)
            # Mutação paramétrica (bit-flip)
            f1 = mutacao(f1, rng)
            f2 = mutacao(f2, rng)
            nova_pop.append(f1)
            if len(nova_pop) < N:
                nova_pop.append(f2)

        pop = np.array(nova_pop)

    return melhor_fitness_hist, melhor_x_global, melhor_f_global


# ══════════════════════════════════════════════════════════════════════════════
# 6. EXECUÇÃO COM N ∈ {10, 30, 100} (item d)
# ══════════════════════════════════════════════════════════════════════════════
populacoes = [10, 30, 100]
cores_ag = {10: 'crimson', 30: 'darkorange', 100: 'steelblue'}
resultados = {}

print("Executando Algoritmo Genético...\n")
for N in populacoes:
    hist, x_opt, f_opt = algoritmo_genetico(N, seed=SEED)
    resultados[N] = dict(hist=hist, x=x_opt, f=f_opt)
    print(f"  N={N:>3}  →  x* = {x_opt:.6f}   f(x*) = {f_opt:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. GRÁFICO — Melhor Fitness × Gerações (item e)
# ══════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(9, 5))
for N in populacoes:
    plt.plot(resultados[N]['hist'],
             label=f'N = {N}', color=cores_ag[N], lw=2)
plt.axhline(2.85, color='black', linestyle='--', lw=1.2,
            label='Ótimo teórico ≈ 2.85')
plt.xlabel('Geração')
plt.ylabel('Melhor f(x)')
plt.title('Melhor Fitness por Geração — Comparação de Tamanhos Populacionais')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q3_fitness_geracoes.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. GRÁFICO — Função objetivo e melhor solução encontrada (extra, ilustrativo)
# ══════════════════════════════════════════════════════════════════════════════
x_plot = np.linspace(X_MIN, X_MAX, 2000)
y_plot = f(x_plot)

plt.figure(figsize=(9, 4))
plt.plot(x_plot, y_plot, 'k-', lw=1.5, label='f(x)')
for N in populacoes:
    xo = resultados[N]['x']
    plt.scatter(xo, f(xo), s=80, zorder=5,
                color=cores_ag[N], label=f'N={N}: x*={xo:.3f}')
plt.xlabel('x  (parâmetro de calibração do atuador)')
plt.ylabel('f(x)')
plt.title('Função Objetivo e Melhores Soluções Encontradas')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q3_funcao_objetivo.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9. TABELA RESUMO
# ══════════════════════════════════════════════════════════════════════════════
print("\n{'─'*50}")
print(f"{'N':>6} | {'x*':>12} | {'f(x*)':>10}")
print('─'*34)
for N in populacoes:
    r = resultados[N]
    print(f"{N:>6} | {r['x']:>12.6f} | {r['f']:>10.6f}")

print("\nArquivos salvos: q3_fitness_geracoes.png | q3_funcao_objetivo.png")
