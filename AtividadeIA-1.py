import numpy as np
import matplotlib.pyplot as plt

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GERAÇÃO DO CONJUNTO DE DADOS
# Contexto: robô móvel em ambiente 2D.
#   x1 = coordenada horizontal normalizada
#   x2 = coordenada vertical  normalizada
#   d  = 1 → zona segura  (x1²+x2² ≤ 0.5)
#   d  = 0 → zona de risco
# ══════════════════════════════════════════════════════════════════════════════
N_TOTAL = 1400                                  # 1000 treino + 400 teste
x1 = rng.uniform(-1, 1, N_TOTAL)
x2 = rng.uniform(-1, 1, N_TOTAL)
d = (x1**2 + x2**2 <= 0.5).astype(float)

X = np.column_stack([x1, x2])                  # (N, 2)
y = d.reshape(-1, 1)                           # (N, 1)

# Split 70 / 30
split = 1000
X_train, y_train = X[:split], y[:split]
X_test,  y_test = X[split:], y[split:]

# ══════════════════════════════════════════════════════════════════════════════
# 2. FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def sigmoid_prime(y):          # recebe a SAÍDA já calculada, não v
    return y * (1.0 - y)


def init_weights(n_in, n_hid, n_out, rng):
    W1 = rng.uniform(-0.5, 0.5, (n_hid, n_in))
    b1 = rng.uniform(-0.5, 0.5, (n_hid,))
    W2 = rng.uniform(-0.5, 0.5, (n_out, n_hid))
    b2 = rng.uniform(-0.5, 0.5, (n_out,))
    return W1, b1, W2, b2

# ══════════════════════════════════════════════════════════════════════════════
# 3. TREINAMENTO — BACKPROPAGATION (modo online, amostra por amostra)
# ══════════════════════════════════════════════════════════════════════════════


def train_mlp(X_train, y_train, eta,
              n_hid=6, max_epochs=3000, tol=1e-2,
              seed=42):
    """
    Rede: 2 → n_hid → 1
    Ativação: sigmoide em todas as camadas
    Retorna: histórico de MSE por época, pesos finais
    """
    rng_local = np.random.default_rng(seed)
    W1, b1, W2, b2 = init_weights(2, n_hid, 1, rng_local)
    mse_history = []

    for epoch in range(1, max_epochs + 1):
        errors = []

        # Percorre as amostras em ordem aleatória (modo online)
        idx = rng_local.permutation(len(X_train))
        for i in idx:
            x = X_train[i]          # (2,)
            t = y_train[i]          # (1,)

            # ── Propagação direta ──────────────────────────────────────────
            v1 = W1 @ x + b1        # (n_hid,)
            y1 = sigmoid(v1)        # (n_hid,)

            v2 = W2 @ y1 + b2       # (1,)
            y2 = sigmoid(v2)        # (1,)

            # ── Erro ──────────────────────────────────────────────────────
            e = t - y2              # (1,)
            errors.append(e[0]**2)

            # ── Retropropagação ───────────────────────────────────────────
            # Camada de saída
            delta2 = e * sigmoid_prime(y2)          # (1,)

            # Camada oculta
            delta1 = (W2.T @ delta2) * sigmoid_prime(y1)   # (n_hid,)

            # ── Atualização dos pesos ─────────────────────────────────────
            W2 += eta * np.outer(delta2, y1)
            b2 += eta * delta2

            W1 += eta * np.outer(delta1, x)
            b1 += eta * delta1

        mse = np.mean(errors)
        mse_history.append(mse)

        if mse < tol:
            print(f"  [η={eta}] Convergiu na época {epoch} | MSE={mse:.6f}")
            break
    else:
        print(f"  [η={eta}] Máx. épocas atingido | MSE final={mse:.6f}")

    return mse_history, W1, b1, W2, b2

# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDIÇÃO
# ══════════════════════════════════════════════════════════════════════════════


def predict(X, W1, b1, W2, b2, threshold=0.5):
    v1 = sigmoid(X @ W1.T + b1)
    v2 = sigmoid(v1 @ W2.T + b2)
    return (v2[:, 0] >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXECUÇÃO COM OS TRÊS VALORES DE η
# ══════════════════════════════════════════════════════════════════════════════
etas = [0.01, 0.1, 0.5]
results = {}

print("Treinando a MLP...")
for eta in etas:
    mse_hist, W1, b1, W2, b2 = train_mlp(X_train, y_train, eta, seed=SEED)
    preds = predict(X_test, W1, b1, W2, b2)
    acc = np.mean(preds == y_test[:, 0])
    results[eta] = dict(mse=mse_hist, W1=W1, b1=b1, W2=W2, b2=b2,
                        preds=preds, acc=acc)
    print(f"  [η={eta}] Acurácia no teste: {acc*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. GRÁFICO 1 — MSE × Épocas (item e)
# ══════════════════════════════════════════════════════════════════════════════
plt.figure(figsize=(8, 4))
cores = {0.01: 'royalblue', 0.1: 'darkorange', 0.5: 'forestgreen'}
for eta in etas:
    plt.plot(results[eta]['mse'], label=f'η = {eta}', color=cores[eta], lw=1.5)
plt.axhline(0.01, color='red', linestyle='--',
            lw=1, label='Critério MSE = 0.01')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('MSE × Épocas — Comparação dos valores de η')
plt.legend()
plt.tight_layout()
plt.savefig('q1_mse_epocas.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 7. GRÁFICO 2 — Dispersão 2D no conjunto de teste (item f)
# ══════════════════════════════════════════════════════════════════════════════
# Usa o modelo treinado com η = 0.1 (melhor desempenho esperado)
eta_best = 0.1
preds_best = results[eta_best]['preds']
real = y_test[:, 0].astype(int)

# Marca: círculo = acerto, ×= erro
acerto = preds_best == real

fig, ax = plt.subplots(figsize=(6, 6))
# Classe real: cor de fundo
ax.scatter(X_test[real == 1, 0], X_test[real == 1, 1],
           c='skyblue', edgecolors='none', s=25, label='Real: zona segura (1)')
ax.scatter(X_test[real == 0, 0], X_test[real == 0, 1],
           c='salmon', edgecolors='none', s=25, label='Real: zona de risco (0)')
# Erros: marcador ×
ax.scatter(X_test[~acerto, 0], X_test[~acerto, 1],
           c='black', marker='x', s=50, linewidths=1.2, label='Erro de classificação')
# Fronteira teórica
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(np.sqrt(0.5)*np.cos(theta), np.sqrt(0.5)*np.sin(theta),
        'k--', lw=1.5, label='Fronteira real (x₁²+x₂²=0.5)')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x₁  (coord. horizontal)')
ax.set_ylabel('x₂  (coord. vertical)')
ax.set_title(f'Dispersão 2D — Conjunto de Teste (η = {eta_best})')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('q1_dispersao.png', dpi=150)
plt.show()

print("\nArquivos salvos: q1_mse_epocas.png | q1_dispersao.png")
