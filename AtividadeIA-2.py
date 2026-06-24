import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTO FICTÍCIO
# Controlador de resfriamento de um motor elétrico industrial.
#   x1 : temperatura do motor  [0, 100] °C
#   x2 : nível de vibração     [0,  10] (u.a.)
#   z  : intensidade de resfriamento [0, 100] %
# ══════════════════════════════════════════════════════════════════════════════

# ── Função triangular (Eq. 3 do enunciado) ────────────────────────────────────


def triangular(x, a, b, c):
    """µ(x) para conjunto triangular com vértices a, b, c."""
    if b == a:
        esq = 1.0 if x >= a else 0.0
    else:
        esq = (x - a) / (b - a)
    if c == b:
        dir_ = 1.0 if x <= c else 0.0
    else:
        dir_ = (c - x) / (c - b)
    return max(min(esq, dir_), 0.0)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEFINIÇÃO DOS CONJUNTOS FUZZY
# ══════════════════════════════════════════════════════════════════════════════


# Temperatura [0, 100]
temp_sets = {
    'Baixa':  (0,   0,  40),
    'Media':  (20, 50,  80),
    'Alta':   (60, 100, 100),
}

# Vibração [0, 10]
vib_sets = {
    'Baixa':  (0, 0,  4),
    'Media':  (2, 5,  8),
    'Alta':   (6, 10, 10),
}

# Resfriamento [0, 100]
resf_sets = {
    'Fraco':    (0,   0,  40),
    'Moderado': (20, 50,  80),
    'Forte':    (60, 100, 100),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. BASE DE REGRAS  (antecedente1, antecedente2) → consequente
# ══════════════════════════════════════════════════════════════════════════════
rules = [
    ('Baixa', 'Baixa', 'Fraco'),
    ('Baixa', 'Media', 'Fraco'),
    ('Baixa', 'Alta',  'Moderado'),
    ('Media', 'Baixa', 'Moderado'),
    ('Media', 'Media', 'Moderado'),
    ('Media', 'Alta',  'Forte'),
    ('Alta',  'Baixa', 'Forte'),
    ('Alta',  'Media', 'Forte'),
    ('Alta',  'Alta',  'Forte'),
]

# ══════════════════════════════════════════════════════════════════════════════
# 3. GRÁFICOS DAS FUNÇÕES DE PERTINÊNCIA (item c)
# ══════════════════════════════════════════════════════════════════════════════


def plot_mf(ax, domain, sets_dict, title, xlabel):
    cores = ['royalblue', 'darkorange', 'forestgreen']
    for (nome, params), cor in zip(sets_dict.items(), cores):
        ys = [triangular(xi, *params) for xi in domain]
        ax.plot(domain, ys, label=nome, color=cor, lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pertinência  µ')
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plot_mf(axes[0], np.linspace(0, 100, 500), temp_sets,
        'Temperatura do Motor', 'Temperatura (°C)')
plot_mf(axes[1], np.linspace(0, 10, 500),  vib_sets,
        'Nível de Vibração', 'Vibração (u.a.)')
plot_mf(axes[2], np.linspace(0, 100, 500), resf_sets,
        'Intensidade de Resfriamento', 'Resfriamento (%)')
plt.suptitle('Funções de Pertinência — Controlador Fuzzy Mamdani',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('q2_pertinencias.png', dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 4. INFERÊNCIA MAMDANI + DEFUZZIFICAÇÃO (Eq. 3 e 4 do enunciado)
# ══════════════════════════════════════════════════════════════════════════════
Z_DOMAIN = np.linspace(0, 100, 1000)   # universo discreto da saída


def mamdani_infer(x1_val, x2_val, verbose=False):
    """
    Recebe valores nítidos de temperatura (x1) e vibração (x2).
    Retorna o valor defuzzificado z*.
    """
    # ── Passo 1: graus de pertinência das entradas ─────────────────────────
    mu_temp = {nome: triangular(x1_val, *p) for nome, p in temp_sets.items()}
    mu_vib = {nome: triangular(x2_val, *p) for nome, p in vib_sets.items()}

    if verbose:
        print(f"\n{'─'*50}")
        print(f"Entradas: Temperatura={x1_val} °C | Vibração={x2_val} u.a.")
        print("\nGraus de pertinência — Temperatura:")
        for k, v in mu_temp.items():
            print(f"  µ_{k}({x1_val}) = {v:.4f}")
        print("\nGraus de pertinência — Vibração:")
        for k, v in mu_vib.items():
            print(f"  µ_{k}({x2_val}) = {v:.4f}")

    # ── Passo 2: avaliação das regras (mínimo nos antecedentes) ───────────
    agr = np.zeros(len(Z_DOMAIN))   # superfície agregada

    if verbose:
        print("\nRegras ativadas (α > 0):")

    for (t_set, v_set, out_set) in rules:
        alpha = min(mu_temp[t_set], mu_vib[v_set])   # força da regra
        if alpha > 0:
            if verbose:
                print(f"  SE Temp={t_set} E Vib={v_set} → Resf={out_set}  "
                      f"  α = min({mu_temp[t_set]:.4f}, {mu_vib[v_set]:.4f}) = {alpha:.4f}")
            # Consequente cortado em α (Mamdani: mínimo)
            params = resf_sets[out_set]
            mu_consq = np.array([min(triangular(z, *params), alpha)
                                 for z in Z_DOMAIN])
            agr = np.maximum(agr, mu_consq)   # agregação: máximo

    # ── Passo 3: defuzzificação — centroide discreto (Eq. 4) ──────────────
    if agr.sum() == 0:
        z_star = 0.0
    else:
        z_star = np.sum(agr * Z_DOMAIN) / np.sum(agr)

    if verbose:
        print(f"\n→ Valor defuzzificado  z* = {z_star:.2f} %")

    return z_star, agr


# ══════════════════════════════════════════════════════════════════════════════
# 5. DEMONSTRAÇÃO NUMÉRICA — DOIS CASOS (item e)
# ══════════════════════════════════════════════════════════════════════════════
casos = [
    (5.0,  0.5,  "Caso 1 — Condição leve   (Temp=5°C, Vib=0.5)"),
    (95.0, 9.5,  "Caso 2 — Condição severa (Temp=95°C, Vib=9.5)"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, (t, v, titulo) in enumerate(casos):
    z_star, agr = mamdani_infer(t, v, verbose=True)

    ax = axes[idx]
    ax.fill_between(Z_DOMAIN, agr, alpha=0.35,
                    color='steelblue', label='Superfície agregada')
    ax.plot(Z_DOMAIN, agr, color='steelblue', lw=1.5)
    ax.axvline(z_star, color='red', lw=2, linestyle='--',
               label=f'z* = {z_star:.1f} %')
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('Resfriamento (%)')
    ax.set_ylabel('µ')
    ax.set_ylim(-0.05, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Defuzzificação — Centroide dos Dois Casos', fontsize=12)
plt.tight_layout()
plt.savefig('q2_defuzz.png', dpi=150)
plt.show()

print("\nArquivos salvos: q2_pertinencias.png | q2_defuzz.png")
