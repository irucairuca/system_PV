import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib_fontja  # 日本語ラベル用（不要ならコメントアウト）
import streamlit as st

# =========================
# Base / constants
# =========================
f0 = 50.0                                 # [Hz]
omega_b = 2.0 * math.pi * f0              # [rad/s]

# =========================
# Network: series Z1 (line etc.) + RL load Z2
# 入力は「ベース周波数時のリアクタンス」
# =========================

def reactances(dw_pu: float, X1_base: float, X2_base: float):
    """周波数偏差 Δω に応じたリアクタンス補正（一次近似）"""
    X1 = X1_base * (1.0 + dw_pu)          # X ∝ ω
    X2 = X2_base * (1.0 + dw_pu)
    return X1, X2

def currents_and_voltages(delta: float, dw: float,
                           E_mag: float,
                           R1: float, X1_base: float,
                           R2: float, X2_base: float):
    """
    E∠δ ---- Z1=R1+jX1 ---- Z2=R2+jX2 (RL load)
    I = E∠δ / (Z1 + Z2)
    V_load = I * Z2
    """
    X1, X2 = reactances(dw, X1_base, X2_base)
    Z1 = complex(R1, X1)
    Z2 = complex(R2, X2)
    Z_tot = Z1 + Z2

    E = cmath.rect(E_mag, delta)          # E∠δ
    I = E / Z_tot
    V_load = I * Z2
    return E, I, V_load

def electrical_power(delta: float, dw: float,
                     E_mag: float,
                     R1: float, X1_base: float,
                     R2: float, X2_base: float) -> float:
    """
    発電機端の実電力 P_e = Re{ E * conj(I) } を返す（pu）
    """
    E, I, _ = currents_and_voltages(delta, dw, E_mag, R1, X1_base, R2, X2_base)
    S_gen = E * np.conj(I)
    return float(S_gen.real)

# =========================
# Swing ODE (pu-unified)
# state = [delta(rad), dw(pu)]
# =========================

def rhs(state, t, P_m,
        H, D,
        E_mag, R1, X1_base, R2, X2_base):
    delta, dw = state
    P_e = electrical_power(delta, dw, E_mag, R1, X1_base, R2, X2_base)
    ddelta_dt = omega_b * dw
    ddw_dt    = (P_m - P_e - D * dw) / (2.0 * H)
    return np.array([ddelta_dt, ddw_dt], dtype=float), P_e

def rk4_step(state, t, dt, P_m,
             H, D,
             E_mag, R1, X1_base, R2, X2_base):
    k1, _ = rhs(state, t, P_m,
                H, D, E_mag, R1, X1_base, R2, X2_base)
    k2, _ = rhs(state + 0.5*dt*k1, t + 0.5*dt, P_m,
                H, D, E_mag, R1, X1_base, R2, X2_base)
    k3, _ = rhs(state + 0.5*dt*k2, t + 0.5*dt, P_m,
                H, D, E_mag, R1, X1_base, R2, X2_base)
    k4, _ = rhs(state + dt*k3,     t + dt,      P_m,
                H, D, E_mag, R1, X1_base, R2, X2_base)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# =========================
# Simulation function
# =========================

def simulate_system(
    H=4.0, D=0.0, E_mag=1.10,
    R1=0.10, X1_base=0.60,
    R2=1.00, X2_base=1.00,
    P_m0=1.00, alpha=100.0,
    t_end=5.0, dt=1.0e-4
):
    """
    単純な電力系統モデル（同期発電機-送電線-静止負荷）の過渡解析
    """
    t     = 0.0
    state = np.array([0.0, 0.0], dtype=float)  # [delta(rad), dw(pu)]
    P_m   = P_m0

    ts, fHz, del_deg = [], [], []
    Vmag, Vang_deg   = [], []
    Pe_log, Pm_log   = [], []
    I_mag            = []

    n_steps = int(t_end / dt)

    for _ in range(n_steps + 1):
        delta, dw = state

        # 現在の電気量
        E, I, V_load = currents_and_voltages(delta, dw, E_mag, R1, X1_base, R2, X2_base)
        P_e = electrical_power(delta, dw, E_mag, R1, X1_base, R2, X2_base)
        f_now = f0 * (1.0 + dw)  # [Hz]

        ts.append(t)
        fHz.append(f_now)
        del_deg.append(delta * 180.0 / math.pi)
        Vmag.append(abs(V_load))
        Vang_deg.append(cmath.phase(V_load) * 180.0 / math.pi)
        Pe_log.append(P_e)
        Pm_log.append(P_m)
        I_mag.append(abs(I))

        # governor-free operation P-control
        k_ary, _ = rhs(state, t, P_m, H, D, E_mag, R1, X1_base, R2, X2_base)
        # k_ary[1] = ddw_dt
        P_m = P_m - alpha * dt * k_ary[1]

        # RK4 で次ステップへ
        state = rk4_step(state, t, dt, P_m, H, D, E_mag, R1, X1_base, R2, X2_base)
        t += dt

    # numpy 配列に変換して返す
    return {
        "t":      np.array(ts),
        "fHz":    np.array(fHz),
        "delta":  np.array(del_deg),
        "Vmag":   np.array(Vmag),
        "Vang":   np.array(Vang_deg),
        "Pe":     np.array(Pe_log),
        "Pm":     np.array(Pm_log),
        "I_mag":  np.array(I_mag),
    }

# =========================
# Streamlit UI
# =========================

st.title("単純な電力系統の過渡解析（Streamlit版）")
st.write(
    "同期発電機 - 送電線 - 静止負荷モデルに対して、"
    "ルンゲクッタ法で過渡応答を数値計算するアプリです。"
)

st.sidebar.header("パラメータ設定")

# 発電機・動揺方程式パラメータ
H = st.sidebar.slider("慣性定数 H [s]", 0.5, 10.0, 4.0, 0.1)
D = st.sidebar.slider("ダンピング係数 D [-]", 0.0, 5.0, 0.0, 0.1)
E_mag = st.sidebar.slider("起電力 |E| [pu]", 0.5, 2.0, 1.10, 0.01)

# ネットワークパラメータ
st.sidebar.subheader("送電線・負荷パラメータ")
R1 = st.sidebar.slider("送電線抵抗 R1 [pu]", 0.0, 0.5, 0.10, 0.01)
X1_base = st.sidebar.slider("送電線リアクタンス X1_base [pu]", 0.1, 2.0, 0.60, 0.05)
R2 = st.sidebar.slider("負荷抵抗 R2 [pu]", 0.1, 5.0, 1.00, 0.1)
X2_base = st.sidebar.slider("負荷リアクタンス X2_base [pu]", 0.1, 5.0, 1.00, 0.1)

# ガバナ・シミュレーション設定
st.sidebar.subheader("ガバナ・シミュレーション条件")
P_m0 = st.sidebar.slider("初期機械入力 Pm0 [pu]", 0.1, 2.0, 1.0, 0.05)
alpha = st.sidebar.slider("ガバナ係数 α", 0.0, 300.0, 100.0, 5.0)
t_end = st.sidebar.slider("シミュレーション時間 t_end [s]", 0.5, 10.0, 5.0, 0.5)

dt = st.sidebar.select_slider(
    "時間刻み Δt [s]（小さいほど精度↑・処理重い）",
    options=[1e-3, 5e-4, 1e-4],
    value=1e-4,
    format_func=lambda x: f"{x:.0e}"
)
st.sidebar.caption("※元記事では Δt=1e-4 です。モバイルや低スペック環境では 1e-3 推奨。")

if st.button("シミュレーション実行"):
    result = simulate_system(
        H=H, D=D, E_mag=E_mag,
        R1=R1, X1_base=X1_base,
        R2=R2, X2_base=X2_base,
        P_m0=P_m0, alpha=alpha,
        t_end=t_end, dt=dt
    )

    t   = result["t"]
    fHz = result["fHz"]
    delta_deg = result["delta"]
    Vmag = result["Vmag"]
    Vang = result["Vang"]
    Pe   = result["Pe"]
    Pm   = result["Pm"]
    I_mag = result["I_mag"]

    st.success("計算が完了しました。以下のグラフで挙動を確認できます。")

    # 周波数
    fig1, ax1 = plt.subplots()
    ax1.plot(t, fHz)
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("f [Hz]")
    ax1.set_title("系統周波数の過渡応答")
    ax1.grid(True)
    st.pyplot(fig1)

    # 回転子角
    fig2, ax2 = plt.subplots()
    ax2.plot(t, delta_deg)
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("δ [deg]")
    ax2.set_title("回転子位相角 δ の応答")
    ax2.grid(True)
    st.pyplot(fig2)

    # 需要家電圧の大きさ
    fig3, ax3 = plt.subplots()
    ax3.plot(t, Vmag)
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("|V_load| [pu]")
    ax3.set_title("需要家電圧の大きさ")
    ax3.grid(True)
    st.pyplot(fig3)

    # 発電機電気的出力と機械的入力
    fig4, ax4 = plt.subplots()
    ax4.plot(t, Pe, label="P_e")
    ax4.plot(t, Pm, label="P_m")
    ax4.set_xlabel("t [s]")
    ax4.set_ylabel("Power [pu]")
    ax4.set_title("発電機の電気的出力 P_e と機械的入力 P_m")
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)

    # 送電線に流れる電流
    fig5, ax5 = plt.subplots()
    ax5.plot(t, I_mag)
    ax5.set_xlabel("t [s]")
    ax5.set_ylabel("|I| [pu]")
    ax5.set_title("送電線・負荷電流の大きさ")
    ax5.grid(True)
    st.pyplot(fig5)
else:
    st.info("左のパラメータを調整して「シミュレーション実行」ボタンを押してください。")
