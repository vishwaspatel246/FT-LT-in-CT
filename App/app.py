# app.py
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import io, base64
import re

#Page Setup
st.set_page_config(page_title="CT FT vs LT Visualizer", layout="wide")
st.title("Continuous-Time: Fourier vs Laplace Transform Visualizer")
st.markdown(
    "Select or input signals in SymPy syntax (e.g., exp(-2*t)*Heaviside(t)). "
    "This app shows time-domain plots, animations, symbolic transforms, numeric FFT, "
    "Laplace ROC, and |X(s)| vs σ visualization."
)

# Symbols
t, s, w = sp.symbols('t s w', real=True)


def plain_str(expr):
    try:
        return str(sp.simplify(expr))
    except Exception:
        return str(expr)

def safe_lambdify(expr, var):
    """Lambdify with Heaviside mapped to numpy (0.5 at 0)."""
    return sp.lambdify(var, expr, modules=['numpy', {'Heaviside': lambda x: np.where(x>0,1.0,np.where(x==0,0.5,0.0))}])

def preprocess_expr(expr_str):
    expr_str = re.sub(r'(\d)(sin|cos|exp|Heaviside|DiracDelta)', r'\1*\2', expr_str)
    expr_str = re.sub(r'pi(\w)', r'pi*\1', expr_str)
    return expr_str

def animation_to_gif_bytes(ani: FuncAnimation, fps=20):
    buf = io.BytesIO()
    writer = PillowWriter(fps=fps)
    ani.save(buf, writer=writer)
    buf.seek(0)
    return buf.read()

# Sidebar: signal selection 
predefined_signals = {
    "Decaying exp": "exp(-2*t)*Heaviside(t)",
    "Growing exp": "exp(2*t)*Heaviside(t)",
    "Sine": "(1)*sin(2*pi*t)*Heaviside(t)",
    "Cosine": "(1)*cos(2*pi*t)*Heaviside(t)",
    "Dirac Delta": "DiracDelta(t)"
}

st.sidebar.header("Select or input signal")
signal_choice = st.sidebar.selectbox("Select a predefined signal", list(predefined_signals.keys()) + ["Custom"])
if signal_choice != "Custom":
    custom_expr = predefined_signals[signal_choice]
else:
    custom_expr = st.sidebar.text_area("Enter custom signal (SymPy syntax)", value="exp(-2*t)*Heaviside(t)", height=110)

st.sidebar.markdown("---")
st.sidebar.subheader("Time sampling / numeric settings")
t0_sample = float(st.sidebar.number_input("Start time (t0)", value=-0.2, step=0.1))
t1_sample = float(st.sidebar.number_input("End time (t1)", value=5.0, step=0.1))
pts = int(st.sidebar.slider("Time points (sampling)", 200, 8000, 1600, step=100))

st.sidebar.markdown("---")
st.sidebar.subheader("Transforms / plots options")
show_laplace_symbolic = st.sidebar.checkbox("Show Laplace (symbolic)", value=True)
show_fourier_symbolic = st.sidebar.checkbox("Show Fourier (symbolic)", value=False)
show_fft_numeric = st.sidebar.checkbox("Show Fourier (numeric FFT)", value=True)
show_Xs_sigma = st.sidebar.checkbox("|X(s)| vs σ plot (for fixed ω)", value=True)
omega_for_sigma = float(st.sidebar.number_input("ω for |X(s)| plot (rad/s)", value=0.0, step=0.1))
sigma_range = st.sidebar.slider("σ range for |X(s)| plot", -10.0, 10.0, (-5.0, 5.0), step=0.5)

# Build symbolic expression
st.header("Signal (symbolic & numeric)")
locals_map = {"t": t, "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta,
              "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "pi": sp.pi}

try:
    x_sym = sp.sympify(preprocess_expr(custom_expr), locals=locals_map)
except Exception as e:
    st.error(f"Could not parse expression. SymPy error: {e}")
    st.stop()

st.subheader("Symbolic x(t) (plain)")
st.write(plain_str(x_sym))

t_vals = np.linspace(t0_sample, t1_sample, pts)

#Numeric evaluation 
x_vals = None
if x_sym.has(sp.DiracDelta):
    st.info("Time-domain numeric plot skipped: DiracDelta cannot be numerically evaluated.")
else:
    try:
        f_num = safe_lambdify(x_sym, t)
        x_vals = np.array(f_num(t_vals), dtype=float)
    except Exception as e:
        st.error(f"Numeric evaluation failed: {e}")

# Time-domain plot
if x_vals is not None:
    fig_t, ax_t = plt.subplots(figsize=(7,3))
    ax_t.plot(t_vals, x_vals)
    ax_t.set_xlabel("t"); ax_t.set_ylabel("x(t)"); ax_t.grid(True)
    ax_t.set_title("Time-domain signal x(t)")
    st.pyplot(fig_t)

# Animation

# if x_vals is not None:
#     st.subheader("Animated build-up (time)")
#     fig_anim, ax_anim = plt.subplots(figsize=(7,3))
#     line, = ax_anim.plot([], [], lw=2)
#     ax_anim.set_xlim(t_vals[0], t_vals[-1])
#     x_min = np.nanmin(x_vals) if np.isfinite(np.nanmin(x_vals)) else -1.0
#     x_max = np.nanmax(x_vals) if np.isfinite(np.nanmax(x_vals)) else 1.0
#     ax_anim.set_ylim(x_min - 0.08*abs(x_min+1e-12), x_max + 0.08*abs(x_max+1e-12))
#     ax_anim.grid(True)

#     def update_ct(frame):
#         line.set_data(t_vals[:frame], x_vals[:frame])
#         return (line,)

#     ani_ct = FuncAnimation(fig_anim, update_ct, frames=len(t_vals), interval=15, blit=False)
#     #ani.save("Test.gif", writer='pillow')
#     try:
#         buf_gif = io.BytesIO()
#         writer = PillowWriter(fps=15)
#         ani_ct.save(buf_gif, writer=writer)
#         buf_gif.seek(0)
#         b64 = base64.b64encode(buf_gif.read()).decode('utf-8')
#         st.markdown(f'<img src="data:image/gif;base64,{b64}" alt="ct_anim">', unsafe_allow_html=True)
        
#     except Exception:
#         st.info("Animation could not be rendered here; try running locally.")

# Fourier & Laplace existence 
st.header("Transforms & Existence checks")
fourier_exists = False
fourier_note = ""

if x_sym.has(sp.DiracDelta):
    fourier_exists = True
    fourier_note = "Dirac delta δ(t): FT exists in distribution sense, X(ω)=1."

elif x_sym.has(sp.sin) or x_sym.has(sp.cos):
    fourier_exists = True
    fourier_note = "Pure sinusoid: FT exists in distribution sense (impulses at ±ω₀)."

else:
    # Absolute integrability test
    try:
        abs_int = sp.integrate(sp.Abs(x_sym), (t, -sp.oo, sp.oo))
        if abs_int.is_finite:
            fourier_exists = True
            fourier_note = f"∫|x(t)| dt = {abs_int} (finite) ⇒ FT exists."
        else:
            fourier_exists = False
            fourier_note = "∫|x(t)| dt diverges ⇒ FT does NOT exist."
    except Exception:
        # Laplace ROC fallback
        try:
            res = sp.laplace_transform(x_sym, t, s, noconds=False)
            if isinstance(res, tuple) and len(res) >= 2:
                _, cond = res[0], res[1]
            else:
                cond = None
            if cond is not None and cond != sp.S.false:
                if cond.subs(s, sp.I*w):
                    fourier_exists = True
                    fourier_note = "ROC includes jω-axis ⇒ FT exists."
                else:
                    fourier_exists = False
                    fourier_note = "ROC excludes jω-axis ⇒ FT does NOT exist."
        except Exception as e:
            fourier_note = f"Could not decide FT existence (error: {e})"




# Symbolic Fourier (only if FT exists)
sympy_fourier = None
if show_fourier_symbolic and fourier_exists:
    try:
        sympy_fourier = sp.fourier_transform(x_sym, t, w)
        fourier_note += " SymPy returned symbolic FT."
    except Exception as e:
        sympy_fourier = None
        fourier_note += f" SymPy could not compute symbolic FT ({e})."
elif show_fourier_symbolic and not fourier_exists:
    fourier_note += " (Symbolic FT skipped: Fourier Transform does not exist.)"


# Laplace symbolic
laplace_exists = False
laplace_note = ""
laplace_expr_plain = None
laplace_cond_plain = None
if show_laplace_symbolic:
    try:
        res = sp.laplace_transform(x_sym, t, s, noconds=False)
        if isinstance(res, tuple):
            if len(res) == 3:
                Xs_expr, cond, _ = res
            elif len(res) == 2:
                Xs_expr, cond = res
            else:
                Xs_expr, cond = res[0], None
        else:
            Xs_expr, cond = res, None
        laplace_expr_plain = plain_str(Xs_expr)
        laplace_cond_plain = plain_str(cond) if cond else None
        if cond is not sp.S.false:
            laplace_exists = True
            laplace_note = "Laplace transform exists (ROC returned by SymPy)."
        else:
            laplace_note = "Laplace may not exist (ROC false)."
    except Exception as e:
        laplace_note = f"Laplace symbolic failed: {e}"

# Results panel 
st.subheader("Results summary")
col1, col2 = st.columns(2)

with col1:
    st.write("*Fourier Transform*")
    st.write("Exists" if fourier_exists else "Does NOT exist")
    st.write(fourier_note)
    if sympy_fourier is not None:
        st.write("Symbolic FT (ω domain):")
        st.latex(sp.latex(sympy_fourier))

with col2:
    st.write("*Laplace Transform*")
    st.write("Exists" if laplace_exists else "Does NOT exist")
    st.write(laplace_note)
    if laplace_expr_plain is not None:
        st.write("Symbolic LT (s domain):")
        st.latex(sp.latex(Xs_expr))
        if laplace_cond_plain:
            st.write(f"ROC / Condition: {laplace_cond_plain}")


# Fourier numeric spectrum 
if show_fft_numeric:   # only if user enabled FFT
    if fourier_exists:
        if x_sym.has(sp.DiracDelta):
            # Special case: δ(t)
            st.subheader("Fourier Transform of δ(t)")
            omega_vals = np.linspace(-20, 20, 400)
            X_vals = np.ones_like(omega_vals)  # FT of δ(t) = 1
            fig_delta, ax_delta = plt.subplots(figsize=(7,3))
            ax_delta.plot(omega_vals, X_vals)
            ax_delta.set_xlabel("Frequency (ω)")
            ax_delta.set_ylabel("|X(ω)|")
            ax_delta.set_title("FT of δ(t): X(ω) = 1")
            ax_delta.grid(True)
            st.pyplot(fig_delta)

        elif x_sym.has(sp.sin) or x_sym.has(sp.cos):
            # Special case: sinusoid
            st.subheader("Fourier Transform of sinusoid")
            st.info("Pure sinusoids are not absolutely integrable, "
                    "but their FT exists in the distribution sense: "
                    "impulses at ±ω₀.")
            try:
                omega0 = float((x_sym.atoms(sp.sin) | x_sym.atoms(sp.cos)).pop().args[0])
            except Exception:
                omega0 = 1.0
            fig_imp, ax_imp = plt.subplots(figsize=(7,3))
            ax_imp.axvline(omega0, color='r', linestyle='--', ymax=0.8)
            ax_imp.axvline(-omega0, color='r', linestyle='--', ymax=0.8)
            ax_imp.set_xlabel("Frequency (ω)")
            ax_imp.set_ylabel("Impulse strength")
            ax_imp.set_title("FT of sinusoid: impulses at ±ω₀")
            ax_imp.grid(True)
            st.pyplot(fig_imp)

        elif x_vals is not None:
            # Normal case: absolutely integrable signals
            st.subheader("Fourier Numeric Spectrum (FFT)")
            dt = (t1_sample - t0_sample) / pts
            freq_vals = np.fft.fftshift(np.fft.fft(x_vals))
            freq_axis = np.fft.fftshift(np.fft.fftfreq(len(x_vals), dt))
            fig_fft, ax_fft = plt.subplots(figsize=(7,3))
            ax_fft.plot(freq_axis, np.abs(freq_vals))
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("|X(f)|")
            ax_fft.set_title("Fourier Magnitude Spectrum")
            ax_fft.grid(True)
            st.pyplot(fig_fft)

    else:
        
        st.info("Fourier Transform does not exist for this signal (no FFT shown).")



# Laplace |X(s)| vs sigma
if laplace_exists and show_Xs_sigma:
    st.subheader("|X(s)| vs σ for ω={}".format(omega_for_sigma))
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], 400)
    Xs_abs_vals = []
    try:
        f_Xs = sp.lambdify(s, Xs_expr, "numpy")
        for sigma_val in sigma_vals:
            Xs_abs_vals.append(abs(f_Xs(sigma_val + 1j*omega_for_sigma)))
        fig_Xs, ax_Xs = plt.subplots(figsize=(7,3))
        ax_Xs.plot(sigma_vals, Xs_abs_vals)
        ax_Xs.set_xlabel("σ"); ax_Xs.set_ylabel("|X(s)|"); ax_Xs.grid(True)
        st.pyplot(fig_Xs)
    except Exception as e:

        st.info(f"|X(s)| plot failed: {e}")
