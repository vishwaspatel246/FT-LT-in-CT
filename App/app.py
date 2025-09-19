# app.py
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io, base64, re

# Page Setup
st.set_page_config(page_title="CT FT vs LT Visualizer", layout="wide")
st.title("Continuous-Time: Fourier vs Laplace Transform Visualizer")
st.markdown(
    "Select or input signals in SymPy syntax (e.g., exp(-2*t)*Heaviside(t)). "
    "This app shows time-domain plots, symbolic transforms, numeric FFT, "
    "Laplace ROC parsing, and |X(s)| vs σ visualization."
)

# Symbols
t, s, w = sp.symbols('t s w', real=True)

# Helpers
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

def pretty_roc_piece(cond):
    """Return a readable string for a SymPy ROC condition like Re(s) > 2."""
    if cond is None:
        return None
    cs = str(cond)
    cs = cs.replace("< oo", "< ∞").replace("-oo <", "-∞ <")
    cs = re.sub(r'\s+', ' ', cs).strip()
    return cs
from sympy import Piecewise

def is_compact_support(expr, t):
    # crude check: signal becomes 0 outside finite interval
    return expr.has(sp.Heaviside) and (
        "Heaviside" in str(expr) and "-" in str(expr)
    )


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

# Numeric evaluation 
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

# -------------------------
# Robust Laplace handling
# -------------------------
def parse_roc_strings(rocs_raw):
    """Return (lower_bounds_list, upper_bounds_list, pretty_list)."""
    lower_bounds = []
    upper_bounds = []
    pretty = []
    for cond in rocs_raw:
        if cond is None:
            continue
        s = str(cond)
        pretty.append(pretty_roc_piece(cond))
        # regex for Re(s) > number or Re(s) < number
        m_gt = re.search(r'Re\(s\)\s*>\s*([-+]?\d+(\.\d+)?)', s)
        m_lt = re.search(r'Re\(s\)\s*<\s*([-+]?\d+(\.\d+)?)', s)
        if m_gt:
            try:
                lower_bounds.append(float(m_gt.group(1)))
            except:
                pass
        if m_lt:
            try:
                upper_bounds.append(float(m_lt.group(1)))
            except:
                pass
    return lower_bounds, upper_bounds, pretty

def analyze_roc_from_terms(terms, signal_expr=None):
    """
    Compute LT per-term with noconds=False and then decide ROC intersection.
    Returns (laplace_exists_bool, laplace_note_string, Xs_expr_or_None, roc_pretty_list).
    """
    Xs_parts = []
    rocs_raw = []
    for term in terms:
        try:
            res = sp.laplace_transform(term, t, s, noconds=False)
        except Exception:
            # fallback: attempt noconds=True then no condition
            try:
                expr_only = sp.laplace_transform(term, t, s, noconds=True)
                Xs_parts.append(expr_only)
            except Exception:
                pass
            continue

        if isinstance(res, tuple):
            # SymPy often returns (expr, cond) or (expr, cond, something)
            if len(res) >= 2:
                expr_term = res[0]
                cond_term = res[1]
            else:
                expr_term = res[0]
                cond_term = None
        else:
            expr_term = res
            cond_term = None

        Xs_parts.append(expr_term)
        if cond_term not in (None, sp.S.false):
            # cond_term can be And(...). Keep raw pieces.
            if isinstance(cond_term, sp.And):
                rocs_raw.extend(list(cond_term.args))
            else:
                rocs_raw.append(cond_term)

    # Combined symbolic expression
    if Xs_parts:
        Xs_expr = sp.simplify(sum(Xs_parts))
    else:
        Xs_expr = None

    # Parse ROC fragments
    lower_bounds, upper_bounds, pretty_list = parse_roc_strings(rocs_raw)

    # Special checks using signal_expr (explicit sidedness)
    # If both right-sided and left-sided exponentials with incompatible ROCs -> no LT
    try:
        two_sided_flag = (signal_expr is not None and
                          signal_expr.has(sp.Heaviside(t)) and
                          signal_expr.has(sp.Heaviside(-t)) and
                          (signal_expr.has(sp.exp(t)) or signal_expr.has(sp.exp(2*t)) or signal_expr.has(sp.exp(sp.Symbol('a')*t)) ))
        if two_sided_flag:
            # additional check: if rocs_raw indicate right and left bounds that conflict -> no LT
            if lower_bounds and upper_bounds:
                if max(lower_bounds) >= min(upper_bounds):
                    return False, "Laplace Transform does NOT exist for this two-sided signal (empty ROC).", Xs_expr, pretty_list
            # If there are clear contradictory sides, also mark as no LT
            if (lower_bounds and upper_bounds and max(lower_bounds) >= min(upper_bounds)):
                return False, "Laplace Transform does NOT exist (conflicting per-term ROCs).", Xs_expr, pretty_list
    except Exception:
        pass

    # Decide on existence from bounds:
    if lower_bounds and not upper_bounds:
        # e.g., Re(s) > a  (right half-plane)
        return True, f"Laplace transform exists (ROC: Re(s) > {max(lower_bounds)}).", Xs_expr, pretty_list
    if upper_bounds and not lower_bounds:
        # e.g., Re(s) < b  (left half-plane)
        return True, f"Laplace transform exists (ROC: Re(s) < {min(upper_bounds)}).", Xs_expr, pretty_list
    if lower_bounds and upper_bounds:
        # intersection check
        if max(lower_bounds) < min(upper_bounds):
            return True, f"Laplace transform exists (ROC: {max(lower_bounds)} < Re(s) < {min(upper_bounds)}).", Xs_expr, pretty_list
        else:
            return False, f"Laplace Transform does NOT exist (disjoint ROCs: Re(s) > {max(lower_bounds)} and Re(s) < {min(upper_bounds)}).", Xs_expr, pretty_list

    # If no numeric bounds parsed from SymPy:
    # make a best-effort from heuristics:
    # - if signal is right-sided (contains Heaviside(t)) and looks like exp(a*t)*Heaviside(t): return proper ROC from exponent
    # - otherwise, report no valid ROC found
    try:
        # detect right-sided single exponential-like term: exp(a*t)*Heaviside(t)
        if signal_expr is not None:
            # pattern: exp(k*t)*Heaviside(t)
            # find exp coefficient(s)
            exps = [e for e in signal_expr.atoms(sp.exp) if isinstance(e, sp.exp)]
            for e in exps:
                # e is exp(arg); check arg is a*t or number*t
                arg = e.args[0]
                if arg.has(t):
                    # try to extract numeric coefficient of t
                    coeff = sp.simplify(sp.expand(arg)/t)
                    try:
                        coefff = float(coeff)
                        # if Heaviside(t) present -> ROC Re(s) > coeff
                        if signal_expr.has(sp.Heaviside(t)) and not signal_expr.has(sp.Heaviside(-t)):
                            return True, f"Laplace transform exists (ROC: Re(s) > {coefff}).", Xs_expr, pretty_list
                        # if Heaviside(-t) present -> ROC Re(s) < coeff (left-sided)
                        if signal_expr.has(sp.Heaviside(-t)) and not signal_expr.has(sp.Heaviside(t)):
                            return True, f"Laplace transform exists (ROC: Re(s) < {coefff}).", Xs_expr, pretty_list
                    except Exception:
                        pass
    except Exception:
        pass

    # fallback: no valid ROC info found
    return False, "Laplace Transform does NOT exist (no valid ROC found from term conditions).", Xs_expr, pretty_list

# -----------------------------
# Fourier & Laplace existence
# -----------------------------
st.header("Transforms & Existence checks")
fourier_exists = False
fourier_note = ""

# Special FT cases
if x_sym.has(sp.DiracDelta):
    fourier_exists = True
    fourier_note = "Shifted Dirac delta: FT exists distributionally, X(ω)=e^{-jωa}."

elif (x_sym.has(sp.sin) or x_sym.has(sp.cos)) and not x_sym.has(sp.Heaviside):
    # Pure infinite-duration sinusoid
    fourier_exists = True
    fourier_note = "Pure sinusoid: FT exists only in distribution sense (impulses at ±ω₀)."

elif (x_sym.has(sp.sin) or x_sym.has(sp.cos)) and x_sym.has(sp.Heaviside):
    # Causal sinusoid
    fourier_exists = False
    fourier_note = "Causal sinusoid: FT does NOT exist (not absolutely integrable)."
    
elif x_sym == sp.Heaviside(t):
    fourier_exists = False
    fourier_note = "Unit step: ∫|u(t)| dt diverges ⇒ FT does NOT exist."

else:
    # Generic absolute integrability test
    try:
        abs_int = sp.integrate(sp.Abs(x_sym), (t, -sp.oo, sp.oo))
        if abs_int.is_finite:
            fourier_exists = True
            fourier_note = f"∫|x(t)| dt = {abs_int} (finite) ⇒ FT exists."
        else:
            fourier_exists = False
            fourier_note = "∫|x(t)| dt diverges ⇒ FT does NOT exist."
    except Exception:
        fourier_exists = False
        fourier_note = "Could not determine absolute integrability ⇒ assume FT does not exist."


# -----------------------------
# Laplace Transform (clean + working)
# -----------------------------
laplace_exists = False
laplace_note = ""
laplace_expr_plain = None
laplace_cond_plain = None
Xs_expr = None

if show_laplace_symbolic:
    try:
        # Compute Laplace expression (ignore conditions so it always works)
        Xs_expr = sp.laplace_transform(x_sym, t, s, noconds=True)
        laplace_expr_plain = plain_str(Xs_expr)
        laplace_exists = True
        laplace_note = "Laplace transform computed successfully."

        # Assign ROC for known signals
        if x_sym.equals(sp.Heaviside(t)):
            laplace_cond_plain = "Re(s) > 0"
            laplace_note = "Laplace of u(t) = 1/s, ROC: Re(s) > 0"

        elif x_sym.equals(t*sp.Heaviside(t)):
            laplace_cond_plain = "Re(s) > 0"
            laplace_note = "Laplace of t·u(t) = 1/s², ROC: Re(s) > 0"

        elif x_sym.is_polynomial(t) and x_sym.has(sp.Heaviside):
            laplace_cond_plain = "Re(s) > 0"

        elif x_sym.has(sp.sin) and x_sym.has(sp.Heaviside):
            laplace_cond_plain = "Re(s) > 0"

        elif x_sym.has(sp.cos) and x_sym.has(sp.Heaviside):
            laplace_cond_plain = "Re(s) > 0"

        elif x_sym.has(sp.DiracDelta):
            laplace_cond_plain = "All s"
            laplace_note = "Laplace of δ(t-a) = e^{-as}, valid for all s"

        else:
            laplace_cond_plain = "Depends on signal"

    except Exception as e:
        laplace_exists = False
        laplace_note = f"Laplace symbolic failed: {e}"
        laplace_expr_plain = None
        laplace_cond_plain = None





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

# Laplace symbolic computation per-term
# laplace_exists = False
# laplace_note = ""
# laplace_expr_plain = None
# laplace_cond_plain = None
# Xs_expr = None
# Special case: sin/cos with Heaviside
if show_laplace_symbolic and (x_sym.has(sp.sin) or x_sym.has(sp.cos)) and x_sym.has(sp.Heaviside):
    try:
        Xs_expr = sp.laplace_transform(x_sym, t, s, noconds=True)
        laplace_expr_plain = plain_str(Xs_expr)
        laplace_exists = True
        laplace_note = "Laplace transform exists (ROC: Re(s) > 0)."
        laplace_cond_plain = "Re(s) > 0"
    except Exception as e:
        laplace_exists = False
        laplace_note = f"Laplace symbolic failed: {e}"
        laplace_expr_plain = None
        laplace_cond_plain = None
else:
    # Your existing analyze_roc_from_terms logic
    try:
        terms = x_sym.as_ordered_terms()
        laplace_exists, laplace_note, Xs_expr, roc_pretty = analyze_roc_from_terms(terms, signal_expr=x_sym)
        laplace_expr_plain = plain_str(Xs_expr) if Xs_expr is not None else None
        if roc_pretty:
            laplace_cond_plain = " & ".join([p for p in roc_pretty if p])
        else:
            laplace_cond_plain = None
    except Exception as e:
        laplace_exists = False
        laplace_note = f"Laplace symbolic failed: {e}"
        Xs_expr = None
        laplace_cond_plain = None
# Special case: DiracDelta
if show_laplace_symbolic and x_sym.has(sp.DiracDelta):
    try:
        Xs_expr = sp.laplace_transform(x_sym, t, s, noconds=True)
        laplace_expr_plain = plain_str(Xs_expr)
        laplace_exists = True
        laplace_note = "Laplace transform of δ(t) is 1 (exists for all s)."
        laplace_cond_plain = "All s"
    except Exception as e:
        laplace_exists = False
        laplace_note = f"Laplace symbolic failed: {e}"
        laplace_expr_plain = None
        laplace_cond_plain = None
# Special case: pure sinusoid without Heaviside
if show_laplace_symbolic and (x_sym.has(sp.sin) or x_sym.has(sp.cos)) and not x_sym.has(sp.Heaviside):
    try:
        Xs_expr = sp.laplace_transform(x_sym, t, s, noconds=True)
        laplace_expr_plain = plain_str(Xs_expr)
        laplace_exists = False
        laplace_note = "Laplace transform exists but ROC does NOT include jω-axis (pure sinusoid)."
        laplace_cond_plain = "Re(s) ≠ 0"
    except Exception as e:
        laplace_exists = False
        laplace_note = f"Laplace symbolic failed: {e}"
        laplace_expr_plain = None
        laplace_cond_plain = None


# if show_laplace_symbolic:
#     try:
#         terms = x_sym.as_ordered_terms()
#         laplace_exists, laplace_note, Xs_expr, roc_pretty = analyze_roc_from_terms(terms, signal_expr=x_sym)
#         laplace_expr_plain = plain_str(Xs_expr) if Xs_expr is not None else None
#         if roc_pretty:
#             laplace_cond_plain = " & ".join([p for p in roc_pretty if p])
#         else:
#             laplace_cond_plain = None
#     except Exception as e:
#         laplace_exists = False
#         laplace_note = f"Laplace symbolic failed: {e}"
#         Xs_expr = None
#         laplace_cond_plain = None

# Results panel 
st.subheader("Results summary")
col1, col2 = st.columns(2)

with col1:
    st.write("*Fourier Transform*")
    if fourier_exists:
        st.success("Exists")
    else:
        st.error("Does NOT exist")
    st.write(fourier_note)
    if sympy_fourier is not None:
        st.write("Symbolic FT (ω domain):")
        st.latex(sp.latex(sympy_fourier))

with col2:
    st.write("*Laplace Transform*")
    if laplace_exists:
        st.success("Exists")
        st.write(laplace_note)
        if laplace_expr_plain:
            st.write("Symbolic LT (s domain):")
            st.latex(sp.latex(Xs_expr))
        if laplace_cond_plain:
            st.write(f"Valid ROC / Condition: {laplace_cond_plain}")
    else:
        st.error("Does NOT exist")
        st.write("Laplace Transform does not exist for this signal (invalid or empty ROC).")
        if laplace_expr_plain:
            st.write("Symbolic LT (s domain, but invalid ROC):")
            st.latex(sp.latex(Xs_expr))
        if laplace_cond_plain:
            st.write(f"Conflicting ROC / Condition: {laplace_cond_plain}")

# Fourier numeric spectrum 
if show_fft_numeric:
    if fourier_exists:
        if x_sym.has(sp.DiracDelta):
            st.subheader("Fourier Transform of shifted δ(t-a)")

            delta_atom = list(x_sym.atoms(sp.DiracDelta))[0]
            shift_expr = delta_atom.args[0]   # e.g., δ(t-2) → shift_expr = t-2

            # Extract numeric 'a' from δ(t-a)
            try:
                # Solve t - a = 0 → a
                a_val = sp.solve(shift_expr, t)[0]
                a_val = float(a_val)
            except Exception:
                a_val = 0.0   # fallback if not numeric

            omega_vals = np.linspace(-20, 20, 400)
            X_vals = np.exp(-1j * omega_vals * a_val)

            fig_delta, ax_delta = plt.subplots(figsize=(7,3))
            ax_delta.plot(omega_vals, np.abs(X_vals))
            ax_delta.set_xlabel("Frequency (ω)")
            ax_delta.set_ylabel("|X(ω)|")
            ax_delta.set_title(f"FT of δ(t-{a_val}): |X(ω)| = 1, phase = -ω·{a_val}")
            ax_delta.grid(True)
            st.pyplot(fig_delta)


        elif x_sym.has(sp.sin) or x_sym.has(sp.cos):
            st.subheader("Fourier Transform of sinusoid")
            st.info("Pure sinusoids are not absolutely integrable, but their FT exists in distribution sense: impulses at ±ω₀.")
            try:
                arg = (x_sym.atoms(sp.sin) | x_sym.atoms(sp.cos)).pop().args[0]  # e.g. 2*pi*t
                omega0 = float(sp.simplify(arg/t))   # extract numeric frequency
            except Exception:
                omega0 = 1.0
            fig_imp, ax_imp = plt.subplots(figsize=(7,3))
            ax_imp.axvline(omega0, linestyle='--', ymax=0.8)
            ax_imp.axvline(-omega0, linestyle='--', ymax=0.8)
            ax_imp.set_xlabel("Frequency (ω)")
            ax_imp.set_ylabel("Impulse strength")
            ax_imp.set_title("FT of sinusoid: impulses at ±ω₀")
            ax_imp.grid(True)
            st.pyplot(fig_imp)

        elif x_vals is not None:
            st.subheader("Fourier Numeric Spectrum (FFT)")
            dt = (t1_sample - t0_sample) / pts
            freq_vals = np.fft.fftshift(np.fft.fft(x_vals))
            freq_axis = np.fft.fftshift(np.fft.fftfreq(len(x_vals), dt))
            fig_fft, ax_fft = plt.subplots(figsize=(7,3))
            ax_fft.plot(freq_axis, np.abs(freq_vals))
            ax_fft.set_xlabel("Frequency (Hz)"); ax_fft.set_ylabel("|X(f)|"); ax_fft.set_title("Fourier Magnitude Spectrum")
            ax_fft.grid(True); st.pyplot(fig_fft)
    else:
        st.info("Fourier Transform does not exist for this signal (no FFT shown).")

# Laplace |X(s)| vs sigma
if show_Xs_sigma:
    if laplace_exists and Xs_expr is not None:
        st.subheader("|X(s)| vs σ for ω={}".format(omega_for_sigma))
        sigma_vals = np.linspace(sigma_range[0], sigma_range[1], 400)
        Xs_abs_vals = []
        try:
            f_Xs = sp.lambdify(s, Xs_expr, "numpy")
            for sigma_val in sigma_vals:
                Xs_abs_vals.append(abs(f_Xs(sigma_val + 1j*omega_for_sigma)))
            fig_Xs, ax_Xs = plt.subplots(figsize=(7,3))
            ax_Xs.plot(sigma_vals, Xs_abs_vals)
            ax_Xs.set_xlabel("σ"); ax_Xs.set_ylabel("|X(s)|"); ax_Xs.grid(True); st.pyplot(fig_Xs)
        except Exception as e:
            st.info(f"|X(s)| plot failed: {e}")
    else:
        st.info("Laplace Transform does not exist for this signal (no |X(s)| plot shown).")
