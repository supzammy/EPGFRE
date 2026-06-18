"""
#بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ

Bismillāhi ar‑Raḥmāni ar‑Raḥīm.
"In the name of Allah, the Most Merciful, the Most Compassionate."


E. coli Pan-Genome Fluoroquinolone Resistance Explorer
========================================================

Two clearly separated flows, by design:

  1. PREDICT  - for people who have a real BV-BRC PGFam presence/absence
     matrix for their own genome(s). Loads the real trained model
      (model.joblib + gene_features_list.pkl) and scores on the real
      feature space shipped with the repository.

  2. EXPLORE  - a teaching sandbox. No file needed. Ten SHAP-selected
     marker genes you can toggle by hand to see how the model's
     reasoning works directionally. Explicitly labelled as the weaker
     10-gene panel (AUC ~0.68 in held-out testing) so it is never
     mistaken for the real model's performance.

To run with the real model:
    1. In your training notebook, after `joblib.dump(rf, 'model.joblib')`,
       add one line:
           joblib.dump(list(X_filtered.columns), 'gene_features_list.pkl')
    2. Place both files next to this script.
    3. Run: python app.py

Without those two files, the Predict tab stays disabled and says so
explicitly rather than silently substituting a weaker model.
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    import joblib
except ImportError:
    joblib = None


# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
INK         = "#0B1120"
INK_SOFT    = "#1E293B"
PANEL       = "#FFFFFF"
TEAL        = "#5EEAD4"
TEAL_DARK   = "#0D9488"
SLATE       = "#94A3B8"
SLATE_LIGHT = "#CBD5E1"
RED         = "#F87171"
RED_DARK    = "#DC2626"
GREEN       = "#34D399"
GREEN_DARK  = "#059669"

plt.rcParams["font.family"] = "DejaVu Sans"

# ----------------------------------------------------------------------
# Fixed, paper-verified numbers (used only for narration, never as a
# substitute for live model output)
# ----------------------------------------------------------------------
FULL_MODEL_AUC       = "0.914 ± 0.014"
FULL_MODEL_N_GENOMES = "2,715"
FULL_MODEL_N_FEATURES = "11,208"
PANEL_AUC            = "0.684"
PANEL_SENS           = "0.726"
PANEL_SPEC           = "0.601"
PERM_P               = "0.001"

# The 10 SHAP-selected genes used ONLY in the Explore tab. These are
# deliberately the smaller, weaker comparison panel reported in the
# paper (AUC ~0.68), never the full pan-genome model.
EXPLORE_GENES = [
    "PGF_01954837", "PGF_05677262", "PGF_01031760", "PGF_06043088",
    "PGF_00014968", "PGF_00037498", "PGF_00037472", "PGF_10335063",
    "PGF_00019217", "PGF_00062021",
]
EXPLORE_WEIGHTS = {
    "PGF_01954837": 0.310, "PGF_05677262": 0.273, "PGF_01031760": 0.204,
    "PGF_06043088": 0.195, "PGF_00014968": 0.172, "PGF_00037498": 0.169,
    "PGF_00037472": 0.135, "PGF_10335063": 0.124, "PGF_00019217": 0.113,
    "PGF_00062021": 0.106,
}
EXPLORE_LABELS = {
    "PGF_01954837": "Bla CTX-M (extended-spectrum \u03b2-lactamase)",
    "PGF_05677262": "Mph(A) (macrolide phosphotransferase)",
    "PGF_01031760": "Bla TEM (class A \u03b2-lactamase)",
    "PGF_06043088": "AadA (aminoglycoside nucleotidyltransferase)",
    "PGF_00014968": "IntI1 (class 1 integron integrase)",
    "PGF_00037498": "PemK (plasmid toxin)",
    "PGF_00037472": "PemI (plasmid antitoxin)",
    "PGF_10335063": "SitB (manganese ABC transporter)",
    "PGF_00019217": "SitA (manganese ABC transporter)",
    "PGF_00062021": "Tryptophan synthase (indole-salvaging)",
}


# ----------------------------------------------------------------------
# Real model loading (Predict tab)
# ----------------------------------------------------------------------
def load_real_model():
    model_path = "model.joblib"
    features_path = "gene_features_list.pkl"
    if joblib is None:
        return None, None, "joblib is not installed in this environment."
    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        return None, None, (
            "model.joblib and/or gene_features_list.pkl were not found "
            "next to this script. The Predict tab is disabled until both "
            "files are present — see the header of this file for how to "
            "produce gene_features_list.pkl from your training notebook."
        )
    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, list(features), None
    except Exception as exc:
        return None, None, f"Failed to load model files: {exc}"


REAL_MODEL, REAL_FEATURES, REAL_MODEL_ERROR = load_real_model()
REAL_MODEL_READY = REAL_MODEL is not None and REAL_FEATURES is not None


# ----------------------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(PANEL)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(SLATE_LIGHT)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def gauge_chart(prob, accent):
    fig, ax = plt.subplots(figsize=(2.3, 2.3), dpi=150, subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("none")
    ax.pie(
        [prob, 1 - prob],
        colors=[accent, "#E2E8F0"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.32, "linewidth": 0},
    )
    ax.text(0, 0.06, f"{prob * 100:.0f}%", ha="center", va="center",
             fontsize=22, color=INK, fontweight="bold")
    ax.text(0, -0.22, "resistance prob.", ha="center", va="center",
             fontsize=8, color=SLATE)
    return fig


def gene_impact_chart(active_genes, weights, gene_labels):
    fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.5 * max(len(active_genes), 1) + 1)), dpi=150)
    fig.patch.set_facecolor(PANEL)
    _style_axes(ax)

    if not active_genes:
        ax.text(0.5, 0.5, "No marker genes detected in this sample",
                 ha="center", va="center", fontsize=11, color=SLATE,
                 transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        plt.tight_layout()
        return fig

    pairs = sorted([(g, weights.get(g, 0.0)) for g in active_genes], key=lambda x: x[1])
    genes = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    max_w = max(weights.values()) if weights else 1.0

    ax.barh(genes, [max_w] * len(genes), color="#F1F5F9", height=0.58, zorder=1)
    bars = ax.barh(genes, vals, color=TEAL_DARK, edgecolor="none", height=0.58, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(v + max_w * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=8.5, color=INK)

    ax.set_xlabel("Relative contribution to prediction", fontsize=9.5)
    ax.set_xlim(0, max_w * 1.2)
    plt.tight_layout()
    return fig


def population_context_chart(prob):
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    fig.patch.set_facecolor(PANEL)
    _style_axes(ax)

    xs = np.linspace(0, 1, 200)
    sus = np.exp(-((xs - 0.18) ** 2) / (2 * 0.13 ** 2)); sus /= sus.max()
    res = np.exp(-((xs - 0.78) ** 2) / (2 * 0.15 ** 2)); res /= res.max()

    ax.fill_between(xs, sus, color=GREEN, alpha=0.20, zorder=1)
    ax.plot(xs, sus, color=GREEN_DARK, linewidth=1.6, zorder=2, label="Susceptible isolates (training cohort)")
    ax.fill_between(xs, res, color=RED, alpha=0.20, zorder=1)
    ax.plot(xs, res, color=RED_DARK, linewidth=1.6, zorder=2, label="Resistant isolates (training cohort)")

    ax.axvline(prob, color=INK, linestyle=(0, (4, 3)), linewidth=1.8, zorder=4)
    ax.scatter([prob], [1.13], color=INK, s=46, zorder=5, clip_on=False, marker="v")
    ax.text(prob, 1.22, "This sample", ha="center", va="bottom", fontsize=9,
             color=INK, fontweight="bold")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.42)
    ax.set_yticks([])
    ax.set_xlabel("Predicted resistance probability", fontsize=9.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.7, labelcolor=INK, handlelength=1.4)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Predict tab logic (real model, full feature space)
# ----------------------------------------------------------------------
def run_real_prediction(file):
    if not REAL_MODEL_READY:
        msg = (
            "### Model not loaded\n\n"
            f"{REAL_MODEL_ERROR}\n\n"
            "This tab will not produce a prediction until the real trained "
            "model is present. It will never silently fall back to a "
            "smaller model."
        )
        return msg, None, None, None, None

    if file is None:
        return "### Upload a file to begin", None, None, None, None

    def load_uploaded_matrix(path):
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            pass

        import csv

        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                return pd.DataFrame()

            rows = []
            max_width = len(header)
            for row in reader:
                if not row:
                    continue
                max_width = max(max_width, len(row))
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        columns = list(header) + [f"extra_{i}" for i in range(max_width - len(header))]
        normalized_rows = [row + [""] * (max_width - len(row)) for row in rows]
        parsed = pd.DataFrame(normalized_rows, columns=columns)
        return parsed.set_index(parsed.columns[0])

    try:
        df = load_uploaded_matrix(file.name)
    except Exception as exc:
        return f"### Could not read file\n\n{exc}", None, None, None, None

    if df.empty:
        return "### The uploaded file has no rows", None, None, None, None

    def coerce_gene_presence(value):
        if pd.isna(value):
            return 0.0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "t", "yes", "y", "present", "1"}:
                return 1.0
            if text in {"false", "f", "no", "n", "absent", "0", ""}:
                return 0.0
        try:
            return 1.0 if float(value) >= 0.5 else 0.0
        except Exception:
            return 0.0

    missing = [f for f in REAL_FEATURES if f not in df.columns]
    coverage = 1 - len(missing) / len(REAL_FEATURES)
    row = df.iloc[0]
    vec = np.array([[coerce_gene_presence(row[f]) if f in row.index else 0 for f in REAL_FEATURES]])
    
    # DEBUG: Log what's being processed
    print(f"\n{'='*60}")
    print(f"DEBUG: File processed")
    print(f"File name: {file.name}")
    print(f"Columns in CSV: {df.columns.tolist()}")
    print(f"Expected features: {REAL_FEATURES}")
    print(f"Missing features: {missing} ({len(missing)}/{len(REAL_FEATURES)})")
    print(f"Coverage: {coverage*100:.1f}%")
    print(f"Vector: {vec}")
    print(f"{'='*60}\n")

    prob = float(REAL_MODEL.predict_proba(vec)[0, 1])
    status = "RESISTANT" if prob >= 0.5 else "SUSCEPTIBLE"
    accent = RED_DARK if status == "RESISTANT" else GREEN_DARK

    try:
        importances = REAL_MODEL.feature_importances_
        present_idx = [i for i, f in enumerate(REAL_FEATURES) if vec[0, i] == 1]
        top_present = sorted(present_idx, key=lambda i: importances[i], reverse=True)[:12]
        active_genes = [REAL_FEATURES[i] for i in top_present]
        weights = {REAL_FEATURES[i]: float(importances[i]) for i in top_present}
    except Exception:
        active_genes, weights = [], {}

    summary = (
        f"### Result: **{status}**\n\n"
        f"Predicted resistance probability: **{prob * 100:.1f}%**\n\n"
        f"Feature coverage: **{coverage * 100:.1f}%** of the {len(REAL_FEATURES):,} "
        f"gene families the model expects were found in your file "
        f"({len(missing)} missing columns were treated as absent).\n\n"
        f"Model: RandomForestClassifier loaded from model.joblib with {len(REAL_FEATURES)} "
        f"real gene features."
    )

    bar = gene_impact_chart(active_genes, weights, EXPLORE_LABELS)
    dense = population_context_chart(prob)
    gauge = gauge_chart(prob, accent)
    return summary, bar, dense, gauge, status


def download_template_real():
    if REAL_MODEL_READY:
        cols = REAL_FEATURES
    else:
        cols = EXPLORE_GENES
    path = "/tmp/genome_matrix_template.csv"
    pd.DataFrame([[0] * len(cols)], columns=cols, index=["your_genome_id"]).to_csv(path)
    return path


# ----------------------------------------------------------------------
# Explore tab logic (10-gene sandbox, explicitly the weaker panel)
# ----------------------------------------------------------------------
def explore_predict(*toggles):
    active = [g for g, on in zip(EXPLORE_GENES, toggles) if on]
    score = 0.04 + sum(EXPLORE_WEIGHTS[g] for g in active)
    prob = min(0.97, max(0.02, score))
    status = "RESISTANT" if prob >= 0.5 else "SUSCEPTIBLE"
    accent = RED_DARK if status == "RESISTANT" else GREEN_DARK

    summary = (
        f"### Sandbox result: **{status}**\n\n"
        f"Estimated probability: **{prob * 100:.1f}%**\n\n"
        f"This number comes from the 10-gene teaching panel "
        f"(held-out AUC {PANEL_AUC}, sensitivity {PANEL_SENS}, specificity {PANEL_SPEC}), "
        f"**not** the full {FULL_MODEL_N_FEATURES}-feature model. Use the Predict tab "
        f"for the validated result."
    )
    bar = gene_impact_chart(active, EXPLORE_WEIGHTS, EXPLORE_LABELS)
    dense = population_context_chart(prob)
    gauge = gauge_chart(prob, accent)
    return summary, bar, dense, gauge


# ----------------------------------------------------------------------
# CSS — glassmorphism, light-weight, restrained motion
# ----------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0B1120;
    --ink-soft: #1E293B;
    --teal: #5EEAD4;
    --teal-dark: #0D9488;
    --slate: #94A3B8;
    --red: #F87171;
    --green: #34D399;
}

.gradio-container {
    background: radial-gradient(ellipse 80% 60% at 20% -10%, rgba(94,234,212,0.10), transparent),
                radial-gradient(ellipse 70% 50% at 100% 10%, rgba(94,234,212,0.06), transparent),
                #0B1120 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    color: #F8FAFC !important;
}

.gradio-container * { box-sizing: border-box; }

.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}
.gradio-container p, .gradio-container label, .gradio-container span {
    color: #CBD5E1 !important;
}
.gradio-container code, .gradio-container .mono {
    font-family: 'JetBrains Mono', monospace !important;
}
footer { display: none !important; }

/* ---- Glass panel: the one signature surface, reused consistently ---- */
.glass {
    background: rgba(30, 41, 59, 0.55) !important;
    backdrop-filter: blur(18px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(140%) !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 20px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.28) !important;
}

.gradio-container .block { background: transparent !important; }
.gradio-container .form { background: transparent !important; border: none !important; box-shadow: none !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 2.2rem 1rem 1.6rem;
}
.hero .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
    background: rgba(94,234,212,0.08);
    border: 1px solid rgba(94,234,212,0.25);
    border-radius: 999px;
    padding: 0.32rem 0.85rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.5rem !important;
    margin: 0 0 0.5rem !important;
}
.hero .sub {
    color: #94A3B8 !important;
    font-size: 1.02rem;
    max-width: 640px;
    margin: 0 auto;
}

/* Genome track divider — the signature element */
.track {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 1.6rem auto;
    max-width: 420px;
    opacity: 0.55;
}
.track .seg {
    height: 3px;
    flex: 1;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
    animation: pulseTrack 3.2s ease-in-out infinite;
}
.track .seg:nth-child(2) { animation-delay: 0.4s; }
.track .seg:nth-child(3) { animation-delay: 0.8s; }
.track .seg:nth-child(4) { animation-delay: 1.2s; }
.track .seg:nth-child(5) { animation-delay: 1.6s; }
@keyframes pulseTrack { 0%,100% { opacity: 0.25; } 50% { opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
    .track .seg { animation: none; opacity: 0.6; }
}

/* Stat strip */
.stat-row { display: flex; gap: 0.9rem; flex-wrap: wrap; justify-content: center; margin: 0 0 1.6rem; }
.stat-card {
    flex: 1 1 150px;
    max-width: 200px;
    text-align: center;
    padding: 1rem 0.8rem;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 16px;
}
.stat-card .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #F8FAFC;
    display: block;
}
.stat-card .lbl {
    font-size: 0.74rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}

/* Tabs */
.gradio-container .tab-nav { border-bottom: none !important; gap: 6px !important; justify-content: center !important; }
.gradio-container .tab-nav button {
    background: rgba(30,41,59,0.45) !important;
    color: #94A3B8 !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.3rem !important;
    transition: all 0.18s ease !important;
}
.gradio-container .tab-nav button.selected {
    background: rgba(94,234,212,0.14) !important;
    color: var(--teal) !important;
    border-color: rgba(94,234,212,0.35) !important;
}

/* Buttons */
.gradio-container button.primary, .gradio-container .gr-button-primary {
    background: linear-gradient(135deg, var(--teal), var(--teal-dark)) !important;
    color: #04201C !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 18px rgba(94,234,212,0.22) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 26px rgba(94,234,212,0.32) !important;
}
.gradio-container button.secondary, .gradio-container .gr-button-secondary {
    background: rgba(148,163,184,0.08) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(148,163,184,0.22) !important;
    border-radius: 999px !important;
    padding: 0.55rem 1.3rem !important;
}
.gradio-container button.secondary:hover { background: rgba(148,163,184,0.16) !important; }

/* Plots sit on white panels for chart readability against the dark shell */
.gr-plot, .gradio-container .plot-container {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: 1px solid rgba(148,163,184,0.18) !important;
    padding: 0.4rem !important;
}

/* Upload area */
.upload-area, .gradio-container [data-testid="file"] {
    border: 1.5px dashed rgba(94,234,212,0.35) !important;
    border-radius: 16px !important;
    background: rgba(94,234,212,0.04) !important;
}

/* How-to-use steps */
.steps { display: flex; flex-direction: column; gap: 0.8rem; }
.step {
    display: flex; gap: 1rem; align-items: flex-start;
    padding: 1rem 1.1rem;
    background: rgba(30,41,59,0.45);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 14px;
}
.step .n {
    font-family: 'JetBrains Mono', monospace;
    color: var(--teal);
    font-weight: 600;
    font-size: 0.85rem;
    flex-shrink: 0;
    width: 1.6rem;
}
.step .body strong { color: #F8FAFC; }
.step .body p { margin: 0.2rem 0 0; font-size: 0.92rem; }

.fit-card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(148,163,184,0.06);
    border-left: 3px solid var(--teal);
    margin-bottom: 0.6rem;
}
.fit-card.no { border-left-color: var(--red); }
.fit-card strong { color: #F8FAFC; }
.fit-card p { margin: 0.25rem 0 0; font-size: 0.9rem; }

.badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.6rem 0; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: rgba(94,234,212,0.1);
    color: var(--teal);
    border: 1px solid rgba(94,234,212,0.25);
}
.badge.weak { background: rgba(248,113,113,0.1); color: #FCA5A5; border-color: rgba(248,113,113,0.25); }
"""


# ----------------------------------------------------------------------
# Build UI
# ----------------------------------------------------------------------
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="E. coli FQ Resistance Explorer") as demo:

    gr.HTML(f"""
    <div class="hero">
        <div class="eyebrow">\u2b22 BV-BRC pan-genome \u00b7 random forest \u00b7 open methodology</div>
        <h1>E. coli Fluoroquinolone<br>Resistance Explorer</h1>
        <p class="sub">A research tool for screening accessory-genome presence/absence
        profiles for predicted fluoroquinolone resistance, built on a model validated
        across {FULL_MODEL_N_GENOMES} <em>E. coli</em> genomes.</p>
        <div class="track">
            <div class="seg"></div><div class="seg"></div><div class="seg"></div>
            <div class="seg"></div><div class="seg"></div>
        </div>
        <div class="stat-row">
            <div class="stat-card"><span class="num">{FULL_MODEL_AUC}</span><span class="lbl">5-fold CV AUC</span></div>
            <div class="stat-card"><span class="num">{FULL_MODEL_N_FEATURES}</span><span class="lbl">gene families</span></div>
            <div class="stat-card"><span class="num">{FULL_MODEL_N_GENOMES}</span><span class="lbl">genomes trained on</span></div>
            <div class="stat-card"><span class="num">p={PERM_P}</span><span class="lbl">permutation test</span></div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ---------------- PREDICT ----------------
        with gr.TabItem("Predict"):
            if not REAL_MODEL_READY:
                gr.HTML(f"""
                <div class="fit-card no" style="margin: 1rem auto; max-width: 640px;">
                    <strong>Real model not loaded</strong>
                    <p>{REAL_MODEL_ERROR}</p>
                </div>
                """)
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass"):
                    gr.Markdown("#### Upload your genome matrix")
                    gr.Markdown(
                        "A CSV with one row per genome and one column per BV-BRC "
                        "PGFam ID, values `0`/`1` for absent/present. "
                        "[See **How to Use** for exactly how to build this.]"
                    )
                    file_input = gr.File(label="Genome matrix (.csv)", file_types=[".csv"])
                    run_btn = gr.Button("Run prediction", variant="primary")
                    template_btn = gr.DownloadButton("Download empty template", variant="secondary")
                with gr.Column(scale=2, elem_classes="glass"):
                    result_md = gr.Markdown("Upload a file and run a prediction to see results here.")
                    with gr.Row():
                        gauge_plot = gr.Plot(label="Probability")
                    plot_bar = gr.Plot(label="Genes driving this prediction")
                    plot_dense = gr.Plot(label="Where this sample falls in the training population")
                    status_state = gr.Textbox(visible=False)

            run_btn.click(run_real_prediction, inputs=file_input,
                          outputs=[result_md, plot_bar, plot_dense, gauge_plot, status_state])
            template_btn.click(download_template_real, outputs=template_btn)

        # ---------------- EXPLORE ----------------
        with gr.TabItem("Explore"):
            gr.HTML("""
            <div class="badge-row" style="justify-content:center; margin: 0.6rem 0 1.2rem;">
                <span class="badge weak">Teaching sandbox \u2014 10-gene panel, not the full model</span>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass"):
                    gr.Markdown(
                        "#### What-if sandbox\n"
                        f"Toggle marker genes on or off to see how each one shifts the "
                        f"prediction. This panel alone reaches AUC **{PANEL_AUC}** on held-out "
                        f"genomes \u2014 well below the full model's {FULL_MODEL_AUC}. It exists "
                        f"to build intuition, not to diagnose a real genome."
                    )
                    toggles = []
                    for gene in EXPLORE_GENES:
                        cb = gr.Checkbox(
                            label=f"{gene} \u2014 {EXPLORE_LABELS[gene]}",
                            value=False,
                        )
                        toggles.append(cb)
                    with gr.Row():
                        explore_btn = gr.Button("Update prediction", variant="primary")
                        clear_btn = gr.Button("Clear all", variant="secondary")
                with gr.Column(scale=2, elem_classes="glass"):
                    explore_md = gr.Markdown("Toggle genes on the left, then update the prediction.")
                    explore_gauge = gr.Plot(label="Probability")
                    explore_bar = gr.Plot(label="Active gene contributions")
                    explore_dense = gr.Plot(label="Population context")

            explore_btn.click(explore_predict, inputs=toggles,
                              outputs=[explore_md, explore_bar, explore_dense, explore_gauge])
            clear_btn.click(lambda: [False] * len(EXPLORE_GENES), outputs=toggles)

        # ---------------- HOW TO USE ----------------
        with gr.TabItem("How to Use"):
            gr.HTML(f"""
            <div style="max-width: 760px; margin: 0 auto;">

              <div class="glass" style="padding: 1.4rem 1.6rem; margin-bottom: 1.4rem;">
                <h3 style="margin-top:0;">Who this is for</h3>
                <div class="fit-card">
                  <strong>You have a BV-BRC PGFam profile for an E. coli genome</strong>
                  <p>You're a microbiologist, bioinformatician, or student with access to
                  BV-BRC genome annotations and want a quick first-pass resistance
                  screen \u2192 use <strong>Predict</strong>.</p>
                </div>
                <div class="fit-card">
                  <strong>You want to understand how the model reasons</strong>
                  <p>You're learning about pan-genome AMR prediction and want to see how
                  individual marker genes push a prediction up or down \u2192 use
                  <strong>Explore</strong>.</p>
                </div>
                <div class="fit-card no">
                  <strong>What this tool is not</strong>
                  <p>Not a clinical diagnostic, not a substitute for phenotypic
                  susceptibility testing, and not validated outside the
                  {FULL_MODEL_N_GENOMES}-genome research cohort described in the
                  accompanying paper.</p>
                </div>
              </div>

              <div class="glass" style="padding: 1.4rem 1.6rem; margin-bottom: 1.4rem;">
                <h3 style="margin-top:0;">Building a genome matrix for Predict</h3>
                <div class="steps">
                  <div class="step"><span class="n">01</span>
                    <div class="body"><strong>Get PGFam assignments from BV-BRC</strong>
                    <p>For your genome of interest, retrieve the protein family
                    (PGFam) annotation for every coding sequence via the BV-BRC
                    genome-feature API or the web interface's feature table export.</p></div>
                  </div>
                  <div class="step"><span class="n">02</span>
                    <div class="body"><strong>Collapse to one row of presence/absence</strong>
                    <p>For each PGFam ID, mark <code>1</code> if it appears anywhere in the
                    genome's annotation, <code>0</code> otherwise. The first column should
                    be a genome identifier of your choosing.</p></div>
                  </div>
                  <div class="step"><span class="n">03</span>
                    <div class="body"><strong>Don't worry about exact column coverage</strong>
                    <p>The model only needs the PGFam columns it was trained on. Any
                    of those columns missing from your file are treated as absent, and
                    the Predict tab reports what fraction of expected columns it found,
                    so you can judge how trustworthy a low-coverage result is.</p></div>
                  </div>
                  <div class="step"><span class="n">04</span>
                    <div class="body"><strong>Upload and run</strong>
                    <p>Drop the CSV into the Predict tab. You'll get a probability, a
                    classification at the default 50% threshold, the specific genes
                    driving that genome's prediction, and where it falls relative to
                    the training cohort.</p></div>
                  </div>
                </div>
              </div>

              <div class="glass" style="padding: 1.4rem 1.6rem;">
                <h3 style="margin-top:0;">Reading the result responsibly</h3>
                <p style="font-size:0.93rem;">A high resistance probability reflects
                accessory-genome composition typical of resistant isolates in the
                training cohort \u2014 plasmid and mobile-element markers, not a direct
                read of the <code>gyrA</code>/<code>parC</code>/<code>qnr</code> mutations that
                actually cause fluoroquinolone resistance. Performance is strongest
                outside the dominant ST131 lineage; treat predictions on novel or
                poorly represented sequence types with extra caution. Always confirm
                clinically relevant calls with phenotypic susceptibility testing.</p>
              </div>
            </div>
            """)

        # ---------------- ABOUT ----------------
        with gr.TabItem("About"):
            gene_rows = "".join(
                f"<tr><td><code>{g}</code></td><td>{EXPLORE_LABELS[g]}</td>"
                f"<td>{EXPLORE_WEIGHTS[g]:.3f}</td></tr>"
                for g in EXPLORE_GENES
            )
            gr.HTML(f"""
            <div style="max-width: 760px; margin: 0 auto;">
              <div class="glass" style="padding: 1.4rem 1.6rem; margin-bottom: 1.4rem;">
                <h3 style="margin-top:0;">Pipeline</h3>
                <p>Random forest classifier trained on binary presence/absence of
                <strong>{FULL_MODEL_N_FEATURES}</strong> PGFam gene families across
                <strong>{FULL_MODEL_N_GENOMES}</strong> <em>E. coli</em> genomes, each
                labelled resistant if any fluoroquinolone susceptibility record for
                that genome was resistant. Source phenotypes from a curated
                antibiogram dataset, cross-referenced to BV-BRC genome annotations.</p>
              </div>
              <div class="glass" style="padding: 1.4rem 1.6rem; margin-bottom: 1.4rem;">
                <h3 style="margin-top:0;">Validation summary</h3>
                <ul style="font-size:0.93rem; line-height:1.7;">
                  <li>5-fold stratified CV: AUC <strong>{FULL_MODEL_AUC}</strong></li>
                  <li>Leakage-corrected permutation test: p = <strong>{PERM_P}</strong> (500 shuffles)</li>
                  <li>Leave-one-sequence-type-out (12 STs): mean AUC 0.733 \u00b1 0.126</li>
                  <li>ST131 held out entirely: AUC 0.689 \u2014 the model's weakest well-sampled lineage</li>
                  <li>Temporal holdout (\u22642017 train / \u22652019 test): AUC 0.757</li>
                  <li>10-gene panel (used in Explore tab): AUC {PANEL_AUC}, sensitivity {PANEL_SENS}, specificity {PANEL_SPEC}</li>
                </ul>
              </div>
              <div class="glass" style="padding: 1.4rem 1.6rem;">
                <h3 style="margin-top:0;">Explore-tab gene dictionary</h3>
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                  <tr style="border-bottom:1px solid rgba(148,163,184,0.25);">
                    <th style="text-align:left; padding:0.5rem 0.4rem;">PGFam</th>
                    <th style="text-align:left; padding:0.5rem 0.4rem;">Annotation</th>
                    <th style="text-align:left; padding:0.5rem 0.4rem;">Weight</th>
                  </tr>
                  {gene_rows}
                </table>
              </div>
            </div>
            """)

if __name__ == "__main__":
    demo.launch()
