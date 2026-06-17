# EPGFRE
Pan-genome machine learning for fluoroquinolone resistance prediction in E. coli – interactive web app (EPGFRE) with 5‑fold CV AUC 0.914, validated across 2,715 genomes.
<img width="1410" height="734" alt="image" src="https://github.com/user-attachments/assets/6c008d4c-3731-413f-a52f-7bd38aebb0fb" />
# EPGFRE — E. coli Pan‑Genome Fluoroquinolone Resistance Explorer

[![Open in Hugging Face](https://img.shields.io/badge/🤗-Open%20in%20Hugging%20Face-blue)](https://huggingface.co/spaces/supzammy/EPGFRE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20733406-blue)](https://doi.org/10.10.5281/zenodo.20733406)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**EPGFRE** (pronounced *“ep‑gee‑free”*) is an interactive web tool that predicts fluoroquinolone (FQ) resistance in *E. coli* from pan‑genome presence/absence profiles. It is powered by a **random forest** classifier trained on **11,208** protein gene families across **2,715** *E. coli* genomes, achieving a **5‑fold cross‑validated AUC of 0.914 ± 0.014**.

The tool is designed for two distinct workflows:

- **Predict** – upload a real genome matrix (CSV) and get a resistance probability, classification, driving genes, and population context.
- **Explore** – a teaching sandbox with 10 SHAP‑selected genes you can toggle to see how each shifts the prediction.

📊 **Live demo:** [https://huggingface.co/spaces/supzammy/EPGFRE](https://huggingface.co/spaces/supzammy/EPGFRE)  
📄 **Data source:** Zenodo [10.5281/zenodo.20733406](https://zenodo.org/records\/10.5281/zenodo.20733406) 


---

## 🔬 Key Results

| Metric | Value |
|--------|-------|
| **5‑fold CV AUC** | **0.914 ± 0.014** |
| **Permutation test p‑value** | **0.001** (1000 shuffles) |
| **Feature space** | **11,208** PGFam gene families |
| **Training genomes** | **2,715** *E. coli* |
| **Multi‑drug generalisation** | Median AUC **0.91** (38 drugs, range 0.634–0.996) |
| **Temporal holdout (≤2017 / ≥2019)** | AUC **0.757** |
| **ST131 holdout** | AUC **0.748** |
| **10‑gene clinical panel** | AUC **0.684** (Sens 0.726, Spec 0.601) |

### What the model reveals

The top predictive features are **not** the canonical FQ resistance genes (`gyrA`/`parC`/`qnr`). Instead, they are **plasmid‑borne mobile elements** – β‑lactamases (TEM, CTX‑M), macrolide phosphotransferase (Mph), aminoglycoside transferases (AadA), integron integrase (IntI1), and toxin‑antitoxin systems (PemI/PemK). This tells us that the model is largely detecting **plasmid co‑residence burden** rather than the direct FQ resistance mechanism.

Performance drops on the dominant ST131 lineage (AUC 0.689), revealing **phylogenetic confounding**. We report this candidly as a quantified limitation.

---

## How to use the web app (Hugging Face Space)

Visit the live Space: [https://huggingface.co/spaces/supzammy/EPGFRE](https://huggingface.co/spaces/supzammy/EPGFRE)

1.  **Predict tab** – upload a CSV with one row per genome and one column per PGFam ID (values `0`/`1`). Click **Run prediction** to get:
    - Resistance probability and classification
    - Feature coverage (% of expected columns found)
    - Driving genes bar chart
    - Population context density plot
2.  **Explore tab** – toggle the 10 marker genes on/off to see how each shifts the prediction (teaching sandbox, not the full model).
3.  **How to Use** – detailed instructions on building a genome matrix.
4.  **About** – pipeline description, validation results, and gene dictionary.

---

## Getting the model files

The Hugging Face Space includes the trained model. For local development, you can download the model files from the Space:

- `model.joblib` – the fitted Random Forest
- `gene_features_list.pkl` – the 11,208 feature names (order must match training)

Place both files in the same directory as `app.py` to enable the **Predict** tab locally.

---

## 💻 Running locally

### 1. Clone the repository

    ```bash
    git clone https://github.com/your-username/EPGFRE.git
    cd EPGFRE


### 2. Install dependencies
    ```bash
    pip install -r requirements.txt

### 3. Run the app
    ```bash
    python app.py
    Open http://127.0.0.1:7860 in your browser.

###  Repository structure

EPGFRE/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Files to exclude from version control
├── sample_genome.csv      # Optional: example genome matrix
└── model/                 # (optional) model files (if included)
    ├── model.joblib
    └── gene_features_list.pkl
###  Citation

If you use EPGFRE or its model in your research, please cite:

    ```bibtex
@article{EPGFRE2025,
  author    = {Your Name and Co-authors},
  title     = {EPGFRE: Pan‑Genome Machine Learning Reveals Plasmid‑Driven Fluoroquinolone Resistance Signatures in *E. coli* with Quantified Phylogenetic Confounding},
  journal   = {[Journal Name]},
  year      = {2025},
  doi       = {[DOI]},
  url       = {https://huggingface.co/spaces/your-username/EPGFRE},
  note      = {Available at Hugging Face Spaces}
}

## Acknowledgements
    ```text
** BV‑BRC – genome annotations and antibiogram data

** Zenodo – curated antibiogram dataset (10.5281/zenodo.15809334)

** Hugging Face – hosting the interactive Space

## Disclaimer

This tool is a research prototype. It is not a clinical diagnostic, not a substitute for phenotypic susceptibility testing, and not validated outside the 2,715‑genome training cohort. Always confirm clinically relevant calls with laboratory methods.

## License

MIT © Zaeem Ahmad Mansoori 2026

---

## Placeholders to update

| Placeholder | Replace with |
|-------------|--------------|
| `your-username` | Your Hugging Face / GitHub username |
| `your-zenodo-record` | Your Zenodo record ID (e.g., `15809334`) |
| `[Journal Name]` | The journal you submit to |
| `[DOI]` | Your paper's DOI |
| `Your Name and Co-authors` | Your author list |


## ScreenShots

<img width="1410" height="734" alt="image" src="https://github.com/user-attachments/assets/6c008d4c-3731-413f-a52f-7bd38aebb0fb" />
(1)
<img width="1406" height="641" alt="image" src="https://github.com/user-attachments/assets/5b1fb95f-6365-4b76-bd3d-e480e33bc2e4" />
(2)
<img width="1417" height="730" alt="image" src="https://github.com/user-attachments/assets/84ee8c9f-d9a6-4fcf-af00-abd2c24eaa43" />
(3)
<img width="1420" height="732" alt="image" src="https://github.com/user-attachments/assets/b543bffd-d7a7-458e-93dd-6b95532596a0" />
(4)
<img width="1421" height="705" alt="image" src="https://github.com/user-attachments/assets/07cc5b5f-ec82-44fd-8016-4807e8c4aed8" />
(5)
<img width="1356" height="727" alt="image" src="https://github.com/user-attachments/assets/39d42ca9-1dc4-4282-9948-35a63fa0b40c" />
(6)

Copy this into your GitHub repository as `README.md`. It will render beautifully and provide all the necessary context, links, and instructions for users and reviewers.
