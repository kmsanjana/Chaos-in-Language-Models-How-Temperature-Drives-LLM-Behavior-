# 🌡️ How Temperature Controls LLM Behavior
### A Dynamical Systems Perspective on Large Language Model Output

> *"Increasing temperature doesn't just make a model more creative — it drives it through the same period-doubling route to chaos that governs a dripping faucet, a beating heart, and the logistic map."*

---

<div align="center">

**Group 6 · George Washington University · MS Data Science · Network Data Science**

*Dhwanil Mori · Jasreen Mehta · Sanjana Kadambe*

</div>

---

## 🔭 The Big Idea

When you ask an LLM a question, the answer you get depends heavily on the **temperature** parameter — a scalar that controls how randomly the model samples its next token. This project asks:

**Is temperature's effect on LLM output just "more random"? Or is there deeper structure?**

The answer is deeper structure — specifically, the same mathematical structure as a **nonlinear dynamical system**. Low temperatures lock models into **fixed attractors** (repetitive loops). Moderate temperatures create **periodic oscillations**. High temperatures push outputs into **chaos**. The transition is sharp, predictable, and mirrors the famous **period-doubling cascade** of the logistic map.

---

## 📐 Research Architecture

```
┌─────────────────────────────────────────────────────────┐
│              EMPIRICAL LAYER (Real LLMs)                │
│   GPT-2 · DistilGPT-2 · GPT-2 XL · GPT-Neo 1.3B       │
│   Prompt: "Is the Earth flat?"  ·  T ∈ [0.10, 1.50]   │
└────────────────────┬────────────────────────────────────┘
                     │ Symbolic Encoding Pipeline
                     ▼
┌─────────────────────────────────────────────────────────┐
│            SYMBOLIC DYNAMICS LAYER                      │
│   Sentences → TF-IDF vectors → Cosine clustering       │
│   → Letter labels (A, B, C…) → Symbol stream barcodes  │
└────────────────────┬────────────────────────────────────┘
                     │ Compare to
                     ▼
┌─────────────────────────────────────────────────────────┐
│          LOGISTIC MAP (THEORETICAL ANCHOR)              │
│   x_{t+1} = r × xₜ × (1 − xₜ)                        │
│   r sweeps [3.40, 4.00] · A-band encoding [0.48, 0.52] │
└────────────────────┬────────────────────────────────────┘
                     │ Geometric mechanism
                     ▼
┌─────────────────────────────────────────────────────────┐
│         TOY TRANSFORMER MODEL (ABDC)                    │
│   4-token 2D embedding space · Tipping point dynamics   │
│   Baseline vs. modified B-embedding experiments         │
└─────────────────────────────────────────────────────────┘
```

---

## 🧩 Module Breakdown

### Module 1 · Temperature Sweep on Real LLMs

**Files:** `GPT_2_symbols_TEMP_CHANGE.ipynb`, `DistilGPT_2_symbols_TEMP_CHANGE.ipynb`, `GPT_2_symbols_MINILAB_STUDENT-org.ipynb`

The core experiment runs four LLMs on the same factual prompt across 7–9 temperature values. Each response is converted into a **symbolic barcode** — a sequence of colored blocks where each color represents a cluster of semantically similar sentences.

#### The Encoding Pipeline

```
Raw text output
      │
      ▼ regex split on [.!?] + newlines
Sentence list
      │
      ▼ TF-IDF character n-grams (range 3–5)
Sparse vectors
      │
      ▼ Greedy cosine clustering (threshold ≈ 0.45)
Integer cluster labels
      │
      ▼ Map to alphabet: 0→A, 1→B, 2→C …
Symbol stream  e.g.  "ABBBBBBBBBBBBB"
      │
      ▼ Color-coded barcode plot
Visual output
```

#### Behavioral Regimes

| Temperature | Regime | Symbol Pattern | LLM Behavior |
|---|---|---|---|
| 0.10 – 0.32 | **Fixed Attractor** | `AAAAA` or `ABBBBB` | Repetitive, locked phrases, near-zero diversity |
| 0.50 – 0.55 | **Transition** | `ABCDE` (exploring) | First hallucinations, slight creativity |
| 0.55 – 0.70 | **Periodic / Mixed** | `ABCABC…` | Multi-cluster, contradictions emerge |
| 0.78 – 1.00 | **Edge of Chaos** | `ABCDEFGHIJ` | Surreal imagery, incoherent but imaginative |
| 1.30 – 1.50 | **Chaotic** | All distinct symbols | Rambling, philosophical, factually lost |

#### Model Personalities

| Model | Character |
|---|---|
| **GPT-2** | Creative surrealist — sharp jump to wild imagery at T≈0.78 |
| **DistilGPT-2** | Conservative — slow, small jumps; least chaotic |
| **GPT-2 XL** | Verbose but fragile — better diversity early, loses logic fast |
| **GPT-Neo 1.3B** | Contradictory rambler — flat vs. sphere in same sentence |

> 💡 **Key insight**: The transition between deterministic and creative behavior happens sharply between T=0.55–0.70, not gradually. This is a **phase transition**, not a smooth curve.

---

### Module 2 · Logistic Map as LLM Analog

**Files:** `llm_symbol_maps_explorer_band_no_transient (1).ipynb`, `attractor_sequence_code_files/llm_symbol_maps_explorer_LOGISTIC_MAP.ipynb`, `Empirical Attractor Periods.pdf`

The logistic map is the simplest system that exhibits a **period-doubling route to chaos**. This module demonstrates that the symbolic output of LLMs follows the same mathematical route.

#### The Logistic Map

```
x_{t+1} = r × xₜ × (1 − xₜ)
```

The control parameter `r` is treated as an analog of LLM temperature `T`.

#### Symbolic Encoding of the Logistic Map

A "band" is defined around the fixed point (x = 0.5):
- Values in `[0.48, 0.52]` → symbol **A**
- Values below → **B** or ranked labels
- Sweep `r ∈ [3.40, 4.00]` with 220 steps
- Per `r`: 30 initial conditions, 900 transient steps discarded, 240-step analysis window

#### Period-Doubling Cascade Recovered

```
r = 3.40 – 3.50  →  Period 2  →  AB AB AB …
r = 3.50 – 3.57  →  Period 4  →  ABCD ABCD …
r ≈ 3.57          →  Period 8  →  ABCDEFGH …
r ≈ 3.83          →  Period 3 window  →  ABC ABC …
r > 3.90          →  Chaos (∞)  →  BBAABBAB …
```

#### Correspondence to LLM Attractors

| Logistic Map | LLM Output |
|---|---|
| Period-1 fixed point | Single repeated phrase (`AAAA`) |
| Period-2 oscillation | Two alternating sentence types (`ABAB`) |
| Period-4 / 8 cascade | Multi-cluster cycling patterns |
| Periodic windows in chaos | Structured passages within otherwise erratic text |
| Chaos | Fully diverse, non-repeating symbol streams |

> 💡 **Key insight**: The same symbolic sequences observed in GPT output can be reproduced by the logistic map. The logistic parameter `r` maps directly to LLM temperature `T`. Both systems are governed by the same attractor geometry.

---

### Module 3 · ABDC Toy Transformer — Tipping Points

**Files:** `tipping_point_general.py`, `llm_dynamics_analyzer_ABDC (2).ipynb`, `LLM Dynamics.pdf`

To understand the *mechanism* behind attractor convergence, a minimal transformer was built from scratch in 2D space.

#### Model Design

- **Vocabulary**: 4 tokens — A, B, C, D — embedded as 2D vectors
- **Architecture**: Transformer decoder with causal multi-head attention, feed-forward network, layer normalization, residual connections
- **Output**: Logits via dot product of last hidden state with embedding matrix transpose

#### Token Embedding Space

```
         D [0.9, 0.5]  ← Strong final attractor
         B [0.8, 0.0]  ← Mid-range influencer
         A [0.4, -0.3] ← Initial seed
         C [-0.2, -0.2] ← Neutral / background
```

#### Observed Trajectory

```
Start: A
↓
A → B → B → B → D → D → D → D → D → D … (fixed attractor)

Logits at step 4 (ABBB):  A=0.301  B=0.599  C=0.000  D=0.599
                                                      ↑ tie → tips to D
```

The context vector (attention-weighted average of all previous embeddings) drifts through the 2D space. The trajectory spirals from near the origin toward D's region, where it locks in permanently.

#### Tipping Point Experiment

| Setup | B Embedding | BBB run length | Behavior |
|---|---|---|---|
| **Baseline** | `[0.8, 0.0]` | 3 tokens | Quickly tips to DDD attractor |
| **Modified** | `[0.88, 0.0]` | Much longer | Delayed tipping; circles B before D |

Increasing B's x-component strengthens its gravitational pull on the context vector. The model stays in the B-attractor basin longer before the trajectory escapes toward D.

> 💡 **Key insight**: Embedding geometry is the mechanism. The **basin of attraction** for each token is shaped by where its embedding vector sits in 2D space. Token D dominates because its vector `[0.9, 0.5]` maximizes the dot product with the growing context average. Changing an embedding shifts the tipping geometry.

---

### Module 4 · Word Embeddings: GPT-2 vs Gemma-2-2b-it

**Files:** `gemma2_embeddings_dotproducts.ipynb`, `attention_analysis.py`, `Word Embeddings.pdf`

A comparative study of how two architecturally different models organize meaning in their embedding spaces.

#### Experiment
Words `privacy`, `important`, `overrated` were extracted from both models. Dot products (full high-dimensional space) and PCA 2D projections were compared.

#### Dot Product Results (Gemma-2-2b-it)

```
             privacy   important   overrated
privacy       2.90       0.33        0.81
important     0.33       2.53        0.60
overrated     0.81       0.60        3.55
```

#### Model Comparison

| Aspect | Gemma-2-2b-it | GPT-2 |
|---|---|---|
| Embedding structure | Smooth, cohesive; *privacy* ≈ *important* in PCA | Scattered; weaker semantic clustering |
| Context understanding | Deep, contextual (instruction-tuned) | Raw co-occurrence from web text |
| Semantic alignment | Groups by shared value or intent | Groups by surface usage pattern |

#### Attention Head Analysis

`attention_analysis.py` shows that Q·K^T attention scores are a **scaled linear transformation** of raw embedding dot products:

```
With identity-like W_q, W_k:
  Q·K^T  ≈  (1 / √d_k) × X·X^T
```

Temperature's effect on attention mirrors its effect on LLM output:
- Low T → sharp, concentrated attention → fixed attractor behavior
- High T → flat, distributed attention → chaotic behavior

---

### Module 5 · Co-occurrence Network (Gephi)

**Files:** `Gephi_NK.pdf`

A word co-occurrence network was constructed from 10 GPT-2 generated community vignettes and visualized using Gephi.

#### Network Statistics
- **84 unique words** (nodes)
- **354 co-occurrence edges**
- Edge weight = adjacency frequency in symbolic sequence

This network shows the **structural skeleton** of LLM output — which symbols appear next to each other most often, creating the backbone of the attractor landscape.

---

### Module 6 · Mechanistic Interpretability (Neuronpedia)

**Files:** `Is the earth flat-neuronpedia.pptx`

Using Neuronpedia, the internal circuits of GPT-2 were probed on:
1. `"Is the Earth flat?"` — factual recall and denial circuits
2. `"Vaccines — bad?"` — sensitive content activation circuits

This connects the **behavioral** observations (symbolic attractor patterns) to the **mechanistic** level (which neurons fire), grounding the dynamical systems framework in actual transformer circuitry.

---

## 📊 Key Results at a Glance

```
TEMPERATURE EFFECT SUMMARY ACROSS ALL MODELS

Low T (0.10–0.32)        Mid T (0.55–0.70)        High T (1.00–1.50)
─────────────────        ─────────────────        ──────────────────
Fixed attractor          Transition zone           Chaotic regime
AAAAAAA or ABBBBB        ABCABCABC                 ABCDEFGHIJKLM
Repetitive + safe        Creative + fragile        Imaginative + incoherent
Deterministic            Hallucinations begin      Surrealism / rambling
Period 1–2               Period 3–8                Period ∞

                         ↑
                    Optimal zone for
                    interesting content
                    (~T = 0.55–0.70)
```

---

## � Visual Results

---

### 🎓 Project Overview

**Fig 1 — Title Slide: How Temperature Controls LLM Behavior**
*Group 6 · Models tested: GPT-2, DistilGPT-2, GPT-2 XL, EleutherAI/GPT-Neo-1.3B*

<img width="600" height="589" alt="image" src="https://github.com/user-attachments/assets/9d9e2428-c9f7-4767-abcd-3cb467a55efd" />


---

### 🌡️ Module 1 — Temperature Effects on Real LLMs

**Fig 2 — GPT-2 & DistilGPT-2: Symbol Streams Across Temperatures**
*Color-coded barcodes show how sentence diversity changes from T=0.10 (single-color block) to T=1.00 (many distinct colors). GPT-2 flips to creative outputs earlier than DistilGPT-2.*

<img width="897" height="497" alt="image" src="https://github.com/user-attachments/assets/966ab1a6-90da-4a29-ba9f-317e07db4cf9" />

<img width="930" height="524" alt="image" src="https://github.com/user-attachments/assets/00b6f625-981a-4322-9cc1-ebfff3fe3491" />




**Fig 3 — GPT-2 XL & GPT-Neo 1.3B: Symbol Streams Across Temperatures**
*GPT-2 XL locks firmly at T=0.10–0.50 then diversifies sharply. GPT-Neo mixes flat/sphere contradictions early and drifts into meta-commentary at high temps.*

<img width="1155" height="629" alt="image" src="https://github.com/user-attachments/assets/de7c9258-3d43-4314-a176-3489c1e089a8" />

<img width="1152" height="615" alt="image" src="https://github.com/user-attachments/assets/4324f1c7-8fce-45c1-9fc4-079652398fc6" />




**Fig 4 — Comparative Table: All Models × All Temperatures**
*Side-by-side summary of behavioral regime for each of the 4 models across 9 temperature values (0.10 → 1.50).*

<img width="460" height="456" alt="image" src="https://github.com/user-attachments/assets/1ab12563-1584-43f2-8f5e-5449024305ac" />


---

### 🔁 Module 2 — Logistic Map & Symbolic Dynamics

**Fig 6 — Empirical Attractor Sequences vs. Temperature (GPT-Neo 1.3B)**
*Period-2 attractors (BA, DE, EF) at low T; period-6 (KCKCKC), period-7 (ABCDEFG), period-10 (ABCBCBCDEF), and period-∞ (ABCDEFGHIJKLM) emerge as T rises — exactly mirroring the logistic map's period-doubling cascade.*

<img width="1016" height="461" alt="image" src="https://github.com/user-attachments/assets/2a58a35b-bcbd-4fd6-9d43-4e0001db348d" />



**Fig 7 — Empirical Attractor Periods vs. Logistic Control Parameter**
*Symbolic dynamics plot showing the period-doubling cascade (Period 2 → 4 → 8 → ∞) and periodic windows (3, 5, 6) embedded in the chaotic regime. The logistic parameter r maps directly to LLM temperature T.*

![Empirical Attractor Periods vs Logistic Parameter](assets/fig07_logistic_map_period_doubling.png)

**Fig 8 — How the Logistic Map Reproduces the Empirical Sequence Bands**
*Setup: A-band [0.48, 0.52], r sweep [3.40, 4.00], 220 steps, 30 initial conditions, 900-step transient discarded. Key finding: band placement is sensitive — too wide merges A with others; too narrow loses the signal entirely.*

<img width="1027" height="538" alt="image" src="https://github.com/user-attachments/assets/766c1831-03bc-437a-ad07-f4a61586d4ee" />


---

### 🧲 Module 3 — ABDC Toy Transformer & Tipping Points

**Fig 9 — Token Embedding Vectors: Role in Tipping Dynamics**
*A=[0.4,−0.3] (initial seed), B=[0.8,0.0] (mid-range influencer), C=[−0.2,−0.2] (neutral), D=[0.9,0.5] (strong final attractor). The geometry of these vectors determines which token wins the context dot-product race.*

<img width="1003" height="559" alt="image" src="https://github.com/user-attachments/assets/0bb0f066-f8fc-4534-8a37-a9371cccf194" />


**Fig 10 — Trajectory: Baseline vs. Modified B-Embedding**
*Left (B_x=0.8): context moves quickly toward D, tips after 6 BBB tokens. Right (B_x=0.88): context circles B much longer before drifting to D. Modifying one embedding shifts the entire tipping geometry.*

<img width="998" height="535" alt="image" src="https://github.com/user-attachments/assets/063a0207-52e1-4763-8afe-f7042d5b646b" />


---

### 🔤 Module 4 — Word Embeddings: GPT-2 vs Gemma-2-2b-it

**Fig 11 — PCA 2D Embedding Comparison: Gemma-2-2b-it vs GPT-2**
*Left (Gemma): "privacy" and "important" cluster closer together — smooth, cohesive semantic space. Right (GPT-2): words are more scattered, reflecting raw web co-occurrence patterns rather than deep semantic alignment.*

<img width="1007" height="541" alt="image" src="https://github.com/user-attachments/assets/0fc8c6bc-d53d-4646-a412-fdf929536222" />


**Fig 12 — Gemma-2-2b-it vs GPT-2: Qualitative Comparison Table**
*Three dimensions compared — embedding structure, context understanding, and semantic alignment — showing how instruction tuning (Gemma) produces richer, more intentional semantic geometry.*

<img width="1001" height="555" alt="image" src="https://github.com/user-attachments/assets/1f4b6282-0478-4857-97b4-61a8f393e46a" />


---

> **To add the images:** Save each screenshot to the `assets/` folder with the filename shown above (e.g. `assets/fig01_title_slide.png`). GitHub will render them inline automatically.

---

## �🗂️ Repository Structure

```
Network datascience/
│
├── 📓 notebooks/
│   ├── GPT_2_symbols_MINILAB_STUDENT-org.ipynb         # Mini-lab: GPT-2 at single temp
│   ├── GPT_2_symbols_TEMP_CHANGE.ipynb                 # GPT-2 full temperature sweep
│   ├── DistilGPT_2_symbols_TEMP_CHANGE.ipynb           # DistilGPT-2 temperature sweep
│   ├── GPT_2_symbols_MINILAB_STUDENT_gemma.ipynb       # Gemma-2-2b-it version
│   ├── gemma2_embeddings_dotproducts.ipynb             # Embedding dot products + PCA
│   ├── llm_dynamics_analyzer_ABDC (2).ipynb            # Full toy transformer (ABDC 2D)
│   ├── llm_symbol_maps_explorer_band_no_transient (1).ipynb  # Logistic map analysis
│   ├── llm_symbol_maps_explorer_LOGISTIC_MAP.ipynb     # Core logistic map explorer
│   └── LLM_Temperature_Studies (5).ipynb               # Full combined study
│
├── 🐍 scripts/
│   ├── tipping_point_general.py                        # ABDC tipping point simulation
│   └── attention_analysis.py                           # Attention head dot-product analysis
│
├── 📄 reports/
│   ├── Temperature Effects on LLM Behavior.pdf         # Main written report
│   ├── Temperature Effects on LLM Behavior.docx        # Editable report
│   ├── Empirical Attractor Periods .pdf                # Logistic map analysis
│   ├── LLM Dynamics.pdf                               # ABDC toy model writeup
│   ├── Word Embeddings.pdf                            # Embedding comparison
│   ├── Gephi_NK.pdf                                   # Co-occurrence network
│   ├── attractor_sequences_assignment.pdf             # Assignment: attractor sequences
│   ├── assignment_2_perceptrons.pdf                   # Assignment: perceptrons
│   ├── final_assignment_perceptrons (1).pdf           # Final perceptron assignment
│   └── GPT_2_symbols_TEMP_CHANGE.ipynb - Colab.pdf   # Colab notebook printout
│
├── 🎞️ presentations/
│   ├── tempeffects.pptx.pdf                           # Main slide deck (PDF export)
│   ├── Is the earth flat-neuronpedia.pptx             # Neuronpedia circuit slides
│   ├── pitch_deck_notes.docx                          # Pitch deck speaker notes
│   └── present_script.docx                            # Presentation script
│
├── �️ assets/
│   ├── attention_analysis_complete.png                # Attention visualization output
│   └── Copy of GW Research Poster Template 36 x 48 (1).jpg  # Research poster
│
├── README.md                                          # This file
├── PROJECT_SUMMARY.md                                 # Detailed module-by-module summary
├── .gitignore                                         # Excludes secrets, caches, OS files
└── .env.example                                       # Template for HF_TOKEN setup
```

---

## ⚙️ Setup & Dependencies

```bash
# Core libraries
pip install transformers==4.* torch scikit-learn matplotlib numpy scipy

# HuggingFace hub (required for Gemma and some GPT-Neo notebooks)
pip install huggingface_hub
```

#### 🔑 API Key Setup (Required for Gemma / gated models)

All hardcoded tokens have been removed. Set your HuggingFace token as an environment variable **before** running any notebook:

```bash
# Option 1 — shell (Linux/macOS)
export HF_TOKEN="your_token_here"

# Option 1 — PowerShell (Windows)
$env:HF_TOKEN = "your_token_here"

# Option 2 — .env file (copy from .env.example, never commit .env)
cp .env.example .env
# then edit .env and fill in your token
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The notebooks read it via:
```python
import os
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""))
```

#### Key Parameters (configurable in notebooks)

| Parameter | Default | Effect |
|---|---|---|
| `TEMPERATURE` | 0.75 | Controls output diversity |
| `SIM_THRESHOLD` | 0.45 | Cosine threshold for sentence clustering (lower = coarser) |
| `MAX_NEW_TOKENS` | 256–300 | Length of generated text |
| `TOP_K` | 50 | Top-k sampling |
| `SEED` | 1 | Reproducibility |
| `NGRAM_MIN/MAX` | 3/5 | Character n-gram range for TF-IDF |

---

## 🧠 Theoretical Connections

```
Logistic Map               LLM Generation
──────────────             ──────────────────────
Control param r      ←→    Temperature T
State variable xₜ   ←→    Context vector (attention-weighted embedding)
Attractor           ←→    Dominant token / repeating phrase
Period-doubling     ←→    Increasing cluster diversity in symbol stream
Chaos               ←→    High-entropy, non-repeating output
Bifurcation point   ←→    Threshold transition at T ≈ 0.55–0.70
Basin of attraction ←→    Embedding geometry (dot product landscape)
```

---

## 💡 Key Takeaways

1. **Temperature is a phase transition control** — not a smooth dial but a bifurcation parameter
2. **The logistic map and LLMs share the same attractor structure** — both exhibit 2→4→8→∞ period doubling
3. **Embedding geometry drives tipping points** — changing a single token's embedding shifts when and where the trajectory tips
4. **Larger ≠ more stable** — GPT-2 XL loses coherence faster than smaller GPT-2 at high temperature
5. **Optimal temperature ~0.55–0.70** — richest creative content with partial coherence preserved
6. **Attention is a dot-product machine** — Q·K^T scores are linear transformations of embedding similarity; temperature scales the sharpness of attention just as it scales the sharpness of output

---

## 📚 References & Inspiration

- May, R. M. (1976). *Simple mathematical models with very complicated dynamics*. Nature.
- Elman, J. L. (1990). *Finding structure in time*. Cognitive Science.
- Vaswani et al. (2017). *Attention is all you need*. NeurIPS.
- Bricken et al. (2023). *Towards monosemanticity*. Anthropic.
- Logistic map & symbolic dynamics: standard chaos theory literature
- Neuronpedia: [neuronpedia.org](https://neuronpedia.org)

---

<div align="center">

*Built with curiosity, Python, and a healthy appreciation for chaos.*

**George Washington University · Spring 2025**

</div>
