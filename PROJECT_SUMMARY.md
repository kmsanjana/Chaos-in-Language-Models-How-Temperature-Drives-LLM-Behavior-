# Project Summary: How Temperature Controls LLM Behavior
### Network Data Science — Group 6
**Team:** Dhwanil Mori · Jasreen Mehta · Sanjana Kadambe

---

## Overview

This project investigates how the **sampling temperature** parameter shapes the output dynamics of large language models (LLMs). It bridges classical dynamical systems theory (specifically the **logistic map** and **attractor theory**) with empirical behavior observed in real transformer models. The work spans five interconnected modules, each building on the previous to form a unified story about LLM output as a **dynamical system**.

---

## Module 1 — Temperature Effects on Real LLMs

### What Was Done
Four publicly available autoregressive models were tested on the prompt **"Is the Earth flat?"** across a sweep of temperatures (0.10 → 1.50). Model outputs were collected, then transformed into **symbolic streams** using a TF-IDF character n-gram clustering pipeline.

### Models Tested
| Model | Size |
|---|---|
| GPT-2 | ~117M params |
| DistilGPT-2 | ~82M params |
| GPT-2 XL | ~1.5B params |
| EleutherAI/GPT-Neo-1.3B | ~1.3B params |

### Symbolic Encoding Pipeline
Each generated response was processed as follows:
1. **Split** into sentences using regex (`[.!?]` + whitespace or newlines)
2. **Vectorize** each sentence with TF-IDF character n-grams (range 3–5)
3. **Cluster** using greedy cosine similarity (threshold = 0.45–0.55); each cluster gets a letter label (A, B, C…)
4. **Output** a symbol stream (e.g., `ABBBBBBBBBB`, `ABCDEFGHIJK`)
5. **Visualize** as a color-coded barcode per temperature

### Dynamics Classification
Each symbol stream was classified into one of three regimes:
- **Fixed Attractor** — one symbol dominates (majority fraction ≥ 0.60, max run ≥ 4)
- **Oscillatory** — rapid switching (≥6 switches, mean run ≤ 1.5)
- **Mixed** — neither purely fixed nor oscillatory

### Key Findings by Model

**GPT-2**
- T=0.10: Tight `ABABAB` oscillation ("Yes it is" / "The earth is flat" repeating)
- T=0.32: Single-symbol dominance (`ABBBB…`) — near fixed attractor
- T=0.50: Heavy repetition of one phrase
- T=0.55: First hallucinations ("radius", "divisions") — creative zone entry
- T=0.70: Contradictory, multi-clause explanations (layers, hydrothermal water)
- T=0.78: Surreal imagery (polar bears, horses) — creative but incoherent
- T=1.00: Highly diverse, cosmic/religious ideas

**DistilGPT-2**
- More conservative than GPT-2 throughout
- Slower transition to diverse outputs; smaller creativity jumps
- T=1.00: Finally reaches multiple clusters with varied topics

**GPT-2 XL**
- T=0.10–0.50: Locks firmly into `"No. The earth is round."` loop
- T=0.55: Sudden explosion of variety — many short distinct clusters
- T=0.85: Numeric hallucinations, drifting topics
- T=1.30–1.50: Long, disorganized philosophical rambling; coherence lost

**GPT-Neo 1.3B**
- Prone to self-contradiction in same response (flat vs. sphere)
- T=0.55: Confidently states Earth's diameter ≈ 180 miles (fabricated)
- T=1.00: Meta/tangential replies; hypothetical reasoning
- T=1.30+: Conspiracy-like, rambling narratives

### Cross-Model Insights
| Temperature Zone | Behavior |
|---|---|
| 0.10 – 0.32 | Deterministic, repetitive, "safe" — fixed attractors |
| 0.50 – 0.70 | Creativity emerges; coherence weakens; hallucinations begin |
| 0.85 – 1.50 | Highly varied, imaginative, factually unreliable |

- **Threshold behavior**: Sharp behavioral transition between T=0.55–0.70 in all models
- **Size ≠ Stability**: GPT-2 XL (largest) is not most coherent — it becomes verbose and loses logic
- **Optimal zone**: ~T=0.55–0.70 yields the richest "interesting" output while partially preserving coherence

---

## Module 2 — Symbolic Dynamics & the Logistic Map

### What Was Done
The empirical attractor patterns observed in LLM outputs were mapped onto the classical **logistic map**, validating that the same mathematical structure underlies both systems.

### The Logistic Map
```
x_{t+1} = r × x_t × (1 − x_t)
```
- `r` is the control parameter (analogous to LLM temperature `T`)
- `x_t ∈ [0,1]` is the state variable
- Exhibits the canonical **period-doubling route to chaos**

### Symbolic Encoding Setup
- A-band defined around 0.5: `τ_L = 0.48`, `τ_U = 0.52`
- Values in band → symbol **A**; outside → symbols B, C, D (by rank within cycle)
- Parameter sweep: `r ∈ [3.40, 4.00]`, 220 steps
- Per `r`: 30 random initial conditions, discard 900 transient steps, analyze 240-step tail window

### Period-Doubling Cascade Recovered
| r Range | Period | Symbolic Sequence |
|---|---|---|
| 2.9–3.0 | 1 | A… |
| 3.0–3.449 | 2 | AB… |
| ~3.5 | 4 | ACBD… |
| ~3.83 | 3 (window) | ABC… |
| >3.57 | ∞ (chaos) | BBABABAB… |

### Key Findings
- Period-doubling cascade **2 → 4 → 8 → ∞** fully recovered
- Periodic windows of period 3, 5, 6 embedded within the chaotic regime
- "Fraction of runs" structure emerges naturally from multiple initial conditions
- Band placement is sensitive: widen `[τ_L, τ_U]` → A merges with others; narrow too much → A signal lost
- **Conclusion**: Increasing `r` (≈ temperature) drives system through the same attractor transitions observed empirically in LLMs

---

## Module 3 — LLM Dynamics Toy Model (ABDC)

### What Was Done
A minimal **2D transformer** with 4-token vocabulary {A, B, C, D} was built from scratch to visualize how attention dynamics create trajectories through embedding space and how **tipping points** emerge.

### Toy Model Architecture
- Vocabulary: 4 tokens in 2D embedding space
- Architecture: Transformer decoder (multi-head attention + feed-forward + layer norm + residual)
- Weight options: Identity matrices (for interpretability) or random Xavier initialization
- Output projection: embedding matrix transpose → logits → softmax

### Token Embeddings
| Token | Vector | Role |
|---|---|---|
| A | [0.4, −0.3] | Initial seed / temporary attractor |
| B | [0.8, 0.0] | Mid-range influencer |
| C | [−0.2, −0.2] | Neutral / background |
| D | [0.9, 0.5] | Strong final attractor |

### Observed Trajectory (Baseline)
Starting from `A`, the model generates:
```
A → B → B → B → D → D → D → D → D → D → D → D → D → D → D …
```
- Sequence settles into the **D attractor** after 4 BBB tokens
- D has the highest logit once the context vector aligns toward it
- Final entropy = 0.000 (deterministic convergence)

### Tipping Point Experiment
**Baseline** (`e_B = [0.8, 0.0]`):
- Context quickly drifts from origin toward D embedding
- Tips into DDD attractor after 6 BBB tokens

**Modified** (`e_B = [0.88, 0.0]`):
- Context stays aligned with BBB embedding for much longer
- Path circles around B before eventually drifting to D
- Result: longer BBB-run before tipping; tipping point delayed

### Insight
Modifying a single token's embedding changes the **geometry of the basin of attraction**. The B token acts as a temporary attractor; strengthening its pull delays the inevitable convergence to D. This demonstrates that **embedding geometry controls tipping dynamics**.

### Scripts/Notebooks
- `tipping_point_general.py` — simulation + animation of ABDC trajectory
- `llm_dynamics_analyzer_ABDC (2).ipynb` — full transformer implementation with visualization

---

## Module 4 — Word Embeddings: GPT-2 vs Gemma-2-2b-it

### What Was Done
A comparative analysis of word embedding spaces between two models — GPT-2 (raw pretrained) and Gemma-2-2b-it (instruction-tuned) — using dot products and PCA.

### Words Analyzed
`privacy`, `important`, `overrated`

### Dot Products (High-Dimensional Space)
| Pair | GPT-2 / Gemma |
|---|---|
| privacy · privacy | 2.90 |
| important · important | 2.53 |
| overrated · overrated | 3.55 |
| privacy · overrated | 0.81 |
| important · overrated | 0.60 |
| privacy · important | 0.33 |

### PCA 2D Visualization
- **Gemma-2-2b-it**: `privacy` and `important` appear semantically close; smoother, cohesive space
- **GPT-2**: Words are more scattered, weaker clustering of related meanings

### Comparison Table
| Aspect | Gemma-2-2b-it | GPT-2 |
|---|---|---|
| Embedding Structure | Smooth, cohesive; privacy ≈ important | Scattered; weaker clustering |
| Context Understanding | Deep contextual relationships (instruction tuned) | Raw co-occurrence from web text |
| Semantic Alignment | Groups by shared value/intent | Groups by surface usage |

### Additional: Attention Head Analysis (`attention_analysis.py`)
- Built a single attention head (d_model=768, d_k=64) to compare embedding dot products vs Q·K^T scores
- Finding: Attention Q·K^T is a **scaled linear transformation** of embedding dot products
- With identity-like W_q, W_k: `Q·K^T ≈ scaled X·X^T`
- Low temperature → sharp/concentrated attention (analogous to fixed-point attractor)
- High temperature → distributed attention (analogous to chaotic regime)

---

## Module 5 — Neighborhood Stories Network (Gephi)

### What Was Done
A symbolic co-occurrence network was built from a small GPT-2 generated corpus and visualized in Gephi.

### Corpus
10 community-focused vignettes (sunflowers, lanterns, jazz, night markets, etc.) generated by GPT-2.

### Network Statistics
- **84 unique words** (nodes)
- **354 co-occurrence links** (edges)
- Each node = a symbol (letter) representing a unique word
- Each edge = two words appeared adjacent in a symbolic sequence
- Edge thickness/darkness = co-occurrence frequency

### Purpose
Demonstrates how symbolic sequential graphs can visualize the **structural patterns** in LLM output, connecting the symbolic dynamics framework to network science tools like Gephi.

---

## Module 6 — Mechanistic Interpretability (Neuronpedia)

### What Was Done
Circuit-level analysis of GPT-2 was performed using Neuronpedia to understand which circuits activate on specific prompts.

### Prompts Analyzed
1. **"Is the Earth flat?"** — identifies circuits responsible for factual recall, denial, and repetitive output
2. **"Vaccines — bad?"** — identifies circuits activating on sensitive/controversial content

### Connection to Main Research
This module grounds the symbolic dynamics findings in mechanistic interpretability: the repetitive attractors and tipping points observed at the behavioral level correspond to identifiable activation circuits at the neural level.

---

## Codebase Summary

| File | Purpose |
|---|---|
| `GPT_2_symbols_MINILAB_STUDENT-org.ipynb` | Original student mini-lab: GPT-2 symbol stream at single temperature |
| `GPT_2_symbols_TEMP_CHANGE.ipynb` | Temperature sweep for GPT-2 (T=0.10 to 1.00); barcode plots |
| `DistilGPT_2_symbols_TEMP_CHANGE.ipynb` | Same pipeline applied to DistilGPT-2 |
| `GPT_2_symbols_MINILAB_STUDENT_gemma.ipynb` | Gemma-2-2b-it version of the mini-lab |
| `gemma2_embeddings_dotproducts.ipynb` | Word embedding dot products + PCA for GPT-2 vs Gemma |
| `llm_dynamics_analyzer_ABDC (2).ipynb` | Full toy transformer implementation (ABDC 2D model) |
| `llm_symbol_maps_explorer_band_no_transient (1).ipynb` | Logistic map symbolic analysis (no-transient version) |
| `attractor_sequence_code_files/llm_symbol_maps_explorer_LOGISTIC_MAP.ipynb` | Core logistic map attractor explorer |
| `attention_analysis.py` | Attention head dot product + temperature pattern analysis |
| `tipping_point_general.py` | ABDC tipping point simulation with animation |

---

## Unified Narrative

All five modules converge on a single insight:

> **LLM token generation is a dynamical system. Temperature is the control parameter. Like the logistic map, LLMs exhibit fixed-point attractors at low temperature, periodic oscillations at moderate temperature, and chaotic/creative behavior at high temperature. The tipping point between regimes is sharp, predictable, and can be located both empirically and theoretically.**

The geometric structure of the embedding space determines the attractor landscape. Modifying embeddings shifts the tipping point. The same period-doubling cascade that appears in the logistic map appears in the symbolic output of real GPT models. Attention dot products are the engine that drives trajectories through this space.
