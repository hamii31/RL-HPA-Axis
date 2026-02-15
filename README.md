# HPA Axis Simulator using Reinforcement Learning with Curriculum Training

Models the Hypothalamic-Pituitary-Adrenal axis as an RL agent that learns to maintain hormonal homeostasis while responding to stress through developmental stages (child → adolescent → adult).

The agent learns to regulate CRH, ACTH, and cortisol levels through experience, minimizing allostatic load (cumulative biological damage) rather than maximizing arbitrary reward points.

There are two implementations of this project - one in Python and one in C. The one in Python is a generally balanced implementation, mixing performance and simplicity. It comes with insightful visualizations, that
I have not implemented in the C variation. The C implementation leans toward performance, allowing for longer training periods that yield more accurate and realistic results.

---

## Key Features

1. **Literature-based physiological parameters** (half-lives, secretion rates)
2. **Ultradian pulses** (90-minute oscillations superimposed on circadian rhythm)
3. **Gland mass dynamics** (weeks-long adaptation to chronic stress)
4. **Dual receptor system** (MR and GR with different affinities)
5. **Extended time scale** for chronic stress simulation (10 days per adult episode)
6. **Curriculum learning** through developmental stages
7. **Biological cost function** (allostatic load) instead of arbitrary rewards

---

## Reward Framework: Allostatic Load (Biological Cost)

The agent learns by **MINIMIZING allostatic load** (cumulative wear and tear):

```python
reward = -allostatic_load + 5.0
```

Where `allostatic_load` is the sum of biological costs per time step.

**This means:**
- Maximizing reward ≡ Minimizing biological damage
- Positive scores = Healthier than baseline
- Negative scores = Accumulated damage exceeding baseline
- Scores map directly to clinical health states

### Allostatic Load Components (per time step)

1. **BASAL METABOLIC COST** (~0.05/step)
   - Minimum energy to maintain HPA axis

2. **CORTISOL DEVIATION COST** (Quadratic)
   - Optimal: 15 μg/dL, Tolerance: ±7 μg/dL
   - Formula: `0.5 × ((deviation - 7) / 7)²`

3. **TISSUE-SPECIFIC DAMAGE** (Dose-dependent)
   - **Hypercortisolism** (>25 μg/dL): `excess × 0.3`
     - Hyperglycemia, muscle wasting, bone loss, immune suppression
   - **Hypocortisolism** (<5 μg/dL): `deficit × 0.7`
     - Hypotension, hypoglycemia, inflammation

4. **ACTH DYSREGULATION** (~0.02 × deviation²)
5. **CRH DYSREGULATION** (~0.01 × deviation²)
6. **RECEPTOR DYSFUNCTION** (MR/GR occupancy costs)
7. **GLAND PATHOLOGY** (hypertrophy/atrophy)
8. **INSTABILITY COST** (variance-based)
9. **STRESS RESPONSE APPROPRIATENESS**

#### Total Load Ranges

| Load/Step | Health State |
|-----------|-------------|
| 0.1 - 0.3 | Excellent regulation |
| 0.3 - 1.0 | Good regulation |
| 1.0 - 2.0 | Acceptable |
| 2.0 - 4.0 | Mild dysfunction |
| 4.0 - 7.0 | Moderate dysfunction |
| 7.0 - 15.0 | Severe dysregulation |
| >15.0 | Crisis state |

---

## Curriculum Training Stages

The agent learns through three developmental stages, mimicking HPA maturation:

### Stage 1: CHILD (Learning Basics)
- **Episode length:** 24 hours (240 steps)
- **Feedback maturity:** 40% (weak negative feedback)
- **Receptor sensitivity:** 60%
- **Training episodes:** 100
- **Expected scores:** +50 to +150

### Stage 2: ADOLESCENT (Intermediate)
- **Episode length:** 72 hours (720 steps) = 3 days
- **Feedback maturity:** 90% (near-adult)
- **Receptor sensitivity:** 95%
- **Training episodes:** 150
- **Expected scores:** +200 to +500

### Stage 3: ADULT (Full Complexity)
- **Episode length:** 240 hours (2400 steps) = 10 days
- **Feedback maturity:** 100%
- **Receptor sensitivity:** 100%
- **Training episodes:** 200
- **Expected scores:** +5000 to +12000

### Transfer Learning
- Q-table persists across stages (knowledge accumulates)
- Exploration rate (epsilon) partially resets between stages
- Agent builds on previous learning
- **Total training:** 450 episodes across all stages

---

## Interpreting Scores

```
Scores = Cumulative Reward = Σ(5.0 - allostatic_load) over all steps
```

### Per-Step Interpretation

| Reward/Step | Load/Step | Health State |
|-------------|-----------|--------------|
| +4.8 | 0.2 | Excellent (minimal damage) |
| +4.0 | 1.0 | Good (low allostatic load) |
| +3.0 | 2.0 | Acceptable (manageable) |
| +2.0 | 3.0 | Mild dysfunction |
| +1.0 | 4.0 | Moderate dysfunction |
| 0.0 | 5.0 | Baseline (break-even) |
| -1.0 | 6.0 | Poor regulation |
| -3.0 | 8.0 | Severe dysregulation |
| -5.0 | 10.0 | Crisis-level damage |

### Adult Stage Score Ranges (2400 steps)

| Score Range | Load/Step | Clinical State |
|-------------|-----------|----------------|
| +10000 to +12000 | ~0.2 | Optimal health |
| +7000 to +10000 | ~0.5 | Strong regulation |
| +4000 to +7000 | ~1.0 | Decent homeostasis |
| +2000 to +4000 | ~1.5 | Manageable |
| 0 to +2000 | ~2.5 | Frequent deviation |
| -2000 to 0 | ~3.0 | Chronic mild dysfunction |
| -5000 to -2000 | ~4.0 | Significant dysregulation |
| < -5000 | >5.0 | Pathological state |

### Clinical Mapping Examples

**Score: +10000** (load ~0.2/step)
- Healthy individual with robust HPA axis
- Appropriate stress responses, quick recovery
- Minimal tissue damage accumulation

**Score: 0** (load ~5.0/step)
- Baseline health threshold
- Break-even point between health and disease

**Score: -5000** (load ~7.0/step)
- Clinical dysfunction
- Similar to mild Cushing's or adrenal insufficiency
- Requires medical intervention

**Score: -10000** (load ~9.0/step)
- Severe pathological state
- Major endocrine disorder
- Emergency medical treatment needed

---

## What Negative Scores Mean

Negative scores indicate cumulative damage exceeding baseline threshold.

**Example:** Score = -2400 over 10 days
- Reward per step: -1.0/step
- Load per step: 6.0/step
- Total load: 14,400 units

**Clinical interpretation:**
- Chronic moderate dysregulation
- Equivalent to subclinical Cushing's for 10 days
- Symptoms: Hyperglycemia, hypertension, immune suppression, mood disturbances

**Note:** This is typical for an **untrained agent** (random actions).  
A well-trained agent achieves positive scores (+8000 to +11000).

---

## Typical Training Progression

### Child Stage (Episodes 1-100)
- Early (1-30): -50 to +50 (exploring)
- Mid (31-70): +50 to +100 (improving)
- Late (71-100): +100 to +150 (decent regulation)

### Adolescent Stage (Episodes 101-250)
- Early (101-150): +100 to +300 (adapting)
- Mid (151-200): +300 to +400 (building knowledge)
- Late (201-250): +350 to +500 (mastering 3-day episodes)

### Adult Stage (Episodes 251-450)
- Early (251-300): +1000 to +4000 (transfer learning)
- Mid (301-400): +4000 to +8000 (refining policy)
- Late (401-450): +8000 to +11000 (near-optimal)

**Final test performance:**  
Expected adult test scores: +9000 to +10500 (±200)

---

## Installation & Usage

```bash
# Install dependencies
pip install numpy matplotlib --break-system-packages

# Run
python hpa.py
```

---

## Technical Details

### Physiological Parameters (Literature-Based)

| Parameter | Value | Source |
|-----------|-------|--------|
| Cortisol half-life | 75 min | [1, 2] |
| ACTH half-life | 10 min | [1, 2] |
| CRH half-life | 15 min | [1, 2] |
| Cortisol daily secretion | 20-30 mg/day | [3] |
| Cortisol serum levels | 5-24 μg/dL | [3] |
| Circadian peak | 07:00-08:00 AM | [4, 5] |
| Circadian nadir | 02:00-04:00 AM | [4, 5] |
| Ultradian period | 60-90 min | [6] |
| MR Kd (affinity) | 0.5 nM | [7, 8] |
| GR Kd (affinity) | 5.0 nM | [7, 8] |

### Reinforcement Learning

- **Algorithm:** Deep Q-Network (DQN) with Q-learning
- **State space:** 12 features (hormones, receptors, glands, time, stress)
- **Action space:** 9 discrete actions (3×3 hormone modulation combinations)
- **Exploration:** ε-greedy with decay (1.0 → 0.01)
- **Learning rate:** 0.0005
- **Discount factor (γ):** 0.98
- **Batch size:** 128
- **Memory:** 10,000 experience replay buffer

---

## Scientific References

### HPA Axis Physiology & Regulation

1. **Smith, S.M., & Vale, W.W. (2006).** The role of the hypothalamic-pituitary-adrenal axis in neuroendocrine responses to stress. *Dialogues in Clinical Neuroscience*, 8(4), 383-395.  
   [PMC1828259](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1828259/)

2. **Herman, J.P., McKlveen, J.M., Ghosal, S., Kopp, B., Wulsin, A., Makinson, R., Scheimann, J., & Myers, B. (2016).** Regulation of the hypothalamic-pituitary-adrenocortical stress response. *Comprehensive Physiology*, 6(2), 603-621.  
   [PMC4867107](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4867107/)

3. **Nicolaides, N.C., Kyratzi, E., Lamprokostopoulou, A., Chrousos, G.P., & Charmandari, E. (2015).** Stress, the stress system and the role of glucocorticoids. *Neuroimmunomodulation*, 22(1-2), 6-19.  
   [Abstract](https://pubmed.ncbi.nlm.nih.gov/25227402/)

### Circadian & Ultradian Rhythms

4. **Kritikou, I., Basta, M., Vgontzas, A.N., Pejovic, S., Liao, D., Tsaoussoglou, M., Bixler, E.O., Stefanakis, Z., & Chrousos, G.P. (2016).** Sleep apnoea and the hypothalamic-pituitary-adrenal axis in men and women: effects of continuous positive airway pressure. *European Respiratory Journal*, 47(2), 531-540.  
   [PMC7830980](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7830980/)

5. **Debono, M., Ghobadi, C., Rostami-Hodjegan, A., Huatan, H., Campbell, M.J., Newell-Price, J., Darzy, K., Merke, D.P., Arlt, W., & Ross, R.J. (2009).** Modified-release hydrocortisone to provide circadian cortisol profiles. *Journal of Clinical Endocrinology & Metabolism*, 94(5), 1548-1554.  
   [PMC3475279](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3475279/)

6. **Chung, S., Son, G.H., & Kim, K. (2011).** Circadian rhythm of adrenal glucocorticoid: Its regulation and clinical implications. *Biochimica et Biophysica Acta (BBA) - Molecular Basis of Disease*, 1812(5), 581-591.  
   [PMC8813037](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8813037/)

7. **Lightman, S.L., Wiles, C.C., Atkinson, H.C., Henley, D.E., Russell, G.M., Leendertz, J.A., McKenna, M.A., Spiga, F., Wood, S.A., & Conway-Campbell, B.L. (2008).** The significance of glucocorticoid pulsatility. *European Journal of Pharmacology*, 583(2-3), 255-262.  
   [Abstract](https://pubmed.ncbi.nlm.nih.gov/18339373/)

### Glucocorticoid Receptors & Feedback

8. **de Kloet, E.R., Joëls, M., & Holsboer, F. (2005).** Stress and the brain: from adaptation to disease. *Nature Reviews Neuroscience*, 6(6), 463-475.  
   [Abstract](https://pubmed.ncbi.nlm.nih.gov/15891777/)

9. **Groeneweg, F.L., Karst, H., de Kloet, E.R., & Joëls, M. (2012).** Mineralocorticoid and glucocorticoid receptors at the neuronal membrane, regulators of nongenomic corticosteroid signalling. *Molecular and Cellular Endocrinology*, 350(2), 299-309.  
   [Abstract](https://pubmed.ncbi.nlm.nih.gov/21736918/)

10. **Spiga, F., Walker, J.J., Gupta, R., Terry, J.R., & Lightman, S.L. (2015).** Role of glucocorticoid negative feedback in the regulation of HPA axis pulsatility. *Stress*, 18(4), 403-416.  
    [PMC6220752](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6220752/)

### Mathematical Modeling of HPA Axis

11. **Rao, R., DuBois, D., Almon, R., Jusko, W.J., & Androulakis, I.P. (2016).** Mathematical modeling of the circadian dynamics of the neuroendocrine-immune network in experimentally induced arthritis. *American Journal of Physiology-Endocrinology and Metabolism*, 311(2), E310-E324.  
    [Abstract](https://pubmed.ncbi.nlm.nih.gov/27245335/)

12. **Bangsgaard, E.O., Ottesen, J.T., Sturis, J., & Pedersen, M.G. (2020).** A new model for the HPA axis explains dysregulation of stress hormones on the timescale of weeks. *PLoS Computational Biology*, 16(7), e1007572.  
    [PMC7364861](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7364861/)

### Allostatic Load Theory

13. **McEwen, B.S. (1998).** Stress, adaptation, and disease: Allostasis and allostatic load. *Annals of the New York Academy of Sciences*, 840(1), 33-44.  
    [Abstract](https://pubmed.ncbi.nlm.nih.gov/9629234/)

14. **McEwen, B.S., & Stellar, E. (1993).** Stress and the individual: Mechanisms leading to disease. *Archives of Internal Medicine*, 153(18), 2093-2101.  
    [Abstract](https://pubmed.ncbi.nlm.nih.gov/8379800/)

15. **Juster, R.P., McEwen, B.S., & Lupien, S.J. (2010).** Allostatic load biomarkers of chronic stress and impact on health and cognition. *Neuroscience & Biobehavioral Reviews*, 35(2), 2-16.  
    [Abstract](https://pubmed.ncbi.nlm.nih.gov/19822172/)

---

## Applications

### Research
- Understanding chronic stress pathophysiology
- Modeling endocrine disorders (Cushing's, Addison's, PTSD)
- Testing therapeutic interventions *in silico*
- Drug development and screening

### Clinical
- Personalized medicine (fit model to patient data)
- Treatment response prediction
- Disease progression modeling
- Medical education and training

### Computational
- Benchmark for physiologically-informed RL
- Transfer learning in biological systems
- Curriculum learning case study

---

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{hpa_axis_rl_simulator,
  title = {HPA Axis Simulator using Reinforcement Learning with Curriculum Training},
  author = {[Hami Ibriyamov]},
  year = {2026},
  url = {[https://github.com/hamii31/RL-HPA-Axis]},
  note = {Clinically-accurate HPA axis model with allostatic load framework}
}
```

---

## Acknowledgments

This work builds upon established neuroendocrinology research and allostatic load theory. Physiological parameters are derived from peer-reviewed literature (see References). The curriculum learning approach is inspired by developmental neuroscience and transfer learning in machine learning.

---

## Contact

[Author](https://github.com/hamii31)

---

**Expected Final Performance:**
- Adult test scores: +9000 to +10500
- Allostatic load: ~0.2-0.4 per step
- Clinical state: Excellent health, robust HPA axis regulation
