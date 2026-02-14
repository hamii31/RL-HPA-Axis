# HPA Axis Simulator using Reinforcement Learning with Curriculum Training
========================================================================

Models the Hypothalamic-Pituitary-Adrenal axis as an RL agent that learns
to maintain hormonal homeostasis while responding to stress through
developmental stages (child → adolescent → adult).

The agent learns to regulate CRH, ACTH, and cortisol levels through experience,
minimizing allostatic load (cumulative biological damage) rather than maximizing
arbitrary reward points.

Key Features:
1. Literature-based physiological parameters (half-lives, secretion rates)
2. Ultradian pulses (90-minute oscillations superimposed on circadian rhythm)
3. Gland mass dynamics (weeks-long adaptation to chronic stress)
4. Dual receptor system (MR and GR with different affinities)
5. Extended time scale for chronic stress simulation (10 days per adult episode)
6. Curriculum learning through developmental stages
7. Biological cost function (allostatic load) instead of arbitrary rewards

================================================================================
REWARD FRAMEWORK: ALLOSTATIC LOAD (BIOLOGICAL COST)
================================================================================

The agent learns by MINIMIZING allostatic load (cumulative wear and tear):

    reward = -allostatic_load + 5.0

Where allostatic_load is the sum of biological costs per time step.

This means:
- Maximizing reward ≡ Minimizing biological damage
- Positive scores = Healthier than baseline
- Negative scores = Accumulated damage exceeding baseline
- Scores map directly to clinical health states

--------------------------------------------------------------------------------
Allostatic Load Components (per time step):
--------------------------------------------------------------------------------

1. BASAL METABOLIC COST (~0.05/step)
   - Minimum energy to maintain HPA axis
   - Always present (like cellular respiration)

2. CORTISOL DEVIATION COST (Quadratic)
   - Optimal: 15 μg/dL (cost ≈ 0.01)
   - Tolerance: ±7 μg/dL (minimal cost in this range)
   - Outside tolerance: cost = 0.5 * ((deviation - 7) / 7)²
   
   Examples:
   • Cortisol = 15 μg/dL (optimal):  cost ≈ 0.01
   • Cortisol = 20 μg/dL (+5):       cost ≈ 0.05
   • Cortisol = 25 μg/dL (+10):      cost ≈ 0.5
   • Cortisol = 35 μg/dL (+20):      cost ≈ 3.5

3. TISSUE-SPECIFIC DAMAGE (Dose-dependent)
   
   Hypercortisolism (>25 μg/dL):
   • Metabolic damage: (excess) * 0.3
     - Hyperglycemia: 0.1 per μg/dL
     - Muscle wasting: 0.05 per μg/dL
     - Bone loss: 0.05 per μg/dL
     - Immune suppression: 0.1 per μg/dL
   • Crisis risk (>35 μg/dL): +2-5 exponential term
   
   Hypocortisolism (<5 μg/dL):
   • Crisis damage: (deficit) * 0.7
     - Hypotension: 0.2 per μg/dL
     - Hypoglycemia: 0.3 per μg/dL
     - Inflammation: 0.2 per μg/dL
   • Addisonian crisis (<2 μg/dL): +5 exponential term

4. ACTH DYSREGULATION (~0.02 * deviation²)
   - Optimal: 25 pg/mL, Tolerance: ±15 pg/mL

5. CRH DYSREGULATION (~0.01 * deviation²)
   - Optimal: 100 pg/mL, Tolerance: ±50 pg/mL

6. RECEPTOR DYSFUNCTION
   - MR occupancy cost: 0.5 * (occupancy - 0.8)²
   - GR occupancy cost: 0.3 * (occupancy - target)²
     (target = 0.3 at baseline, 0.7 during stress)
   - Receptor downregulation: 0.5 * (1 - receptor_density)²

7. GLAND PATHOLOGY
   - Adrenal hypertrophy/atrophy: 0.3 * (mass - 1.0)²
   - Pituitary changes: 0.3 * (mass - 1.0)²
   - Severe pathology (>50% change): 3 * multiplier

8. INSTABILITY COST
   - Low variance (<25): cost ≈ 0.0
   - High variance (>50): cost ≈ (variance - 25) / 100

9. STRESS RESPONSE APPROPRIATENESS
   - During stress: cortisol should match (20 + stress_level * 2)
   - Response error: 0.5 * (error / 10)²

--------------------------------------------------------------------------------
Total Allostatic Load per Step:
--------------------------------------------------------------------------------

Typical ranges:
• Excellent regulation:     0.1 - 0.3 load/step
• Good regulation:          0.3 - 1.0 load/step
• Acceptable:               1.0 - 2.0 load/step
• Mild dysfunction:         2.0 - 4.0 load/step
• Moderate dysfunction:     4.0 - 7.0 load/step
• Severe dysregulation:     7.0 - 15.0 load/step
• Crisis state:             >15.0 load/step

================================================================================
CURRICULUM TRAINING STAGES
================================================================================

The agent learns through three developmental stages, mimicking HPA maturation:

--------------------------------------------------------------------------------
STAGE 1: CHILD (Learning Basics)
--------------------------------------------------------------------------------
Episode length:     24 hours (240 steps at 0.1 hr/step)
Time step:          6 minutes
Feedback maturity:  40% (weak negative feedback)
Receptor sensitivity: 60%
Stress resilience:  50% (high vulnerability)
Training episodes:  100

Purpose: Learn fundamental hormone regulation without complexity
Expected scores: +50 to +150 per episode

--------------------------------------------------------------------------------
STAGE 2: ADOLESCENT (Intermediate Complexity)
--------------------------------------------------------------------------------
Episode length:     72 hours (720 steps) = 3 days
Time step:          6 minutes
Feedback maturity:  90% (near-adult feedback)
Receptor sensitivity: 95%
Stress resilience:  85%
Training episodes:  150

Purpose: Handle longer episodes, stronger feedback, more stress events
Expected scores: +200 to +500 per episode

--------------------------------------------------------------------------------
STAGE 3: ADULT (Full Complexity)
--------------------------------------------------------------------------------
Episode length:     240 hours (2400 steps) = 10 days
Time step:          6 minutes
Feedback maturity:  100% (full feedback strength)
Receptor sensitivity: 100%
Stress resilience:  100%
Training episodes:  200

Purpose: Master full HPA complexity with chronic adaptation
Expected scores: +5000 to +12000 per episode

--------------------------------------------------------------------------------
Transfer Learning:
--------------------------------------------------------------------------------
• Q-table persists across stages (knowledge accumulates)
• Exploration rate (epsilon) partially resets between stages
• Agent builds on previous learning rather than starting from scratch
• Total training: 450 episodes across all stages

================================================================================
INTERPRETING SCORES
================================================================================

Scores = Cumulative Reward = Σ(5.0 - allostatic_load) over all steps

--------------------------------------------------------------------------------
Per-Step Interpretation:
--------------------------------------------------------------------------------

reward_per_step = 5.0 - load

Reward/Step  |  Load/Step  |  Health State
-------------|-------------|----------------------------------------
  +4.8       |    0.2      |  Excellent (minimal damage)
  +4.0       |    1.0      |  Good (low allostatic load)
  +3.0       |    2.0      |  Acceptable (manageable stress)
  +2.0       |    3.0      |  Mild dysfunction
  +1.0       |    4.0      |  Moderate dysfunction
   0.0       |    5.0      |  Baseline (break-even point)
  -1.0       |    6.0      |  Poor regulation
  -3.0       |    8.0      |  Severe dysregulation
  -5.0       |   10.0      |  Crisis-level damage

--------------------------------------------------------------------------------
Episode Score Ranges by Stage:
--------------------------------------------------------------------------------

CHILD STAGE (240 steps):
  +200 to +250:    Excellent regulation
  +100 to +200:    Good control
  +50 to +100:     Acceptable, learning
  0 to +50:        Struggling
  < 0:             Poor regulation

ADOLESCENT STAGE (720 steps):
  +700 to +900:    Excellent regulation
  +400 to +700:    Good control
  +200 to +400:    Acceptable
  0 to +200:       Struggling
  < 0:             Poor regulation

ADULT STAGE (2400 steps):
  +10000 to +12000:  Excellent (load ~0.2/step) - Optimal health
  +7000 to +10000:   Very good (load ~0.5/step) - Strong regulation
  +4000 to +7000:    Good (load ~1.0/step) - Decent homeostasis
  +2000 to +4000:    Acceptable (load ~1.5/step) - Manageable
  0 to +2000:        Borderline (load ~2.5/step) - Frequent deviation
  -2000 to 0:        Poor (load ~3.0/step) - Chronic mild dysfunction
  -5000 to -2000:    Bad (load ~4.0/step) - Significant dysregulation
  < -5000:           Severe (load >5.0/step) - Pathological state

--------------------------------------------------------------------------------
Clinical Mapping of Adult Scores:
--------------------------------------------------------------------------------

Score: +10000 (load ~0.2/step)
→ Healthy individual with robust HPA axis
→ Appropriate stress responses, quick recovery
→ Minimal tissue damage accumulation
→ Normal cortisol circadian rhythm maintained

Score: +5000 (load ~1.0/step)
→ Generally healthy with occasional dysregulation
→ May have mild metabolic changes under chronic stress
→ Slightly elevated disease risk long-term

Score: 0 (load ~5.0/step)
→ Baseline health threshold
→ Break-even point between health and disease
→ Accumulated damage equals baseline

Score: -2500 (load ~6.0/step)
→ Subclinical dysfunction
→ Equivalent to chronic mild hypercortisolism
→ Metabolic syndrome risk, mild immune suppression
→ Weight gain, mood changes likely

Score: -5000 (load ~7.0/step)
→ Clinical dysfunction
→ Overt symptoms of dysregulation
→ Similar to mild Cushing's or adrenal insufficiency
→ Requires medical intervention

Score: -10000 (load ~9.0/step)
→ Severe pathological state
→ Major endocrine disorder equivalent
→ High crisis risk, organ damage
→ Emergency medical treatment needed

================================================================================
WHAT NEGATIVE SCORES MEAN
================================================================================

Negative scores indicate cumulative damage exceeding the baseline threshold.
This is REALISTIC and INTERPRETABLE:

Example: Score = -2400 over 10 days (adult stage)
  • Reward per step: -2400 / 2400 = -1.0/step
  • Load per step: 5.0 - (-1.0) = 6.0/step
  • Total load accumulated: 6.0 * 2400 = 14,400 units
  
Clinical interpretation:
  • Chronic moderate dysregulation
  • Sustained hypercortisolism or hypocortisolism
  • Equivalent to subclinical Cushing's for 10 days
  • Would manifest as:
    - Hyperglycemia, hypertension
    - Immune suppression
    - Mood disturbances
    - Weight changes
    - Increased disease risk

This is what happens with an UNTRAINED agent (random actions)!
A well-trained agent achieves positive scores (+8000 to +11000).

================================================================================
TYPICAL TRAINING PROGRESSION
================================================================================

Child Stage (Episodes 1-100):
  Early (1-30):      -50 to +50    (exploring, learning basics)
  Mid (31-70):       +50 to +100   (improving control)
  Late (71-100):     +100 to +150  (decent regulation)

Adolescent Stage (Episodes 101-250):
  Early (101-150):   +100 to +300  (adapting to longer episodes)
  Mid (151-200):     +300 to +400  (building on child knowledge)
  Late (201-250):    +350 to +500  (mastering 3-day episodes)

Adult Stage (Episodes 251-450):
  Early (251-300):   +1000 to +4000   (transfer learning, exploring)
  Mid (301-400):     +4000 to +8000   (refining policy)
  Late (401-450):    +8000 to +11000  (near-optimal regulation)

Final test performance (adult stage, greedy policy):
  Expected: +9000 to +10500 with low variance (±200)

================================================================================
SUMMARY
================================================================================

This HPA axis simulator uses:

1. BIOLOGICAL COST FUNCTION
   - Allostatic load = cumulative wear and tear
   - Reward = minimize damage
   - Scores map to health states

2. CURRICULUM TRAINING
   - Child -> Adolescent -> Adult stages
   - Progressive difficulty
   - Transfer learning between stages

3. EXTENDED TIME SCALES
   - 24 hours (child) → 72 hours (adolescent) → 240 hours (adult)
   - Allows observation of chronic adaptation
   - Gland mass changes over weeks

4. PHYSIOLOGICALLY ACCURATE
   - Literature-based parameters
   - Ultradian + circadian rhythms
   - MR/GR dual receptor system
   - Dose-dependent tissue damage

5. INTERPRETABLE OUTPUTS
   - Positive scores = healthy regulation
   - Negative scores = accumulated damage
   - Direct clinical mapping
   - Research-grade quality

Expected final performance:
  • Adult test scores: +9000 to +10500
  • Allostatic load: ~0.2-0.4 per step
  • Clinical state: Excellent health, robust HPA axis
