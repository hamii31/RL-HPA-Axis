Physiologically grounded RL environment modelling the HPA axis.
Adapted to match in-depth analysis of [PMC3181830](https://pmc.ncbi.nlm.nih.gov/articles/PMC3181830/) and [PMC4867107](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4867107/).

Biological components modelled:
  --- CORE CASCADE ---
  - Full CRF → ACTH → Cortisol cascade
  - AVP (Arginine Vasopressin): co-secretagogue (V1b / Gq / PKC pathway)
    Expression increases with chronic stress; potentiates ACTH release
  - POMC processing: ACTH, beta-endorphin (stress buffering), melanocortins
    (alpha-MSH: anti-inflammatory, metabolic; CART: reward/stress/feeding)
  - CRFR1 / CRFR2 receptor populations
  - Urocortins (Ucn1, Ucn2, Ucn3): CRF-family peptides modulating CRFR1/2
      Ucn1: Edinger-Westphal nucleus; high affinity CRFR1+CRFR2 agonist
      Ucn2: PVN and LC; CRFR2 preferring
      Ucn3: perifornical hypothalamus, BNST, lateral septum, amygdala; CRFR2

  --- FEEDBACK MECHANISMS ---
  - Fast nongenomic feedback: sensitive to rate of cortisol rise (membrane GR)
  - Slow genomic feedback: sensitive to cortisol level (nuclear GR)
  - MR vs GR: MR dominates at basal levels; GR dominates under stress
  - Hippocampal feedback: MR+GR in hippocampus suppress basal/stress HPA
    (hippocampal lesion → elevated basal cortisol, prolonged ACTH response)

  --- PVN AFFERENTS ---
  1. NTS (Nucleus Solitary Tract): major excitatory drive to PVN
     Receives input from mPFC (social cognition) and CeA (fear/anxiety)
  2. SFO (Subfornical Organ) / Lamina Terminalis: angiotensinergic input
     to PVN based on blood pressure and osmotic composition of blood;
     promotes CRF secretion and biosynthesis
  3. GABAergic inhibition (DMH / POA): stress-activated inhibitory brake
     on PVN; POA integrates gonadal steroids with HPA; lesions amplify HPA
  4. Arcuate Nucleus neuropeptides (metabolic-HPA bridge):
     NPY / AGRP: activate HPA, directly increase CRF
     alpha-MSH / CART: increase ACTH and corticosteroids, stimulate CRF
  5. Limbic structures:
     Amygdala (CeA): physical stressors → NTS → PVN; glucocorticoids
       potentiate CRF expression in CeA (chronic-stress amplification loop)
     Amygdala (MeA): emotional/social stressors → BNST/MePO → PVN
     Hippocampus: inhibitory (multisynaptic via BNST/peri-PVN GABA)
     PFC (mPFC / prelimbic): inhibitory on HPA; damage amplifies ACTH/cortisol
  6. LC (Locus Coeruleus): noradrenergic; activated by stress/CRF;
     elicits ACTH release, anxiety behaviour, immune suppression

  --- PLASTICITY ---
  - Gland plasticity: adrenal hypertrophy/atrophy, pituitary adaptation
  - Receptor downregulation under chronic cortisol (GR, MR)
  - CRFR1 downregulation (homologous desensitisation), CRFR2 upregulation
  - Hippocampal damage index accumulates with chronic stress (GR/MR loss)
  - CeA sensitisation: glucocorticoids increase CeA CRF expression
  - Circadian rhythm (SCN-driven, modulated by AVP projections)
  - Ultradian pulsatility (~60-90 min periodicity)
  - Allostatic load: cumulative biological cost
