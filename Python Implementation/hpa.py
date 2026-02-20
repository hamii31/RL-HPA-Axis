import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


# ============================================================
#  ENVIRONMENT
# ============================================================

class HPAEnvironment:
    """
    Physiologically realistic HPA axis simulation environment.

    State (27-dim):
        [0]  stress_emotional      physical component (0-1)
        [1]  stress_physical       emotional/social component (0-1)
        [2]  crh                   (normalised)
        [3]  acth                  (normalised)
        [4]  cortisol              (normalised)
        [5]  avp                   (normalised)
        [6]  beta_endorphin        (normalised)
        [7]  melanocortin_tone     alpha-MSH / CART aggregate (normalised)
        [8]  crfr1_density         (normalised)
        [9]  crfr2_density         (normalised)
        [10] ucn_tone              aggregate Ucn1+2+3 signal (normalised)
        [11] time_of_day           (0-1 circadian phase)
        [12] cortisol_trend        (recent rate of change)
        [13] circadian_amplitude   (normalised)
        [14] mr_occupancy          (0-1)
        [15] gr_occupancy          (0-1)
        [16] hippocampal_feedback  (normalised inhibitory signal)
        [17] hippocampal_damage    (0-1, cumulative stress damage)
        [18] nts_drive             (normalised excitatory input to PVN)
        [19] gaba_inhibition       (normalised inhibitory brake)
        [20] sfo_drive             angiotensinergic input (normalised)
        [21] cea_activity          central amygdala — physical stress (normalised)
        [22] mea_activity          medial amygdala — emotional stress (normalised)
        [23] pfc_inhibition        prefrontal cortex inhibitory tone (normalised)
        [24] lc_activity           locus coeruleus noradrenergic tone (normalised)
        [25] pituitary_mass        (normalised)
        [26] adrenal_mass          (normalised)

    Actions (27):
        Cartesian product of {-1, 0, +1} for (CRF mod, ACTH mod,
        cortisol mod). Decoded as action = crh_a + 3*acth_a + 9*cort_a.

    Reward:
        reward = offset - allostatic_load
    """

    # ------------------------------------------------------------------
    #  Physiological constants (literature-grounded)
    # ------------------------------------------------------------------
    HALFLIFE_CORTISOL  = 1.25   # ~75 min
    HALFLIFE_ACTH      = 0.17   # ~10 min
    HALFLIFE_CRH       = 0.25   # ~15 min
    HALFLIFE_AVP       = 0.33   # ~20 min
    HALFLIFE_BETAEP    = 0.5    # ~30 min
    HALFLIFE_UCN1      = 0.5    # ~30 min (slower CRF relative)
    HALFLIFE_UCN23     = 1.0    # Ucn2/3 longer half-life

    MR_KD = 0.5     # nM — high affinity
    GR_KD = 5.0     # nM — low affinity

    CORTISOL_TO_NM = 27.6   # μg/dL → nM

    OPT_CORTISOL = 15.0   # μg/dL
    OPT_ACTH     = 25.0   # pg/mL
    OPT_CRH      = 100.0  # pg/mL
    OPT_AVP      = 4.0    # pg/mL

    TOL_CORTISOL = 7.0
    TOL_ACTH     = 15.0
    TOL_CRH      = 50.0

    CRH_BASAL    = 50.0
    ACTH_BASAL   = 15.0
    CORT_BASAL   = 8.0
    AVP_BASAL    = 2.0
    BETAEP_BASAL = 5.0

    GLAND_GROWTH   = 0.001
    GLAND_ATROPHY  = 0.0008

    def __init__(self, time_step_hours: float = 0.1, max_steps: int = 2400):
        self.dt        = time_step_hours
        self.max_steps = max_steps

        self.k_cort   = np.log(2) / self.HALFLIFE_CORTISOL
        self.k_acth   = np.log(2) / self.HALFLIFE_ACTH
        self.k_crh    = np.log(2) / self.HALFLIFE_CRH
        self.k_avp    = np.log(2) / self.HALFLIFE_AVP
        self.k_betaep = np.log(2) / self.HALFLIFE_BETAEP
        self.k_ucn1   = np.log(2) / self.HALFLIFE_UCN1
        self.k_ucn23  = np.log(2) / self.HALFLIFE_UCN23

        self.ultradian_period = 1.5

        self._hist_len = 50
        self.cortisol_history = deque(maxlen=self._hist_len)

        self.reset()

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset to a randomised physiological starting state."""
        # Core hormones
        self.crh             = 100.0 + np.random.normal(0, 20)
        self.acth            = 25.0  + np.random.normal(0, 5)
        self.cortisol        = 12.0  + np.random.normal(0, 2)
        self.avp             = 4.0   + np.random.normal(0, 0.5)
        self.beta_endorphin  = 5.0   + np.random.normal(0, 1)

        # POMC-derived products
        # melanocortin_tone aggregates alpha-MSH and CART signalling;
        # both drive ACTH and CRF and are downstream of POMC processing
        self.melanocortin_tone = 1.0

        # Urocortins (Ucn1 peaks at Edinger-Westphal; Ucn2/3 at PVN/BNST)
        # They modulate CRFR1 and CRFR2 tone; included as separate pools
        self.ucn1  = 1.0    # CRFR1+CRFR2 agonist (Edinger-Westphal)
        self.ucn23 = 1.0    # CRFR2-preferring (PVN, BNST, lateral septum, amygdala)

        # Receptor populations
        self.mr_receptors   = 1.0
        self.gr_receptors   = 1.0
        self.crfr1_density  = 1.0
        self.crfr2_density  = 1.0

        # Gland masses
        self.pituitary_mass = 1.0
        self.adrenal_mass   = 1.0

        # ---- Upstream regulatory signals ----

        # NTS: major excitatory drive on PVN (receives from mPFC and CeA)
        self.nts_drive       = 0.2

        # GABAergic inhibition from DMH/POA (stress-activated brake on PVN)
        # POA also integrates gonadal steroids with HPA
        self.gaba_inhibition = 0.3

        # SFO / Lamina terminalis angiotensinergic drive on PVN
        # Reflects osmotic / blood-pressure state → promotes CRF
        self.sfo_drive = 0.2

        # ---- Amygdala (split by stressor type, per analysis) ----
        # CeA: physical stressors (hemorrhage, immune challenge)
        #   → densely innervates NTS and parabrachial nucleus
        #   → glucocorticoids potentiate CRF in CeA (chronic amplification loop)
        self.cea_activity = 0.2

        # MeA: emotional/social stressors (predator, social, restraint)
        #   → projects via BNST, MePO, Pmv → PVN (no large direct CeA→PVN pathway)
        self.mea_activity = 0.2

        # Cumulative CeA sensitisation by glucocorticoids
        # (GR/MR expressed in CeA; glucocorticoids potentiate CRF there)
        self.cea_sensitisation = 0.0

        # ---- PFC inhibitory tone ----
        # mPFC / prelimbic cortex: inhibitory on HPA; damage amplifies response
        # Infralimbic cortex → BNST, amygdala, NTS (fear inhibition)
        # Prelimbic cortex → POA, DMH (indirect inhibition)
        # High GR density in PFC layers II/III/VI (glucocorticoid feedback)
        self.pfc_inhibition = 0.4

        # ---- LC (Locus Coeruleus) noradrenergic tone ----
        # Largest cluster of noradrenergic neurons; innervates whole neuroaxis
        # Activated by: stressors, CRF
        # Effects: ACTH release, anxiety, immune suppression
        # Dysfunction linked to affective and stress-related disorders
        self.lc_activity = 0.2

        # ---- Hippocampal state ----
        # hippocampal_damage accumulates with chronic stress
        # (chronic glucocorticoids damage hippocampal neurons → loss of HPA inhibition)
        # When damage is high: basal cortisol rises, CRF/AVP expression increases,
        # ACTH/cortisol response to stress is prolonged
        self.hippocampal_damage = 0.0

        # ---- Arcuate nucleus metabolic signals ----
        # NPY/AGRP: activated by low glucose/insulin/leptin; activate HPA; AGRP ↑CRF
        # alpha-MSH/CART: ↑ ACTH and corticosteroids; stimulate CRF; activate cAMP-BP
        # Modelled as a single metabolic_drive variable (positive = NPY/AGRP dominated)
        self.arcuate_metabolic_drive = 0.0   # neutral

        # Stress decomposition: physical vs emotional
        self.stress_physical  = np.random.uniform(0, 1.5)
        self.stress_emotional = np.random.uniform(0, 1.5)

        # Chronic-stress memory
        self.chronic_stress_index = 0.0

        # Physiological state
        self.time_hours       = np.random.uniform(0, 24)
        self.day              = 0
        self.ultradian_phase  = np.random.uniform(0, 2 * np.pi)

        self._prev_cortisol            = self.cortisol
        self._fast_feedback_signal     = 0.0

        self.current_step    = 0
        self.cumulative_load = 0.0

        self.cortisol_history.clear()
        for _ in range(self._hist_len):
            self.cortisol_history.append(self.cortisol)

        return self._get_state()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @property
    def stress_level(self) -> float:
        """Total stress (physical + emotional), used in legacy paths."""
        return self.stress_physical + self.stress_emotional

    def _cortisol_nm(self) -> float:
        return self.cortisol * self.CORTISOL_TO_NM

    def _receptor_occupancy(self) -> tuple[float, float]:
        cnm = self._cortisol_nm()
        mr  = cnm / (self.MR_KD  + cnm)
        gr  = cnm / (self.GR_KD  + cnm)
        return mr, gr

    # ---- Feedback mechanisms ----------------------------------------

    def _fast_nongenomic_feedback(self) -> float:
        """
        Fast nongenomic feedback (membrane GR).
        Sensitive to rate of cortisol rise. Hypothesised second mechanism
        referenced in Aguilera (2011).
        """
        rate     = (self.cortisol - self._prev_cortisol) / self.dt
        pos_rate = max(0.0, rate)
        signal   = pos_rate / (pos_rate + 5.0)
        self._fast_feedback_signal = 0.7 * signal + 0.3 * self._fast_feedback_signal
        return self._fast_feedback_signal

    def _slow_genomic_feedback(self, mr_occ: float, gr_occ: float) -> float:
        """
        Slow genomic feedback via nuclear GR.
        GR dominates during high stress; MR dominates at basal levels.
        GR density in PFC layers II/III/VI also contributes here as PFC
        downward-modulation of CRF transcription.
        """
        mr_fb = 0.3 * mr_occ * self.mr_receptors
        gr_fb = 0.7 * gr_occ * self.gr_receptors
        # PFC provides additional glucocorticoid-mediated genomic suppression
        pfc_genomic = 0.1 * gr_occ * self.gr_receptors * self.pfc_inhibition
        return (mr_fb + gr_fb + pfc_genomic)

    def _hippocampal_feedback(self, mr_occ: float, gr_occ: float) -> float:
        """
        Hippocampal corticosteroid feedback to PVN.
        MR drives tonic suppression; GR provides stress-proportional inhibition.
        Projects via multisynaptic pathway: subiculum/CA1 → BNST/peri-PVN
        GABAergic neurons → parvocellular PVN (inhibitory).

        Hippocampal lesions:
          - elevate basal circulating glucocorticoids
          - increase CRF and AVP expression
          - prolong ACTH/corticosterone response to stress

        These effects are captured via hippocampal_damage (0-1): as damage
        accumulates, the inhibitory signal is attenuated proportionally.
        """
        hip_signal = 0.4 * mr_occ * self.mr_receptors + \
                     0.6 * gr_occ * self.gr_receptors
        # Chronic stress damages hippocampal neurons → reduced feedback
        # (chronic_stress_index AND hippocampal_damage both contribute)
        damage_factor = max(0.1, 1.0 - 0.5 * self.hippocampal_damage
                                      - 0.3 * np.tanh(self.chronic_stress_index))
        return hip_signal * damage_factor

    def _total_negative_feedback(self) -> tuple[float, float, float]:
        """Combined negative feedback to PVN/pituitary."""
        mr_occ, gr_occ = self._receptor_occupancy()
        slow_genomic    = self._slow_genomic_feedback(mr_occ, gr_occ)
        fast_nongenomic = self._fast_nongenomic_feedback()
        hippocampal     = self._hippocampal_feedback(mr_occ, gr_occ)
        total = slow_genomic + 0.3 * fast_nongenomic + 0.4 * hippocampal
        return total, mr_occ, gr_occ

    # ---- Urocortins -------------------------------------------------

    def _update_urocortins(self) -> None:
        """
        Urocortin (Ucn1, Ucn2, Ucn3) dynamics.

        Ucn1 (Edinger-Westphal nucleus):
          - High-affinity agonist at both CRFR1 and CRFR2
          - Modulates pupil constriction / lens accommodation (autonomic)
          - Contributes to CRFR1-mediated HPA drive and CRFR2 brake

        Ucn2 (PVN and LC):
          - CRFR2-preferring; expressed where AVP and CRF also act
          - Locus coeruleus expression connects it to arousal/noradrenergic tone

        Ucn3 (perifornical hypothalamus, BNST, lateral septum, amygdala):
          - Highly CRFR2-selective; modulates anxiety, sleep-wakefulness,
            emotional processing via BNST and lateral septum
          - Upregulated by emotional stressors

        Net effect: Ucn1 potentiates CRFR1 drive; Ucn2/3 potentiate the
        CRFR2 brake (especially under chronic emotional stress).
        """
        # Ucn1: mild stress-driven increase, also basal autonomic tone
        ucn1_prod = 0.5 + 0.1 * self.stress_level
        self.ucn1 = np.clip(
            self.ucn1 + (ucn1_prod - self.k_ucn1 * self.ucn1) * self.dt,
            0.1, 5.0
        )

        # Ucn2/3: preferentially driven by emotional/social stressors
        # (Ucn3 localised in amygdala and BNST — sites of emotional stress)
        ucn23_prod = 0.5 + 0.2 * self.stress_emotional + 0.05 * self.lc_activity
        self.ucn23 = np.clip(
            self.ucn23 + (ucn23_prod - self.k_ucn23 * self.ucn23) * self.dt,
            0.1, 8.0
        )

    def _ucn_crfr_modulation(self) -> tuple[float, float]:
        """
        Return (crfr1_boost, crfr2_boost) from current urocortin levels.
        Ucn1 drives both; Ucn2/3 preferentially drive CRFR2.
        """
        crfr1_boost = 0.05 * (self.ucn1 - 1.0)
        crfr2_boost = 0.04 * (self.ucn1 - 1.0) + 0.06 * (self.ucn23 - 1.0)
        return crfr1_boost, crfr2_boost

    # ---- Amygdala ---------------------------------------------------

    def _update_amygdala(self, mr_occ: float, gr_occ: float) -> None:
        """
        Update CeA and MeA activity.

        CeA (central amygdala):
          - Activated by physical stressors (hemorrhage, immune challenge)
          - Densely innervates NTS and parabrachial nucleus → PVN excitation
          - GR and MR expressed in CeA
          - KEY: glucocorticoids POTENTIATE CRF expression in CeA
            (contrast to hippocampus/PFC where they suppress it)
            This creates a chronic-stress amplification loop:
            cortisol → ↑ CeA CRF → ↑ NTS → ↑ PVN CRF → ↑ cortisol
          - cea_sensitisation accumulates with chronic GR activation in CeA

        MeA (medial amygdala):
          - Activated by emotional/social stressors (predator, social, restraint)
          - Projects via BNST, MePO, ventral premammillary nucleus → PVN
          - Limited direct projections to parvocellular PVN
        """
        # CeA driven by physical stress
        cea_target = 0.1 + 0.4 * np.tanh(self.stress_physical / 3.0)
        # Glucocorticoids potentiate CeA CRF (sensitisation loop)
        cea_gluco_boost = 0.1 * gr_occ * self.gr_receptors * (1.0 + self.cea_sensitisation)
        cea_target = min(1.0, cea_target + cea_gluco_boost)
        self.cea_activity += 0.08 * (cea_target - self.cea_activity) * self.dt

        # CeA sensitisation accumulates with chronic GR occupancy in CeA
        # Models the documented observation that GCs increase CeA CRF expression
        self.cea_sensitisation = np.clip(
            self.cea_sensitisation + 0.001 * gr_occ * self.chronic_stress_index * self.dt
            - 0.0005 * self.dt,
            0.0, 2.0
        )

        # MeA driven by emotional/social stress
        mea_target = 0.1 + 0.4 * np.tanh(self.stress_emotional / 3.0)
        self.mea_activity += 0.06 * (mea_target - self.mea_activity) * self.dt

        self.cea_activity = np.clip(self.cea_activity, 0.0, 1.0)
        self.mea_activity = np.clip(self.mea_activity, 0.0, 1.0)

    # ---- PFC inhibitory tone ----------------------------------------

    def _update_pfc(self, mr_occ: float, gr_occ: float) -> None:
        """
        Prefrontal cortex (mPFC / prelimbic / infralimbic) inhibitory tone.

        - Normally inhibitory on HPA: releases catecholamines following
          acute AND chronic stressor exposure to dampen response
        - Damage to ACC and prelimbic cortex amplifies ACTH and glucocorticoid
          responses (empirical evidence cited in analysis)
        - High GR density in PFC layers II/III/VI → PFC itself is subject
          to glucocorticoid feedback (GR binding modulates PFC function)
        - Infralimbic → BNST, amygdala, NTS (fear inhibition pathway)
        - Prelimbic → POA, DMH (indirect GABAergic/GABA modulation)

        Chronic stress degrades PFC inhibitory function (GR downregulation,
        dendritic retraction — modelled here via chronic_stress_index).
        """
        # PFC inhibitory target: normally robust; weakened by chronic stress
        # and further modulated by GR occupancy (genomic modulation of PFC)
        pfc_target = 0.5 * (1.0 - 0.3 * np.tanh(self.chronic_stress_index))
        # Acute high stress transiently suppresses PFC inhibitory tone
        acute_suppression = 0.1 * np.tanh(self.stress_level / 5.0)
        pfc_target = max(0.05, pfc_target - acute_suppression)
        self.pfc_inhibition += 0.04 * (pfc_target - self.pfc_inhibition) * self.dt
        self.pfc_inhibition  = np.clip(self.pfc_inhibition, 0.0, 1.0)

    # ---- Locus Coeruleus --------------------------------------------

    def _update_lc(self) -> None:
        """
        Locus coeruleus (LC) noradrenergic activity.

        - Largest noradrenergic cluster in brain; innervates whole neuroaxis
        - Activated by wide array of stressors → ACTH release, anxiety,
          immune suppression
        - CRF alters LC neuron activity; catabolism of noradrenergic neurons
          in terminal regions
        - Dysfunction of catecholaminergic neurons in LC linked to affective
          and stress-related disorders (anxiety, PTSD, depression)
        - Ucn2 is expressed in PVN and LC, creating a Ucn2→LC→ACTH pathway
        """
        # LC activated by both physical and emotional stress, and by CRH
        lc_target = (0.1
                     + 0.2 * np.tanh(self.stress_level / 5.0)
                     + 0.1 * (self.crh / self.OPT_CRH)
                     + 0.05 * (self.ucn23 - 1.0))  # Ucn2 expressed in LC
        # Chronic stress eventually degrades LC function (catecholamine depletion)
        lc_target *= max(0.4, 1.0 - 0.15 * np.tanh(self.chronic_stress_index))
        self.lc_activity += 0.05 * (lc_target - self.lc_activity) * self.dt
        self.lc_activity  = np.clip(self.lc_activity, 0.0, 1.0)

    # ---- SFO / Lamina Terminalis ------------------------------------

    def _update_sfo(self) -> None:
        """
        Subfornical organ (SFO) angiotensinergic drive on PVN.

        The lamina terminalis (SFO, MePO, VOLT) relays osmotic composition
        and blood-pressure state to the PVN.
        SFO neurons projecting to PVN are angiotensinergic and promote CRF
        secretion and biosynthesis.

        Modelled as: rises with stress (peripheral volume/pressure changes)
        and is correlated with AVP (osmotic regulation shares SFO circuitry).
        """
        # SFO activity correlates with osmotic/cardiovascular stress
        # and with AVP (both regulated by osmolality via SFO → PVN → posterior pituitary)
        sfo_target = 0.1 + 0.15 * np.tanh(self.stress_physical / 4.0) \
                         + 0.1  * (self.avp / self.OPT_AVP - 1.0)
        self.sfo_drive += 0.04 * (sfo_target - self.sfo_drive) * self.dt
        self.sfo_drive  = np.clip(self.sfo_drive, 0.0, 1.0)

    # ---- Arcuate nucleus metabolic drive ----------------------------

    def _update_arcuate(self) -> None:
        """
        Arcuate nucleus neuropeptide drive on HPA (metabolic-HPA bridge).

        NPY / AGRP (activated by low glucose / insulin / leptin):
          - NPY activates HPA axis
          - AGRP significantly increases CRF release
          - Stress exposure can suppress insulin/leptin → activate arcuate

        alpha-MSH / CART:
          - Both increase ACTH and corticosteroids
          - Induce cAMP-binding protein phosphorylation in CRF neurons
          - Stimulate CRF release
          - Downstream of POMC processing (alpha-MSH is a POMC product)

        arcuate_metabolic_drive:
          > 0 → NPY/AGRP dominated (energy deficit / starvation signal)
          < 0 → alpha-MSH/CART dominated (POMC-driven signal)
          Both have excitatory effects on HPA but via different routes.
        """
        # Stress (especially chronic) mimics mild energy deficit → shifts toward NPY/AGRP
        npy_agrp_drive = 0.1 * np.tanh(self.chronic_stress_index) \
                       + 0.05 * np.tanh(self.stress_physical / 3.0)

        # alpha-MSH is a POMC product; rises with ACTH/POMC processing
        # CART also involved in stress/reward; driven by emotional stress
        msh_cart_drive = -0.05 * self.melanocortin_tone \
                         - 0.03 * np.tanh(self.stress_emotional / 3.0)

        target = npy_agrp_drive + msh_cart_drive
        self.arcuate_metabolic_drive += 0.02 * (target - self.arcuate_metabolic_drive) * self.dt
        self.arcuate_metabolic_drive  = np.clip(self.arcuate_metabolic_drive, -1.0, 1.0)

    # ---- AVP dynamics -----------------------------------------------

    def _update_avp(self) -> None:
        """
        AVP dynamics.
        - Parvocellular AVP: potentiates ACTH via V1b/Gq/PKC
        - SCN AVP: modulates circadian cortisol rhythm
        - Parvocellular AVP expression increases with chronic stress (V1b ↑ too)
        - SFO also regulates AVP (osmotic/blood-pressure)
        """
        chronic_boost    = 1.0 + 0.3 * np.tanh(self.chronic_stress_index)
        stress_drive_avp = 0.4 * self.stress_level
        sfo_avp_link     = 0.2 * self.sfo_drive   # osmotic regulation

        avp_production = (
            self.AVP_BASAL * chronic_boost
            + stress_drive_avp
            + sfo_avp_link
            - self.AVP_BASAL * 0.3 * (self.cortisol / self.OPT_CORTISOL)
        )
        avp_decay  = self.k_avp * self.avp
        self.avp   = np.clip(self.avp + (avp_production - avp_decay) * self.dt, 0.5, 30.0)

    # ---- POMC / beta-endorphin / melanocortins ----------------------

    def _update_pomc_products(self) -> None:
        """
        POMC processing products.

        Beta-endorphin: co-released with ACTH; reduces stress, manages pain,
        helps maintain homeostasis.

        Melanocortins (alpha-MSH, beta-MSH, CART-like):
          - ACTH itself is a melanocortin (binds MC2-R in adrenal cortex)
          - alpha-MSH: skin pigmentation, anti-inflammatory, metabolism
          - Also feed back to arcuate nucleus → CART signals
          - melanocortin_tone is a normalised aggregate
        """
        betaep_prod  = self.BETAEP_BASAL + 0.15 * (self.acth - self.OPT_ACTH) * self.pituitary_mass
        betaep_decay = self.k_betaep * self.beta_endorphin
        self.beta_endorphin = np.clip(
            self.beta_endorphin + (betaep_prod - betaep_decay) * self.dt, 0.0, 40.0
        )

        # Melanocortin tone scales with POMC-processing rate (≈ ACTH level)
        mcr_target         = 0.5 + 0.5 * (self.acth / self.OPT_ACTH)
        self.melanocortin_tone += 0.05 * (mcr_target - self.melanocortin_tone) * self.dt
        self.melanocortin_tone  = np.clip(self.melanocortin_tone, 0.1, 3.0)

    # ---- NTS and GABAergic signals ----------------------------------

    def _update_upstream_signals(self, mr_occ: float, gr_occ: float) -> None:
        """
        NTS excitatory drive and DMH/POA GABAergic inhibition to PVN.

        NTS:
          - Major excitatory input to medial parvocellular PVN
          - Receives psychosocial signals from mPFC and CeA
          - CeA → NTS projection is the main pathway for physical stressor
            activation of HPA (CeA densely innervates NTS)
          - A2/C2 region NTS neurons innervate medial parvocellular PVN
          - NTS drive induces CRF expression

        GABA (DMH/POA):
          - Counter-regulatory; lesions amplify HPA response
          - POA integrates gonadal steroids with HPA (high androgen, estrogen,
            progesterone receptor expression in POA neurons)
          - Glutamate microstimulation of DMH → inhibitory postsynaptic
            potentials in PVN hypophysiotropic neurons
          - PFC prelimbic cortex projects to POA and DMH (indirect inhibition)
        """
        # NTS: driven by direct stress + CeA input (physical) + mPFC drive
        nts_target  = 0.1 + 0.2 * np.tanh(self.stress_level / 5.0) \
                          + 0.3 * self.cea_activity   # CeA → NTS (physical stress)
        # PFC infralimbic → BNST, amygdala, NTS: PFC partially gates NTS
        nts_target *= max(0.3, 1.0 - 0.3 * self.pfc_inhibition)
        self.nts_drive += 0.05 * (nts_target - self.nts_drive) * self.dt

        # GABA: activated by stress (counter-regulatory); modulated by PFC prelimbic
        gaba_target  = 0.15 + 0.2 * np.tanh(self.stress_level / 4.0) \
                           + 0.1 * self.pfc_inhibition   # PFC → POA/DMH → GABA
        self.gaba_inhibition += 0.03 * (gaba_target - self.gaba_inhibition) * self.dt

        self.nts_drive       = np.clip(self.nts_drive,       0.0, 1.0)
        self.gaba_inhibition = np.clip(self.gaba_inhibition, 0.0, 1.0)

    # ---- Hippocampal damage -----------------------------------------

    def _update_hippocampal_damage(self, gr_occ: float) -> None:
        """
        Hippocampal damage accumulation.
        Chronic glucocorticoid exposure damages hippocampal neurons (empirical).
        Loss of hippocampal volume is documented in PTSD and major depression.
        Effects: reduced HPA inhibition, elevated basal cortisol, ↑CRF/AVP,
        prolonged stress-induced ACTH/corticosterone, exaggerated restraint
        and open-field response (stressor-specific).
        """
        # Hippocampal damage driven by sustained high GR occupancy
        damage_rate = 0.0002 * max(0.0, gr_occ - 0.4) * self.chronic_stress_index
        # Partial recovery possible (neurogenesis) but very slow
        repair_rate = 0.00002
        self.hippocampal_damage = np.clip(
            self.hippocampal_damage + (damage_rate - repair_rate) * self.dt,
            0.0, 1.0
        )

    # ---- CRFR1/CRFR2 regulation -------------------------------------

    def _update_crf_receptors(self) -> None:
        """
        CRFR1 and CRFR2 density.
        Urocortins modulate both (Ucn1 → CRFR1+CRFR2; Ucn2/3 → CRFR2).
        """
        ucn_crfr1, ucn_crfr2 = self._ucn_crfr_modulation()

        # CRFR1: downregulated by high CRH (homologous desensitisation)
        if self.crh > 150:
            self.crfr1_density -= 0.00015 * self.dt
        else:
            self.crfr1_density += 0.0001 * self.dt * (1.0 - self.crfr1_density)
        self.crfr1_density = np.clip(self.crfr1_density + ucn_crfr1 * self.dt, 0.2, 1.5)

        # CRFR2: compensatory brake; upregulated by chronic stress and Ucn2/3
        chronic_upmod = 0.00005 * self.chronic_stress_index * self.dt
        self.crfr2_density += chronic_upmod * (1.2 - self.crfr2_density)
        self.crfr2_density += 0.00005 * self.dt * (1.0 - self.crfr2_density)
        self.crfr2_density = np.clip(self.crfr2_density + ucn_crfr2 * self.dt, 0.5, 1.8)

    # ---- Gland plasticity -------------------------------------------

    def _update_glands(self) -> None:
        """Adrenal and pituitary structural adaptation; receptor density."""
        if self.acth > 40:
            self.adrenal_mass += self.GLAND_GROWTH * self.dt
        elif self.acth < 15:
            self.adrenal_mass -= self.GLAND_ATROPHY * self.dt

        if self.cortisol > 25:
            self.pituitary_mass -= self.GLAND_ATROPHY * self.dt
        elif self.cortisol < 10:
            self.pituitary_mass += self.GLAND_GROWTH * self.dt

        self.adrenal_mass   = np.clip(self.adrenal_mass,   0.5, 2.0)
        self.pituitary_mass = np.clip(self.pituitary_mass, 0.5, 2.0)

        cnm = self._cortisol_nm()
        if cnm > 100:
            self.gr_receptors *= (1 - 0.0001 * self.dt)
            self.mr_receptors *= (1 - 0.00005 * self.dt)
        else:
            self.gr_receptors += 0.0001 * self.dt * (1.0 - self.gr_receptors)
            self.mr_receptors += 0.00005 * self.dt * (1.0 - self.mr_receptors)

        self.gr_receptors = np.clip(self.gr_receptors, 0.3, 1.5)
        self.mr_receptors = np.clip(self.mr_receptors, 0.5, 1.2)

    # ---- Circadian / ultradian rhythms ------------------------------

    def _circadian_amplitude(self) -> float:
        """
        SCN-driven circadian cortisol peak (~8 AM).
        AVP from SCN neurons reinforces circadian amplitude.
        Chronic stress disrupts circadian rhythm via AVP and SCN innervation.
        """
        phase          = 2 * np.pi * (self.time_hours - 8) / 24
        base_amplitude = 9.0 + 9.0 * np.cos(phase)
        avp_mod        = 1.0 + 0.05 * (self.avp - self.OPT_AVP)
        # Chronic stress (via AVP dysregulation) dampens circadian amplitude
        circadian_disruption = max(0.5, 1.0 - 0.1 * self.chronic_stress_index)
        return base_amplitude * np.clip(avp_mod, 0.5, 1.5) * circadian_disruption

    def _ultradian_pulse(self) -> float:
        """Ultradian pulsatile cortisol release (~60-90 min periodicity)."""
        self.ultradian_phase += 2 * np.pi * self.dt / self.ultradian_period
        return 3.0 * np.sin(self.ultradian_phase) + np.random.normal(0, 0.5)

    # ---- State representation ---------------------------------------

    def _get_state(self) -> np.ndarray:
        """Build normalised 27-dim state vector."""
        mr_occ, gr_occ = self._receptor_occupancy()
        cortisol_avg   = float(np.mean(self.cortisol_history))
        cortisol_trend = (self.cortisol - cortisol_avg) / 10.0
        hip_feedback   = self._hippocampal_feedback(mr_occ, gr_occ)
        ucn_tone       = (self.ucn1 + self.ucn23) / 2.0

        return np.array([
            self.stress_emotional    / 5.0,            # [0]
            self.stress_physical     / 5.0,            # [1]
            self.crh                 / 300.0,          # [2]
            self.acth                / 100.0,          # [3]
            self.cortisol            / 40.0,           # [4]
            self.avp                 / 20.0,           # [5]
            self.beta_endorphin      / 30.0,           # [6]
            self.melanocortin_tone   / 3.0,            # [7]
            self.crfr1_density       / 1.5,            # [8]
            self.crfr2_density       / 1.8,            # [9]
            ucn_tone                 / 5.0,            # [10]
            self.time_hours          / 24.0,           # [11]
            cortisol_trend,                            # [12]
            self._circadian_amplitude() / 20.0,        # [13]
            mr_occ,                                    # [14]
            gr_occ,                                    # [15]
            hip_feedback,                              # [16]
            self.hippocampal_damage,                   # [17]
            self.nts_drive,                            # [18]
            self.gaba_inhibition,                      # [19]
            self.sfo_drive,                            # [20]
            self.cea_activity,                         # [21]
            self.mea_activity,                         # [22]
            self.pfc_inhibition,                       # [23]
            self.lc_activity,                          # [24]
            self.pituitary_mass      / 2.0,            # [25]
            self.adrenal_mass        / 2.0,            # [26]
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Execute one time step.

        Action encoding (27 discrete actions):
            crh_mod  = (action % 3)      - 1  ∈ {-1, 0, +1}
            acth_mod = (action // 3 % 3) - 1
            cort_mod = (action // 9 % 3) - 1
        """
        crh_mod  = ((action % 3)      - 1) * 0.3
        acth_mod = ((action // 3 % 3) - 1) * 0.5
        cort_mod = ((action // 9 % 3) - 1) * 0.8

        self._prev_cortisol = self.cortisol

        # --- Regulatory signals ---
        total_feedback, mr_occ, gr_occ = self._total_negative_feedback()

        # --- Update limbic/upstream structures (order reflects biology) ---
        self._update_amygdala(mr_occ, gr_occ)   # amygdala: glucocorticoid loop
        self._update_pfc(mr_occ, gr_occ)        # PFC inhibitory tone
        self._update_lc()                        # LC noradrenergic
        self._update_sfo()                       # SFO angiotensinergic
        self._update_arcuate()                   # arcuate metabolic signals
        self._update_upstream_signals(mr_occ, gr_occ)   # NTS and GABA (use updated limbic)
        self._update_urocortins()                # urocortin tone
        self._update_hippocampal_damage(gr_occ)  # hippocampal damage accumulation

        # --- AVP synergy (V1b/Gq/PKC; increases with chronic stress) ---
        avp_synergy = 1.0 + 0.15 * (self.avp / self.OPT_AVP - 1.0) * \
                      (1.0 + 0.2 * self.chronic_stress_index)

        # --- CRFR2 dampening ---
        crfr2_brake = max(0.5, 1.0 - 0.2 * (self.crfr2_density - 1.0))

        # --- CRFR1-mediated CRH→ACTH efficiency ---
        crh_to_acth_efficiency = self.crfr1_density * crfr2_brake

        # --- Arcuate contributions ---
        # NPY/AGRP (positive drive): activates HPA, increases CRF
        # alpha-MSH/CART (negative drive in variable): increases ACTH/cortisol, CRF
        # Both are net excitatory on HPA; just via different mechanistic paths
        arcuate_crh_boost  = 10.0 * abs(self.arcuate_metabolic_drive)   # CRF drive (both polarities)
        arcuate_acth_boost =  5.0 * abs(self.arcuate_metabolic_drive)   # ACTH drive

        # MeA → BNST → PVN adds additional excitatory drive to CRH
        mea_crh_drive = 10.0 * self.mea_activity

        # LC contributes to ACTH release (documented effect)
        lc_acth_contribution = 3.0 * self.lc_activity

        # --- CRH / CRF dynamics ---
        crh_production = (
            self.CRH_BASAL
            + 10.0 * self.stress_level
            + 30.0 * self.nts_drive                # NTS excitatory drive (major)
            + 15.0 * self.sfo_drive                # SFO angiotensinergic
            + mea_crh_drive                        # MeA → BNST → PVN
            + arcuate_crh_boost                    # arcuate NPY/AGRP or alpha-MSH/CART
            - self.CRH_BASAL * total_feedback * 0.8
            - 20.0 * self.gaba_inhibition          # DMH/POA GABAergic brake
            + crh_mod * 20.0
        )
        crh_decay = self.k_crh * self.crh
        self.crh  = np.clip(self.crh + (crh_production - crh_decay) * self.dt, 0.0, 400.0)

        # --- ACTH dynamics ---
        crh_stimulation  = 0.2 * (self.crh - 100.0) * crh_to_acth_efficiency
        avp_contribution = 0.1 * (self.avp - self.OPT_AVP) * avp_synergy

        acth_production = (
            self.ACTH_BASAL * self.pituitary_mass
            + crh_stimulation
            + avp_contribution
            + lc_acth_contribution                # LC → ACTH (documented)
            + arcuate_acth_boost                  # alpha-MSH/CART → ACTH
            - self.ACTH_BASAL * total_feedback * 0.5
            + acth_mod * 10.0
        )
        acth_decay = self.k_acth * self.acth
        self.acth  = np.clip(self.acth + (acth_production - acth_decay) * self.dt, 0.0, 200.0)

        # --- Cortisol dynamics ---
        circadian_drive = self._circadian_amplitude()
        ultradian       = self._ultradian_pulse()
        acth_stim       = 0.15 * (self.acth - self.OPT_ACTH) * self.adrenal_mass
        stress_direct   = 2.0 * self.stress_level

        cort_production = (
            (circadian_drive / 12.0) * self.CORT_BASAL
            + acth_stim
            + stress_direct
            + ultradian * 0.3
            + cort_mod * 2.0
        )
        cort_decay    = self.k_cort * self.cortisol
        self.cortisol = np.clip(
            self.cortisol + (cort_production - cort_decay) * self.dt, 0.0, 60.0
        )
        self.cortisol_history.append(self.cortisol)

        # --- POMC products ---
        self._update_pomc_products()

        # --- AVP ---
        self._update_avp()

        # --- Structural adaptation ---
        self._update_glands()
        self._update_crf_receptors()

        # --- Time ---
        self.time_hours += self.dt
        if self.time_hours >= 24.0:
            self.time_hours -= 24.0
            self.day += 1

        # --- Stress process ---
        # Decompose into physical and emotional stressor events
        self.stress_physical  = max(0.0, self.stress_physical  * 0.97 - 0.03)
        self.stress_emotional = max(0.0, self.stress_emotional * 0.97 - 0.03)

        if np.random.random() < 0.02:
            magnitude   = np.random.choice([2, 5, 8], p=[0.6, 0.3, 0.1])
            is_physical = np.random.random() < 0.4   # 40% physical, 60% emotional
            if is_physical:
                self.stress_physical  = min(10.0, self.stress_physical  + magnitude)
            else:
                self.stress_emotional = min(10.0, self.stress_emotional + magnitude)

        # Chronic stress index (slow integrator)
        self.chronic_stress_index = np.clip(
            self.chronic_stress_index * (1 - 0.001 * self.dt)
            + 0.001 * self.stress_level * self.dt,
            0.0, 5.0
        )

        # --- Reward ---
        allostatic_load     = self._allostatic_load(mr_occ, gr_occ)
        self.cumulative_load += allostatic_load
        reward               = 5.0 - allostatic_load

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_state(), reward, done

    # ------------------------------------------------------------------
    #  Allostatic load
    # ------------------------------------------------------------------

    def _allostatic_load(self, mr_occ: float, gr_occ: float) -> float:
        """
        Biological cost per time step.

        Components (aligned with analysis):
        1.  Basal metabolic cost
        2.  Cortisol deviation (quadratic in tolerance, sharper outside)
        3.  Tissue damage: hypercortisolism / hypocortisolism
        4.  ACTH / CRH dysregulation
        5.  Receptor occupancy vs context-optimal
        6.  Receptor downregulation (chronic exposure marker)
        7.  Gland pathology (hypertrophy / atrophy)
        8.  Cortisol instability (PTSD / panic phenotype — high lability)
        9.  Stress response appropriateness
        10. Beta-endorphin buffering credit
        11. CRFR1/CRFR2 imbalance
        12. Chronic stress penalty
        13. CeA sensitisation penalty (amplified chronic stress loop)
        14. Hippocampal damage penalty (lost HPA inhibitory capacity)
        15. LC dysfunction penalty (catecholamine dysregulation)
        16. PFC inhibitory failure penalty
        """

        load = 0.05

        # Cortisol deviation
        dev = self.cortisol - self.OPT_CORTISOL
        if abs(dev) <= self.TOL_CORTISOL:
            load += 0.01 * (dev / self.TOL_CORTISOL) ** 2
        else:
            excess = abs(dev) - self.TOL_CORTISOL
            load  += 0.5 * (excess / self.TOL_CORTISOL) ** 2

        # Tissue damage
        if self.cortisol > 25:
            excess = self.cortisol - 25
            load  += excess * 0.3
            if self.cortisol > 35:
                crisis = ((self.cortisol - 35) / 10) ** 2
                load  += crisis * 2.0
        elif self.cortisol < 5:
            deficit = 5 - self.cortisol
            load   += deficit * 0.7
            if self.cortisol < 2:
                crisis = ((2 - self.cortisol) / 2) ** 2
                load  += crisis * 5.0

        # ACTH dysregulation
        acth_dev = abs(self.acth - self.OPT_ACTH)
        if acth_dev > self.TOL_ACTH:
            load += 0.02 * ((acth_dev - self.TOL_ACTH) / self.TOL_ACTH) ** 2
        crh_dev = abs(self.crh - self.OPT_CRH)
        if crh_dev > self.TOL_CRH:
            load += 0.01 * ((crh_dev - self.TOL_CRH) / self.TOL_CRH) ** 2

        # Receptor occupancy vs context-optimal
        mr_cost = 0.5 * (mr_occ - 0.8) ** 2
        gr_optimal = 0.7 if self.stress_level > 5 else 0.3
        gr_cost    = 0.3 * (gr_occ - gr_optimal) ** 2
        load += mr_cost + gr_cost

        # Receptor downregulation
        load += 0.5 * ((1.0 - self.gr_receptors) ** 2 + (1.0 - self.mr_receptors) ** 2)

        # Gland pathology
        adrenal_path   = (self.adrenal_mass   - 1.0) ** 2
        pituitary_path = (self.pituitary_mass - 1.0) ** 2
        if self.adrenal_mass   < 0.5 or self.adrenal_mass   > 1.5: adrenal_path   *= 3.0
        if self.pituitary_mass < 0.5 or self.pituitary_mass > 1.5: pituitary_path *= 3.0
        load += (adrenal_path + pituitary_path) * 0.3

        # Cortisol instability (high variance → PTSD / panic phenotype)
        if len(self.cortisol_history) >= 10:
            variance = float(np.var(list(self.cortisol_history)[-10:]))
            if variance > 25:
                load += (variance - 25) / 100

        # Stress response appropriateness
        if self.stress_level > 6:
            expected = 20 + self.stress_level * 2
            err = abs(self.cortisol - expected)
            if err > 10:
                load += 0.5 * (err / 10) ** 2
        elif self.stress_level < 2 and self.cortisol > 25:
            load += 0.3 * ((self.cortisol - 25) / 10) ** 2

        # Beta-endorphin buffering credit
        load -= 0.02 * min(self.beta_endorphin / 10.0, 1.0)

        # CRFR1/CRFR2 imbalance
        load += 0.05 * (abs(self.crfr1_density - 1.0) + abs(self.crfr2_density - 1.0))

        # Chronic stress (slow-accumulating tissue damage)
        load += 0.02 * self.chronic_stress_index

        # CeA sensitisation penalty
        # Potentiating CRF in CeA (via glucocorticoids) creates runaway loop
        load += 0.03 * self.cea_sensitisation

        # Hippocampal damage penalty
        # Hippocampal damage → lost inhibitory capacity → elevated basal cortisol
        load += 0.1 * self.hippocampal_damage ** 2

        # LC dysfunction
        # Very high LC activity (hypernoradrenergic) or very low (depleted)
        # are both pathological — U-shaped cost
        lc_optimal = 0.3
        load += 0.05 * (self.lc_activity - lc_optimal) ** 2

        # PFC inhibitory failure
        # Low PFC inhibition → runaway HPA (amplified ACTH/cortisol)
        if self.pfc_inhibition < 0.2:
            load += 0.1 * (0.2 - self.pfc_inhibition) ** 2

        return max(0.0, load)

    def get_state_info(self) -> dict:
        """Return a labelled snapshot of the current physiological state."""
        mr_occ, gr_occ = self._receptor_occupancy()
        return {
            "stress_emotional":    self.stress_emotional,
            "stress_physical":     self.stress_physical,
            "stress_total":        self.stress_level,
            "crh_pg_ml":           self.crh,
            "acth_pg_ml":          self.acth,
            "cortisol_ug_dl":      self.cortisol,
            "avp_pg_ml":           self.avp,
            "beta_endorphin":      self.beta_endorphin,
            "melanocortin_tone":   self.melanocortin_tone,
            "ucn1":                self.ucn1,
            "ucn23":               self.ucn23,
            "crfr1_density":       self.crfr1_density,
            "crfr2_density":       self.crfr2_density,
            "mr_occupancy":        mr_occ,
            "gr_occupancy":        gr_occ,
            "pituitary_mass":      self.pituitary_mass,
            "adrenal_mass":        self.adrenal_mass,
            "nts_drive":           self.nts_drive,
            "gaba_inhibition":     self.gaba_inhibition,
            "sfo_drive":           self.sfo_drive,
            "cea_activity":        self.cea_activity,
            "mea_activity":        self.mea_activity,
            "cea_sensitisation":   self.cea_sensitisation,
            "pfc_inhibition":      self.pfc_inhibition,
            "lc_activity":         self.lc_activity,
            "hippocampal_damage":  self.hippocampal_damage,
            "arcuate_drive":       self.arcuate_metabolic_drive,
            "chronic_stress_idx":  self.chronic_stress_index,
            "cumulative_load":     self.cumulative_load,
        }


# ============================================================
#  AGENT
# ============================================================

class DQNAgent:
    """
    Tabular Q-learning agent with experience replay.
    State dimension expanded to 27 to accommodate new biological components.
    """

    def __init__(
        self,
        state_size:    int   = 27,
        action_size:   int   = 27,
        learning_rate: float = 0.0005,
        gamma:         float = 0.98,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.01,
        epsilon_decay: float = 0.9999999,
        batch_size:    int   = 128,
        memory_size:   int   = 10_000,
    ):
        self.state_size    = state_size
        self.action_size   = action_size
        self.lr            = learning_rate
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.memory        = deque(maxlen=memory_size)
        self.q_table       = {}

    def _key(self, state: np.ndarray) -> tuple:
        return tuple(np.round(state, 1))

    def _q(self, state: np.ndarray) -> np.ndarray:
        k = self._key(state)
        if k not in self.q_table:
            self.q_table[k] = np.zeros(self.action_size)
        return self.q_table[k]

    def act(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self._q(state)))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target        = reward
            if not done:
                target   += self.gamma * float(np.max(self._q(next_state)))
            q_vals         = self._q(state)
            q_vals[action] += self.lr * (target - q_vals[action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================
#  TRAINING
# ============================================================

def train(
    episodes:    int   = 300,
    max_steps:   int   = 2400,
    dt:          float = 0.1,
    print_every: int   = 10,
) -> tuple[DQNAgent, HPAEnvironment, list[float]]:
    env   = HPAEnvironment(time_step_hours=dt, max_steps=max_steps)
    agent = DQNAgent(state_size=27, action_size=27)

    scores = []

    print("=" * 65)
    print("HPA Axis RL Training (Adapted — Full Biological Model)")
    print(f"  Episodes   : {episodes}")
    print(f"  Steps/ep   : {max_steps} ({max_steps * dt:.0f} h = "
          f"{max_steps * dt / 24:.1f} days)")
    print(f"  State dim  : {agent.state_size}  (was 18)")
    print(f"  Action dim : {agent.action_size}")
    print("=" * 65)

    for ep in range(1, episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        total_load   = 0.0
        done         = False

        while not done:
            action                   = agent.act(state)
            next_state, rew, done    = env.step(action)
            agent.remember(state, action, rew, next_state, done)
            agent.replay()
            total_reward += rew
            total_load   += (5.0 - rew)
            state         = next_state

        scores.append(total_reward)
        avg        = float(np.mean(scores[-50:])) if len(scores) >= 50 else float(np.mean(scores))
        avg_load_h = total_load / (max_steps * dt)

        if ep % print_every == 0:
            print(
                f"  Ep {ep:4d}/{episodes} | "
                f"Score: {total_reward:8.1f} | "
                f"Avg50: {avg:8.1f} | "
                f"Load/h: {avg_load_h:.3f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Q-states: {len(agent.q_table)}"
            )

    return agent, env, scores


# ============================================================
#  EVALUATION
# ============================================================

def evaluate(
    agent:      DQNAgent,
    dt:         float = 0.1,
    max_steps:  int   = 2400,
    n_episodes: int   = 5,
) -> dict:
    env = HPAEnvironment(time_step_hours=dt, max_steps=max_steps)
    saved_eps     = agent.epsilon
    agent.epsilon = 0.0

    all_scores = []
    traj_keys  = [
        "cortisol_ug_dl", "acth_pg_ml", "crh_pg_ml", "avp_pg_ml",
        "stress_total", "stress_emotional", "stress_physical",
        "nts_drive", "gaba_inhibition", "sfo_drive",
        "cea_activity", "mea_activity", "pfc_inhibition", "lc_activity",
        "hippocampal_damage", "cea_sensitisation",
        "ucn1", "ucn23", "melanocortin_tone", "arcuate_drive",
        "chronic_stress_idx",
    ]
    trajectories = {k: [] for k in traj_keys}

    for ep in range(n_episodes):
        state    = env.reset()
        done     = False
        ep_score = 0.0
        ep_traj  = {k: [] for k in traj_keys}

        while not done:
            action           = agent.act(state)
            state, rew, done = env.step(action)
            ep_score        += rew
            info             = env.get_state_info()
            for k in traj_keys:
                ep_traj[k].append(info[k])

        all_scores.append(ep_score)
        for k in traj_keys:
            trajectories[k].append(ep_traj[k])

    agent.epsilon = saved_eps
    mean_scores   = float(np.mean(all_scores))
    std_scores    = float(np.std(all_scores))
    print(f"\nEvaluation over {n_episodes} episodes:")
    print(f"  Score: {mean_scores:.1f} ± {std_scores:.1f}")

    return {
        "mean_score":  mean_scores,
        "std_score":   std_scores,
        "scores":      all_scores,
        "trajectories": trajectories,
    }


# ============================================================
#  PLOTTING
# ============================================================

def plot_results(
    scores:       list[float],
    eval_results: dict,
    save_path:    str = "hpa_training_results.png",
) -> None:
    """
    6-panel figure:
      [0,0] Training reward curve + moving average
      [0,1] Cortisol trajectory (greedy policy)
      [1,0] Upstream PVN signals: NTS, GABA, SFO
      [1,1] Hormone cascade: AVP, ACTH, CRH (normalised)
      [2,0] Limbic signals: CeA, MeA, PFC inhibition, LC
      [2,1] Chronic markers: hippocampal damage, CeA sensitisation,
                             chronic stress index, urocortins
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("HPA Axis RL — Full Biological Model (Adapted)", fontsize=13, fontweight="bold")
    trajs = eval_results["trajectories"]

    def _hours(key):
        return np.arange(len(trajs[key][0])) * 0.1

    # --- Panel [0,0]: Training scores ---
    ax = axes[0, 0]
    ax.plot(scores, alpha=0.3, color="steelblue", label="Episode score")
    if len(scores) >= 50:
        avg50 = np.convolve(scores, np.ones(50) / 50, mode="valid")
        ax.plot(range(49, len(scores)), avg50, color="darkblue", linewidth=2, label="50-ep avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("Training Progress"); ax.legend(); ax.grid(alpha=0.3)

    # --- Panel [0,1]: Cortisol ---
    ax = axes[0, 1]
    if trajs["cortisol_ug_dl"]:
        h = _hours("cortisol_ug_dl")
        mc = np.mean(trajs["cortisol_ug_dl"], axis=0)
        ax.plot(h, mc, color="tomato", linewidth=1.5, label="Cortisol")
        ax.axhline(15, color="green",  linestyle="--", alpha=0.6, label="Optimal (15)")
        ax.axhline(25, color="orange", linestyle="--", alpha=0.6, label="Concern (25)")
        ax.axhline(5,  color="red",    linestyle="--", alpha=0.6, label="Risk (<5)")
        ax.set_xlabel("Hours"); ax.set_ylabel("Cortisol (μg/dL)")
        ax.set_title("Cortisol — Greedy Policy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Panel [1,0]: Upstream PVN signals ---
    ax = axes[1, 0]
    if trajs["nts_drive"]:
        h = _hours("nts_drive")
        ax.plot(h, np.mean(trajs["nts_drive"],        axis=0), label="NTS excitation",   color="firebrick")
        ax.plot(h, np.mean(trajs["gaba_inhibition"],  axis=0), label="GABA inhibition",  color="forestgreen")
        ax.plot(h, np.mean(trajs["sfo_drive"],        axis=0), label="SFO (angiotensin)",color="goldenrod")
        ax.plot(h, np.mean(trajs["stress_total"],     axis=0) / 10, label="Stress (norm)",
                color="slategray", linestyle="--")
        ax.set_xlabel("Hours"); ax.set_ylabel("Signal (norm)")
        ax.set_title("Upstream PVN Regulatory Signals (NTS, GABA, SFO)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Panel [1,1]: Hormone cascade ---
    ax = axes[1, 1]
    if trajs["avp_pg_ml"]:
        h = _hours("avp_pg_ml")
        ax.plot(h, np.mean(trajs["crh_pg_ml"],  axis=0) / 300, label="CRH/300",   color="purple")
        ax.plot(h, np.mean(trajs["acth_pg_ml"], axis=0) / 100, label="ACTH/100",  color="orange")
        ax.plot(h, np.mean(trajs["avp_pg_ml"],  axis=0) / 20,  label="AVP/20",    color="cornflowerblue")
        ax.plot(h, np.mean(trajs["melanocortin_tone"], axis=0) / 3, label="Melanocortins/3",
                color="mediumpurple", linestyle=":")
        ax.set_xlabel("Hours"); ax.set_ylabel("Normalised")
        ax.set_title("Hormone Cascade (normalised)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Panel [2,0]: Limbic signals ---
    ax = axes[2, 0]
    if trajs["cea_activity"]:
        h = _hours("cea_activity")
        ax.plot(h, np.mean(trajs["cea_activity"],   axis=0), label="CeA (physical)",   color="crimson")
        ax.plot(h, np.mean(trajs["mea_activity"],   axis=0), label="MeA (emotional)",  color="darkorange")
        ax.plot(h, np.mean(trajs["pfc_inhibition"], axis=0), label="PFC inhibition",   color="steelblue")
        ax.plot(h, np.mean(trajs["lc_activity"],    axis=0), label="LC (noradrenergic)",color="darkgreen")
        ax.set_xlabel("Hours"); ax.set_ylabel("Signal (norm)")
        ax.set_title("Limbic Signals: Amygdala (CeA/MeA), PFC, LC")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Panel [2,1]: Chronic markers ---
    ax = axes[2, 1]
    if trajs["hippocampal_damage"]:
        h = _hours("hippocampal_damage")
        ax.plot(h, np.mean(trajs["hippocampal_damage"],  axis=0), label="Hippocampal damage",    color="saddlebrown")
        ax.plot(h, np.mean(trajs["cea_sensitisation"],   axis=0) / 2, label="CeA sensitisation/2",color="crimson", linestyle="--")
        ax.plot(h, np.mean(trajs["chronic_stress_idx"],  axis=0) / 5, label="Chronic stress/5",  color="gray")
        ax.plot(h, np.mean(trajs["ucn1"],   axis=0) / 5, label="Ucn1/5",   color="violet", linestyle=":")
        ax.plot(h, np.mean(trajs["ucn23"],  axis=0) / 8, label="Ucn2-3/8", color="plum",   linestyle=":")
        ax.set_xlabel("Hours"); ax.set_ylabel("Normalised")
        ax.set_title("Chronic Stress Markers & Urocortins")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    agent, env, scores = train(
        episodes    = 300,
        max_steps   = 2400,
        dt          = 0.1,
        print_every = 10,
    )
    eval_results = evaluate(agent, n_episodes=5)
    print(f"\nFinal Q-table: {len(agent.q_table):,} states")
    print(f"Final epsilon:  {agent.epsilon:.4f}")
    plot_results(scores, eval_results, save_path="hpa_training_results.png")
