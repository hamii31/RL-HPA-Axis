#ifndef HPA_H
#define HPA_H

#ifdef __cplusplus
extern "C" {
#endif

    /* Developmental stages */
    typedef enum {
        STAGE_CHILD = 0,
        STAGE_ADOLESCENT = 1,
        STAGE_ADULT = 2
    } DevelopmentalStage;

    /* HPA Environment State */
    typedef struct HPA {
        /* Hormone concentrations */
        double cortisol;      // μg/dL
        double acth;          // pg/mL
        double crh;           // pg/mL

        /* Gland masses */
        double pituitary_mass;
        double adrenal_mass;

        /* Receptor populations */
        double mr_receptors;
        double gr_receptors;

        /* State variables */
        double stress_level;
        double time_hours;
        int day;

        /* Ultradian oscillator */
        double ultradian_phase;
        double ultradian_period;

        /* Cortisol history (for variance) */
        double cortisol_history[50];
        int history_index;
        double cumulative_load;

        /* Episode parameters */
        int current_step;
        int max_steps;

        /* Developmental stage parameters */
        double feedback_maturity;
        double receptor_sensitivity;
        double stress_resilience;
        DevelopmentalStage stage;

        /* Time parameters */
        double dt;  // time step (hours)

        /* Physiological parameters (decay constants) */
        double k_cortisol;
        double k_acth;
        double k_crh;

        /* Secretion rates */
        double crh_basal_secretion;
        double acth_basal_secretion;
        double cortisol_basal_secretion;

        /* Receptor binding constants */
        double mr_kd;
        double gr_kd;

        /* Feedback strengths */
        double mr_feedback_strength;
        double gr_feedback_strength;

        /* Gland adaptation rates */
        double gland_growth_rate;
        double gland_atrophy_rate;

        /* Optimal ranges */
        double optimal_cortisol;
        double optimal_acth;
        double optimal_crh;

        /* Tolerance windows */
        double cortisol_tolerance;
        double acth_tolerance;
        double crh_tolerance;

    } HPA;

/* State vector size (for RL agent) */
#define HPA_STATE_SIZE 12

/* Action space size */
#define HPA_ACTION_SIZE 9

    /* Initialize HPA environment */
    void HPA_init(HPA* self, double time_step_hours, DevelopmentalStage stage);

    /* Reset environment to initial state */
    void HPA_reset(HPA* self, double* state);

    /* Execute one time step */
    void HPA_step(HPA* self, int action, double* next_state, double* reward, int* done);

    /* Get current state vector */
    void HPA_get_state(HPA* self, double* state);

    /* Calculate allostatic load */
    double HPA_calculate_allostatic_load(HPA* self, double mr_occ, double gr_occ);

    /* Helper functions */
    double HPA_cortisol_to_nmol(double cortisol_ugdl);
    void HPA_calculate_receptor_occupancy(HPA* self, double cortisol_nM, double* mr_occ, double* gr_occ);
    double HPA_get_circadian_amplitude(HPA* self);
    double HPA_get_ultradian_pulse(HPA* self);
    void HPA_update_gland_masses(HPA* self);

    /* Cleanup */
    void HPA_destroy(HPA* self);

#ifdef __cplusplus
}
#endif

#endif /* HPA_H */
