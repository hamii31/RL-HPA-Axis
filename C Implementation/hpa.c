#include "hpa.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================================================
// RNG 
// ========================================================

// Random uniform [min, max]
double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Random normal (Box-Muller transform)
double rand_normal(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;

    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }

    has_spare = 1;
    double u, v, s;
    do {
        u = rand_uniform(-1, 1);
        v = rand_uniform(-1, 1);
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

// ========================================================
// Helper Functions
// ========================================================

/* Convert cortisol from μg / dL to nM for receptor binding calculations */
double HPA_cortisol_to_nmol(double cortisol_ugdl) {
    /* Conversion factor: 1 μg/dL cortisol ≈ 27.6 nM */
    return cortisol_ugdl * 27.6;
}

/* Calculate circadian rhythm amplitude for cortisol */
double HPA_get_circadian_amplitude(HPA * self) {
    /* Cosine wave with peak at 8 AM, nadir at 8 PM */
    double phase = 2.0 * M_PI * (self->time_hours - 8.0) / 24.0;
    return 9.0 + 9.0 * cos(phase);  /* Range: 0-18 μg/dL */
}

/* Generate ultradian pulse (90-minute periodicity) */
double HPA_get_ultradian_pulse(HPA * self) {
    /* Update phase */
    self->ultradian_phase += 2.0 * M_PI * self->dt / self->ultradian_period;

    /* Sinusoidal pulse with noise */
    double pulse_amplitude = 3.0 * sin(self->ultradian_phase);
    pulse_amplitude += rand_normal(0.0, 0.5);

    return pulse_amplitude;
}

/* Calculate MR and GR receptor occupancy using Hill equation */
void HPA_calculate_receptor_occupancy(HPA * self, double cortisol_nM,
    double* mr_occ, double* gr_occ) {
    /* Occupancy = [Cortisol] / (Kd + [Cortisol]) */
    *mr_occ = cortisol_nM / (self->mr_kd + cortisol_nM);
    *gr_occ = cortisol_nM / (self->gr_kd + cortisol_nM);
}

/* Calculate total negative feedback from both receptor types */
static double calculate_total_feedback(HPA * self) {
    double cortisol_nM = HPA_cortisol_to_nmol(self->cortisol);
    double mr_occ, gr_occ;
    HPA_calculate_receptor_occupancy(self, cortisol_nM, &mr_occ, &gr_occ);

    /* Weighted sum of receptor occupancies */
    double total_feedback = (
        self->mr_feedback_strength * mr_occ * self->mr_receptors +
        self->gr_feedback_strength * gr_occ * self->gr_receptors
        ) * self->receptor_sensitivity;

    return total_feedback;
}

// ========================================================
// Gland Mass (Chronic) Adaptation 
// ========================================================
void HPA_update_gland_masses(HPA* self) {
    // Adrenal growth with high ACTH
    if (self->acth > 40.0) {
        self->adrenal_mass += self->gland_growth_rate * self->dt;
    }
    else if (self->acth < 15.0) {
        self->adrenal_mass -= self->gland_atrophy_rate * self->dt;
    }

    // Pituitary suppression with high cortisol
    if (self->cortisol > 25.0) {
        self->pituitary_mass -= self->gland_atrophy_rate * self->dt;
    }
    else if (self->cortisol < 10.0) {
        self->pituitary_mass += self->gland_growth_rate * self->dt;
    }

    self->adrenal_mass = fmax(0.5, fmin(2.0, self->adrenal_mass));
    self->pituitary_mass = fmax(0.5, fmin(2.0, self->pituitary_mass));

    // Receptor downregulation
    double cortisol_nM = HPA_cortisol_to_nmol(self->cortisol);
    if (cortisol_nM > 100.0) {
        self->gr_receptors *= (1.0 - 0.0001 * self->dt);
        self->mr_receptors *= (1.0 - 0.00005 * self->dt);
    }
    else {
        self->gr_receptors += 0.0001 * self->dt * (1.0 - self->gr_receptors);
        self->mr_receptors += 0.00005 * self->dt * (1.0 - self->mr_receptors);
    }

    self->gr_receptors = fmax(0.3, fmin(1.5, self->gr_receptors));
    self->mr_receptors = fmax(0.5, fmin(1.2, self->mr_receptors));
}

// ========================================================
// Allostatic Load (Biological Cost) Calculation
// ========================================================
double HPA_calculate_allostatic_load(HPA* self, double mr_occ, double gr_occ) {
    double load = 0.0;

    // Basal metabolic cost
    load += 0.05;

    // Cortisol deviation cost (quadratic)
    double cortisol_deviation = self->cortisol - self->optimal_cortisol;
    if (fabs(cortisol_deviation) <= self->cortisol_tolerance) {
        load += 0.01 * pow(cortisol_deviation / self->cortisol_tolerance, 2);
    }
    else {
        double excess = fabs(cortisol_deviation) - self->cortisol_tolerance;
        load += 0.5 * pow(excess / self->cortisol_tolerance, 2);
    }

    // Tissue-specific damage
    double tissue_damage = 0.0;

    // Hypercortisolism
    if (self->cortisol > 25.0) {
        double excess_cortisol = self->cortisol - 25.0;
        tissue_damage += excess_cortisol * 0.3;

        // Crisis risk
        if (self->cortisol > 35.0) {
            double crisis_mult = pow((self->cortisol - 35.0) / 10.0, 2);
            tissue_damage += crisis_mult * 2.0;
        }
    }
    // Hypocortisolism
    else if (self->cortisol < 5.0) {
        double deficit = 5.0 - self->cortisol;
        tissue_damage += deficit * 0.7;

        // Crisis risk
        if (self->cortisol < 2.0) {
            double crisis_mult = pow((2.0 - self->cortisol) / 2.0, 2);
            tissue_damage += crisis_mult * 5.0;
        }
    }
    load += tissue_damage;

    // ACTH dysregulation
    double acth_deviation = fabs(self->acth - self->optimal_acth);
    if (acth_deviation > self->acth_tolerance) {
        double excess_acth = acth_deviation - self->acth_tolerance;
        load += 0.02 * pow(excess_acth / self->acth_tolerance, 2);
    }

    // CRH dysregulation
    double crh_deviation = fabs(self->crh - self->optimal_crh);
    if (crh_deviation > self->crh_tolerance) {
        double excess_crh = crh_deviation - self->crh_tolerance;
        load += 0.01 * pow(excess_crh / self->crh_tolerance, 2);
    }

    // Receptor dysfunction
    double mr_optimal = 0.8;
    double mr_loss = fabs(mr_occ - mr_optimal);
    load += 0.5 * pow(mr_loss, 2);

    double gr_optimal = (self->stress_level > 5.0) ? 0.7 : 0.3;
    double gr_loss = fabs(gr_occ - gr_optimal);
    load += 0.3 * pow(gr_loss, 2);

    double receptor_downreg = pow(1.0 - self->gr_receptors, 2) +
        pow(1.0 - self->mr_receptors, 2);
    load += receptor_downreg * 0.5;

    // Gland pathology
    double adrenal_path = pow(self->adrenal_mass - 1.0, 2);
    double pituitary_path = pow(self->pituitary_mass - 1.0, 2);

    if (self->adrenal_mass < 0.5 || self->adrenal_mass > 1.5) {
        adrenal_path *= 3.0;
    }
    if (self->pituitary_mass < 0.5 || self->pituitary_mass > 1.5) {
        pituitary_path *= 3.0;
    }

    load += (adrenal_path + pituitary_path) * 0.3;

    // Instability cost
    double variance = 0.0;
    double mean = 0.0;
    for (int i = 0; i < 10 && i < 50; i++) {
        int idx = (self->history_index - i + 50) % 50;
        mean += self->cortisol_history[idx];
    }
    mean /= 10.0;

    for (int i = 0; i < 10 && i < 50; i++) {
        int idx = (self->history_index - i + 50) % 50;
        variance += pow(self->cortisol_history[idx] - mean, 2);
    }
    variance /= 10.0;

    if (variance > 25.0) {
        load += (variance - 25.0) / 100.0;
    }

    // 9. Stress response appropriateness
    if (self->stress_level > 6.0) {
        double expected_cort = 20.0 + self->stress_level * 2.0;
        double response_error = fabs(self->cortisol - expected_cort);
        if (response_error > 10.0) {
            load += 0.5 * pow(response_error / 10.0, 2);
        }
    }
    else if (self->stress_level < 2.0 && self->cortisol > 25.0) {
        load += 0.3 * pow((self->cortisol - 25.0) / 10.0, 2);
    }

    // Adjust by stress resilience
    double vulnerability = 2.0 - self->stress_resilience;
    load *= vulnerability;

    return load;
}

// ========================================================
// Initialize 
// ========================================================
void HPA_init(HPA* self, double time_step_hours, DevelopmentalStage stage) {
    // Time parameters
    self->dt = time_step_hours;
    self->stage = stage;

    // Initial hormone levels
    self->cortisol = 12.0;
    self->acth = 25.0;
    self->crh = 100.0;

    // Glands
    self->pituitary_mass = 1.0;
    self->adrenal_mass = 1.0;

    // Receptors
    self->mr_receptors = 1.0;
    self->gr_receptors = 1.0;

    // State
    self->stress_level = 0.0;
    self->time_hours = 8.0;
    self->day = 0;
    self->current_step = 0;

    // Ultradian
    self->ultradian_phase = 0.0;
    self->ultradian_period = 1.5;  // 90 minutes

    // History
    memset(self->cortisol_history, 0, sizeof(self->cortisol_history));
    self->history_index = 0;
    self->cumulative_load = 0.0;

    // Set stage-specific parameters
    switch (stage) {
    case STAGE_CHILD:
        self->max_steps = 240;  // 24 hours
        self->feedback_maturity = 0.4;
        self->receptor_sensitivity = 0.6;
        self->stress_resilience = 0.5;
        break;

    case STAGE_ADOLESCENT:
        self->max_steps = 720;  // 72 hours
        self->feedback_maturity = 0.9;
        self->receptor_sensitivity = 0.95;
        self->stress_resilience = 0.85;
        break;

    case STAGE_ADULT:
        self->max_steps = 2400;  // 240 hours
        self->feedback_maturity = 1.0;
        self->receptor_sensitivity = 1.0;
        self->stress_resilience = 1.0;
        break;
    }

    // Physiological parameters (literature-based)
    double cortisol_halflife = 1.25;  // hours
    double acth_halflife = 0.17;
    double crh_halflife = 0.25;

    self->k_cortisol = log(2.0) / cortisol_halflife;
    self->k_acth = log(2.0) / acth_halflife;
    self->k_crh = log(2.0) / crh_halflife;

    // Secretion rates
    self->crh_basal_secretion = 50.0;
    self->acth_basal_secretion = 15.0;
    self->cortisol_basal_secretion = 8.0;

    // Receptor binding
    self->mr_kd = 0.5;  // nM
    self->gr_kd = 5.0;  // nM

    // Feedback (modified by maturity)
    self->mr_feedback_strength = 0.3 * self->feedback_maturity;
    self->gr_feedback_strength = 0.7 * self->feedback_maturity;

    // Gland adaptation
    self->gland_growth_rate = 0.001;
    self->gland_atrophy_rate = 0.0008;

    // Optimal ranges
    self->optimal_cortisol = 15.0;
    self->optimal_acth = 25.0;
    self->optimal_crh = 100.0;

    // Tolerances
    self->cortisol_tolerance = 7.0;
    self->acth_tolerance = 15.0;
    self->crh_tolerance = 50.0;
}

// ========================================================
// Start new episode
// ========================================================
void HPA_reset(HPA* self, double* state) {
    // Randomize initial conditions slightly
    self->cortisol = 12.0 + rand_normal(0, 2);
    self->acth = 25.0 + rand_normal(0, 5);
    self->crh = 100.0 + rand_normal(0, 20);

    // Reset glands
    self->pituitary_mass = 1.0;
    self->adrenal_mass = 1.0;
    self->mr_receptors = 1.0;
    self->gr_receptors = 1.0;

    // Random initial stress and time
    self->stress_level = rand_uniform(0, 3);
    self->time_hours = rand_uniform(0, 24);
    self->day = 0;
    self->ultradian_phase = rand_uniform(0, 2 * M_PI);

    self->current_step = 0;
    self->cumulative_load = 0.0;

    // Clear history
    for (int i = 0; i < 50; i++) {
        self->cortisol_history[i] = self->cortisol;
    }
    self->history_index = 0;

    // Get initial state
    HPA_get_state(self, state);
}

// ========================================================
// Extract 12-element state vector for RL agent
// ========================================================
void HPA_get_state(HPA* self, double* state) {
    // Calculate receptor occupancy
    double cortisol_nM = HPA_cortisol_to_nmol(self->cortisol);
    double mr_occ, gr_occ;

    HPA_calculate_receptor_occupancy(self, cortisol_nM, &mr_occ, &gr_occ);

    // Calculate cortisol trend
    double cortisol_avg = 0.0;
    for (int i = 0; i < 50; i++) {
        cortisol_avg += self->cortisol_history[i];
    }

    cortisol_avg /= 50.0;
    double cortisol_trend = (self->cortisol - cortisol_avg) / 10.0;

    // State vector (normalized)
    state[0] = self->stress_level / 10.0;
    state[1] = self->crh / 300.0;
    state[2] = self->acth / 100.0;
    state[3] = self->cortisol / 40.0;
    state[4] = self->time_hours / 24.0;
    state[5] = cortisol_trend;
    state[6] = HPA_get_circadian_amplitude(self) / 20.0;
    state[7] = mr_occ;
    state[8] = gr_occ;
    state[9] = self->pituitary_mass / 2.0;
    state[10] = self->adrenal_mass / 2.0;
    state[11] = (double)self->day / 10.0;
}

// =========================================================
// Step funcion: Main simulation
// =========================================================
void HPA_step(HPA* self, int action, double* next_state, double* reward, int* done) {
    // Decode action (0-8 → 3x3 grid)
    int crh_action = (action % 3) - 1;      // -1, 0, 1
    int acth_action = ((action / 3) % 3) - 1;
    int cortisol_action = ((action / 9) % 3) - 1;

    double crh_mod = crh_action * 0.3;
    double acth_mod = acth_action * 0.5;
    double cortisol_mod = cortisol_action * 0.8;

    // Calculate feedback
    double cortisol_nM = HPA_cortisol_to_nmol(self->cortisol);
    double mr_occ, gr_occ;
    HPA_calculate_receptor_occupancy(self, cortisol_nM, &mr_occ, &gr_occ);

    double total_feedback = (self->mr_feedback_strength * mr_occ * self->mr_receptors +
        self->gr_feedback_strength * gr_occ * self->gr_receptors) *
        self->receptor_sensitivity;

    // === CRH DYNAMICS ===
    double crh_production = self->crh_basal_secretion +
        10.0 * self->stress_level -  // stress_to_crh
        self->crh_basal_secretion * total_feedback +
        crh_mod * 20.0;
    double crh_decay = self->k_crh * self->crh;
    double d_crh = (crh_production - crh_decay) * self->dt;
    self->crh = fmax(0.0, fmin(400.0, self->crh + d_crh));

    // === ACTH DYNAMICS ===
    double crh_stimulation = 0.2 * (self->crh - 100.0);
    double acth_production = self->acth_basal_secretion * self->pituitary_mass +
        crh_stimulation -
        self->acth_basal_secretion * total_feedback * 0.5 +
        acth_mod * 10.0;
    double acth_decay = self->k_acth * self->acth;
    double d_acth = (acth_production - acth_decay) * self->dt;
    self->acth = fmax(0.0, fmin(200.0, self->acth + d_acth));

    // === CORTISOL DYNAMICS ===
    double circadian_drive = HPA_get_circadian_amplitude(self);
    double ultradian_pulse = HPA_get_ultradian_pulse(self);
    double acth_stimulation = 0.15 * (self->acth - 25.0);
    double stress_drive = 2.0 * self->stress_level;  // stress_to_cortisol

    double cortisol_production = (circadian_drive / 12.0) * self->cortisol_basal_secretion +
        acth_stimulation * self->adrenal_mass +
        stress_drive +
        ultradian_pulse * 0.3 +
        cortisol_mod * 2.0;
    double cortisol_decay = self->k_cortisol * self->cortisol;
    double d_cortisol = (cortisol_production - cortisol_decay) * self->dt;
    self->cortisol = fmax(0.0, fmin(60.0, self->cortisol + d_cortisol));

    // Update history
    self->cortisol_history[self->history_index] = self->cortisol;
    self->history_index = (self->history_index + 1) % 50;

    // Update gland masses
    HPA_update_gland_masses(self);

    // Update time
    self->time_hours += self->dt;
    if (self->time_hours >= 24.0) {
        self->time_hours -= 24.0;
        self->day++;
    }

    // Update stress (decay + random events)
    self->stress_level = fmax(0.0, self->stress_level * 0.98 - 0.05);

    // Random stress events (2% chance)
    if (rand_uniform(0, 1) < 0.02) {
        double r = rand_uniform(0, 1);
        double stress_mag;
        if (r < 0.6) stress_mag = 2.0;
        else if (r < 0.9) stress_mag = 5.0;
        else stress_mag = 8.0;

        self->stress_level = fmin(10.0, self->stress_level + stress_mag);
    }

    self->current_step++;

    // Calculate allostatic load
    HPA_calculate_receptor_occupancy(self, HPA_cortisol_to_nmol(self->cortisol),
        &mr_occ, &gr_occ);
    double allostatic_load = HPA_calculate_allostatic_load(self, mr_occ, gr_occ);
    self->cumulative_load += allostatic_load;

    // Reward = minimize load
    *reward = -allostatic_load + 5.0;

    // Check if done
    *done = (self->current_step >= self->max_steps);

    // Get next state
    HPA_get_state(self, next_state);
}

// ========================================================
// Cleanup
// ========================================================
void HPA_destroy(HPA* self) {
    /* No dynamic memory to free in this implementation */
    /* Could be used for cleanup if you add malloc'd components later */
    (void)self;  /* Suppress unused parameter warning */
}

// ========================================================
// Initialize RNG
// ========================================================
void HPA_initialize_random() {
    /* Initialize random number generator */
    srand((unsigned int)time(NULL));
}
