import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random


class HPAEnvironment:
    """
    Physiologically accurate HPA axis environment.

    HPA environment where reward = -allostatic_load
    
    Allostatic load = cumulative biological cost of maintaining homeostasis
    and responding to stressors. High allostatic load → disease.
    
    Based on literature values:
    - Cortisol half-life: ~60-90 minutes
    - ACTH half-life: ~10 minutes
    - CRH half-life: ~15 minutes
    - Cortisol secretion: 20-30 mg/day baseline, up to 300 mg under severe stress
    - Ultradian pulses: ~60-90 minute periodicity
    """
    
    def __init__(self, time_step_hours=0.1, developmental_stage="adult"):
        """
        Initialize with physiologically realistic parameters.
        
        Parameters:
        -----------
        time_step_hours : float
            Simulation time step in hours (default 0.1 = 6 minutes)
        """
        self.dt = time_step_hours
        self.stage = developmental_stage
        
        # Hormone concentrations (physiological units)
        self.cortisol = 12.0  # μg/dL
        self.acth = 25.0      # pg/mL
        self.crh = 100.0      # pg/mL
        
        # Gland masses
        self.pituitary_mass = 1.0
        self.adrenal_mass = 1.0
        
        # Receptor populations
        self.mr_receptors = 1.0
        self.gr_receptors = 1.0
        
        # State variables
        self.stress_level = 0.0
        self.time_hours = 8.0
        self.day = 0
        
        # Ultradian oscillator
        self.ultradian_phase = np.random.uniform(0, 2*np.pi)
        self.ultradian_period = 1.5  # hours
        
        # History
        self.cortisol_history = deque(maxlen=50)
        for _ in range(50):
            self.cortisol_history.append(self.cortisol)
        
        # Episode parameters (vary by developmental stage)
        self.current_step = 0
        self._set_developmental_parameters()
        
        # Setup physiology
        self._setup_physiological_parameters()
        
        # Cumulative load tracking
        self.cumulative_load = 0.0
        
    def _set_developmental_parameters(self):
        """Set parameters based on developmental stage (curriculum learning)."""
        
        if self.stage == "infant":
            # Immature HPA - simpler dynamics, shorter episodes
            self.max_steps = 240  # 24 hours
            self.feedback_maturity = 0.4  # Weak feedback
            self.receptor_sensitivity = 0.6
            self.stress_resilience = 0.5  # High vulnerability
            
        elif self.stage == "child":
            # Developing HPA - intermediate
            self.max_steps = 720  # 3 days
            self.feedback_maturity = 0.7
            self.receptor_sensitivity = 0.8
            self.stress_resilience = 0.7
            
        elif self.stage == "adolescent":
            # Maturing HPA - more complex
            self.max_steps = 1440  # 6 days
            self.feedback_maturity = 0.9
            self.receptor_sensitivity = 0.95
            self.stress_resilience = 0.85
            
        else:  # adult (default)
            # Fully mature HPA - full complexity
            self.max_steps = 2400  # 10 days
            self.feedback_maturity = 1.0
            self.receptor_sensitivity = 1.0
            self.stress_resilience = 1.0
    
    def _setup_physiological_parameters(self):
        """Set up physiological parameters from literature."""
        
        # Half-lives
        self.cortisol_halflife = 1.25  # hours
        self.acth_halflife = 0.17
        self.crh_halflife = 0.25
        
        self.k_cortisol = np.log(2) / self.cortisol_halflife
        self.k_acth = np.log(2) / self.acth_halflife
        self.k_crh = np.log(2) / self.crh_halflife
        
        # Secretion rates
        self.crh_basal_secretion = 50.0
        self.acth_basal_secretion = 15.0
        self.cortisol_basal_secretion = 8.0
        
        # Stress response
        self.stress_to_crh = 10.0
        self.stress_to_cortisol = 2.0
        
        # Receptor binding
        self.mr_kd = 0.5   # nM
        self.gr_kd = 5.0   # nM
        
        # Feedback strengths (modified by maturity)
        self.mr_feedback_strength = 0.3 * self.feedback_maturity
        self.gr_feedback_strength = 0.7 * self.feedback_maturity
        
        # Gland adaptation
        self.gland_growth_rate = 0.001
        self.gland_atrophy_rate = 0.0008
        
        # Optimal ranges (what evolution selected for)
        self.optimal_cortisol = 15.0  # μg/dL (set point)
        self.optimal_acth = 25.0      # pg/mL
        self.optimal_crh = 100.0      # pg/mL
        
        # Tolerance windows (how far from optimal before damage)
        self.cortisol_tolerance = 7.0  # ±7 μg/dL safe zone
        self.acth_tolerance = 15.0
        self.crh_tolerance = 50.0
    
    def reset(self):
        """Reset environment to initial state."""
        self.cortisol = 12.0 + np.random.normal(0, 2)
        self.acth = 25.0 + np.random.normal(0, 5)
        self.crh = 100.0 + np.random.normal(0, 20)
        
        self.pituitary_mass = 1.0
        self.adrenal_mass = 1.0
        self.mr_receptors = 1.0
        self.gr_receptors = 1.0
        
        self.stress_level = np.random.uniform(0, 3)
        self.time_hours = np.random.uniform(0, 24)
        self.day = 0
        self.ultradian_phase = np.random.uniform(0, 2*np.pi)
        
        self.current_step = 0
        self.cumulative_load = 0.0
        
        self.cortisol_history.clear()
        for _ in range(50):
            self.cortisol_history.append(self.cortisol)
        
        return self._get_state()
    
    def _cortisol_to_nmol(self, cortisol_ugdl):
        """Convert cortisol from μg/dL to nM."""
        return cortisol_ugdl * 27.6
    
    def _calculate_receptor_occupancy(self, cortisol_nM):
        """Calculate MR and GR occupancy."""
        mr_occ = cortisol_nM / (self.mr_kd + cortisol_nM)
        gr_occ = cortisol_nM / (self.gr_kd + cortisol_nM)
        return mr_occ, gr_occ
    
    def _calculate_total_feedback(self):
        """Calculate total negative feedback."""
        cortisol_nM = self._cortisol_to_nmol(self.cortisol)
        mr_occ, gr_occ = self._calculate_receptor_occupancy(cortisol_nM)
        
        total_feedback = (
            self.mr_feedback_strength * mr_occ * self.mr_receptors +
            self.gr_feedback_strength * gr_occ * self.gr_receptors
        )
        
        return total_feedback * self.receptor_sensitivity, mr_occ, gr_occ
    
    def _get_circadian_amplitude(self):
        """Circadian rhythm amplitude."""
        phase = 2 * np.pi * (self.time_hours - 8) / 24
        amplitude = 9.0 + 9.0 * np.cos(phase)
        return amplitude
    
    def _get_ultradian_pulse(self):
        """Generate ultradian pulses."""
        self.ultradian_phase += 2 * np.pi * self.dt / self.ultradian_period
        pulse_amplitude = 3.0 * np.sin(self.ultradian_phase)
        pulse_amplitude += np.random.normal(0, 0.5)
        return pulse_amplitude
    
    def _update_gland_masses(self):
        """Update gland masses based on chronic stimulation."""
        # Adrenal growth with high ACTH
        if self.acth > 40:
            self.adrenal_mass += self.gland_growth_rate * self.dt
        elif self.acth < 15:
            self.adrenal_mass -= self.gland_atrophy_rate * self.dt
        
        # Pituitary suppression with high cortisol
        if self.cortisol > 25:
            self.pituitary_mass -= self.gland_atrophy_rate * self.dt
        elif self.cortisol < 10:
            self.pituitary_mass += self.gland_growth_rate * self.dt
        
        self.adrenal_mass = np.clip(self.adrenal_mass, 0.5, 2.0)
        self.pituitary_mass = np.clip(self.pituitary_mass, 0.5, 2.0)
        
        # Receptor regulation
        cortisol_nM = self._cortisol_to_nmol(self.cortisol)
        if cortisol_nM > 100:
            self.gr_receptors *= (1 - 0.0001 * self.dt)
            self.mr_receptors *= (1 - 0.00005 * self.dt)
        else:
            self.gr_receptors += 0.0001 * self.dt * (1.0 - self.gr_receptors)
            self.mr_receptors += 0.00005 * self.dt * (1.0 - self.mr_receptors)
        
        self.gr_receptors = np.clip(self.gr_receptors, 0.3, 1.5)
        self.mr_receptors = np.clip(self.mr_receptors, 0.5, 1.2)
    
    def _get_state(self):
        """Return normalized state vector."""
        total_feedback, mr_occ, gr_occ = self._calculate_total_feedback()
        
        cortisol_avg = np.mean(self.cortisol_history)
        cortisol_trend = (self.cortisol - cortisol_avg) / 10.0
        
        state = np.array([
            self.stress_level / 10.0,
            self.crh / 300.0,
            self.acth / 100.0,
            self.cortisol / 40.0,
            self.time_hours / 24.0,
            cortisol_trend,
            self._get_circadian_amplitude() / 20.0,
            mr_occ,
            gr_occ,
            self.pituitary_mass / 2.0,
            self.adrenal_mass / 2.0,
            self.day / 10.0
        ])
        
        return state
    
    def step(self, action):
        """Execute one time step."""
        # Decode action
        crh_mod = (action % 3 - 1) * 0.3
        acth_mod = ((action // 3) % 3 - 1) * 0.5
        cortisol_mod = ((action // 9) % 3 - 1) * 0.8
        
        # Calculate feedback
        total_feedback, mr_occ, gr_occ = self._calculate_total_feedback()
        
        # === CRH DYNAMICS ===
        crh_production = (
            self.crh_basal_secretion +
            self.stress_to_crh * self.stress_level -
            self.crh_basal_secretion * total_feedback +
            crh_mod * 20
        )
        crh_decay = self.k_crh * self.crh
        d_crh = (crh_production - crh_decay) * self.dt
        self.crh = np.clip(self.crh + d_crh, 0, 400)
        
        # === ACTH DYNAMICS ===
        crh_stimulation = 0.2 * (self.crh - 100)
        acth_production = (
            self.acth_basal_secretion * self.pituitary_mass +
            crh_stimulation -
            self.acth_basal_secretion * total_feedback * 0.5 +
            acth_mod * 10
        )
        acth_decay = self.k_acth * self.acth
        d_acth = (acth_production - acth_decay) * self.dt
        self.acth = np.clip(self.acth + d_acth, 0, 200)
        
        # === CORTISOL DYNAMICS ===
        circadian_drive = self._get_circadian_amplitude()
        ultradian_pulse = self._get_ultradian_pulse()
        acth_stimulation = 0.15 * (self.acth - 25)
        stress_drive = self.stress_to_cortisol * self.stress_level
        
        cortisol_production = (
            (circadian_drive / 12.0) * self.cortisol_basal_secretion +
            acth_stimulation * self.adrenal_mass +
            stress_drive +
            ultradian_pulse * 0.3 +
            cortisol_mod * 2
        )
        
        cortisol_decay = self.k_cortisol * self.cortisol
        d_cortisol = (cortisol_production - cortisol_decay) * self.dt
        self.cortisol = np.clip(self.cortisol + d_cortisol, 0, 60)
        
        # Update history
        self.cortisol_history.append(self.cortisol)
        
        # Update glands
        self._update_gland_masses()
        
        # Update time
        self.time_hours += self.dt
        if self.time_hours >= 24:
            self.time_hours -= 24
            self.day += 1
        
        # Update stress
        self.stress_level = max(0, self.stress_level * 0.98 - 0.05)
        
        # Random stress events
        if np.random.random() < 0.02:
            stress_magnitude = np.random.choice([2, 5, 8], p=[0.6, 0.3, 0.1])
            self.stress_level = min(10, self.stress_level + stress_magnitude)
        
        self.current_step += 1
        
        # === CALCULATE BIOLOGICAL COST (ALLOSTATIC LOAD) ===
        allostatic_load = self._calculate_allostatic_load(mr_occ, gr_occ)
        self.cumulative_load += allostatic_load
        
        # Reward = minimize cost
        reward = -allostatic_load + 5.0  # Offset so healthy baseline is positive
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done
    
    def _calculate_allostatic_load(self, mr_occ, gr_occ):
        """
        Calculate biological cost (allostatic load) per time step.
        
        Allostatic load = cumulative wear and tear on the body from
        maintaining homeostasis and responding to stress.
        
        Components:
        1. Basal metabolic cost (always present)
        2. Deviation cost (quadratic - extremes are exponentially worse)
        3. Tissue-specific damage (dose-dependent on hormone levels)
        4. Receptor dysfunction cost
        5. Gland pathology cost
        6. Instability cost (variance)
        
        Returns:
        --------
        float : Allostatic load (higher = more damage)
        """
        
        # Basal metabolic cost
        # Minimum energy required to maintain HPA axis
        # ~0.1-0.2% of total metabolic rate
        basal_cost = 0.05
        

        # Cortisol deviation cost (Quadratic)
        # Cost increases quadratically as you deviate from optimal
        cortisol_deviation = self.cortisol - self.optimal_cortisol
        
        # If within tolerance window, minimal cost
        if abs(cortisol_deviation) <= self.cortisol_tolerance:
            deviation_cost = 0.01 * (cortisol_deviation / self.cortisol_tolerance) ** 2
        else:
            # Outside tolerance: quadratic cost
            excess = abs(cortisol_deviation) - self.cortisol_tolerance
            deviation_cost = 0.5 * (excess / self.cortisol_tolerance) ** 2
        

        tissue_damage = 0.0
        
        # Hypercortisolism (Cushing's-like)
        if self.cortisol > 25:
            excess_cortisol = self.cortisol - 25
            
            # Metabolic damage (dose-dependent)
            # - Hyperglycemia: 0.1 per μg/dL above 25
            # - Muscle wasting: 0.05 per μg/dL
            # - Bone loss: 0.05 per μg/dL
            # - Immune suppression: 0.1 per μg/dL
            tissue_damage += excess_cortisol * 0.3
            
            # Severe hypercortisolism (>35 μg/dL)
            if self.cortisol > 35:
                # Exponential damage: psychosis, crisis risk
                crisis_multiplier = ((self.cortisol - 35) / 10) ** 2
                tissue_damage += crisis_multiplier * 2.0
        
        # Hypocortisolism (Addison's-like)
        elif self.cortisol < 5:
            deficit = 5 - self.cortisol
            
            # - Hypotension: 0.2 per μg/dL below 5
            # - Hypoglycemia: 0.3 per μg/dL
            # - Inflammation: 0.2 per μg/dL
            tissue_damage += deficit * 0.7
            
            # Crisis risk (< 2 μg/dL)
            if self.cortisol < 2:
                crisis_multiplier = ((2 - self.cortisol) / 2) ** 2
                tissue_damage += crisis_multiplier * 5.0  # High crisis cost
        
        
        # ACTH dysregulation cost
        acth_deviation = abs(self.acth - self.optimal_acth)
        if acth_deviation > self.acth_tolerance:
            excess_acth = acth_deviation - self.acth_tolerance
            acth_cost = 0.02 * (excess_acth / self.acth_tolerance) ** 2
        else:
            acth_cost = 0.0


        # CRH dysregulation cost
        crh_deviation = abs(self.crh - self.optimal_crh)
        if crh_deviation > self.crh_tolerance:
            excess_crh = crh_deviation - self.crh_tolerance
            crh_cost = 0.01 * (excess_crh / self.crh_tolerance) ** 2
        else:
            crh_cost = 0.0
        

        # Loss of receptor sensitivity = loss of feedback control
        # Clinically: glucocorticoid resistance in depression
        
        # MR loss (should be ~70-90% occupied)
        mr_optimal = 0.8
        mr_loss = abs(mr_occ - mr_optimal)
        mr_cost = 0.5 * (mr_loss ** 2)
        
        # GR dysregulation (stress-dependent optimal)
        if self.stress_level > 5:
            gr_optimal = 0.7  # Should be high during stress
        else:
            gr_optimal = 0.3  # Should be low at baseline
        gr_loss = abs(gr_occ - gr_optimal)
        gr_cost = 0.3 * (gr_loss ** 2)
        
        # Receptor downregulation (chronic stress marker)
        receptor_downregulation = (
            (1.0 - self.gr_receptors) ** 2 +
            (1.0 - self.mr_receptors) ** 2
        )
        receptor_cost = receptor_downregulation * 0.5
        

        # Hypertrophy or atrophy = pathological adaptation        
        # Adrenal hypertrophy/atrophy
        adrenal_pathology = (self.adrenal_mass - 1.0) ** 2
        
        # Pituitary pathology
        pituitary_pathology = (self.pituitary_mass - 1.0) ** 2
        
        # Severe pathology (>50% change)
        if self.adrenal_mass < 0.5 or self.adrenal_mass > 1.5:
            adrenal_pathology *= 3.0  # Severe dysfunction
        if self.pituitary_mass < 0.5 or self.pituitary_mass > 1.5:
            pituitary_pathology *= 3.0
        
        gland_cost = (adrenal_pathology + pituitary_pathology) * 0.3
        

        # High variability = poor regulation
        # Clinically: labile cortisol in PTSD, panic disorder        
        if len(self.cortisol_history) >= 10:
            recent_cortisol = list(self.cortisol_history)[-10:]
            variance = np.var(recent_cortisol)
            
            # Normalized variance cost
            if variance > 25:  # High instability
                instability_cost = (variance - 25) / 100
            else:
                instability_cost = 0.0
        else:
            instability_cost = 0.0
        

        # Inappropriate stress response = additional cost        
        stress_response_cost = 0.0
        
        # During high stress: cortisol should be elevated
        if self.stress_level > 6:
            expected_cortisol = 20 + self.stress_level * 2
            response_error = abs(self.cortisol - expected_cortisol)
            
            if response_error > 10:
                # Poor stress response (inadequate or excessive)
                stress_response_cost = 0.5 * (response_error / 10) ** 2
        
        # During low stress: cortisol should be normal
        elif self.stress_level < 2:
            if self.cortisol > 25:
                # Inappropriate elevation without stressor
                stress_response_cost = 0.3 * ((self.cortisol - 25) / 10) ** 2
        
        total_allocastic_load = (
            basal_cost +
            deviation_cost +
            tissue_damage +
            acth_cost +
            crh_cost +
            mr_cost +
            gr_cost +
            receptor_cost +
            gland_cost +
            instability_cost +
            stress_response_cost
        )
        
        # Modify by stress resilience (developmental stage factor)
        # Less mature systems are more vulnerable
        vulnerability_factor = 2.0 - self.stress_resilience
        adjusted_load = total_allocastic_load * vulnerability_factor
        
        return adjusted_load
    
    def get_load_components(self):
        """Return breakdown of allostatic load components (for debugging)."""
        total_feedback, mr_occ, gr_occ = self._calculate_total_feedback()
        
        # Recalculate with component tracking
        components = {}
        
        # Would need to modify _calculate_allostatic_load to return components
        # This is a simplified version
        components['cortisol_deviation'] = abs(self.cortisol - self.optimal_cortisol)
        components['gland_pathology'] = abs(self.adrenal_mass - 1.0) + abs(self.pituitary_mass - 1.0)
        components['receptor_dysfunction'] = (2.0 - self.gr_receptors - self.mr_receptors)
        components['cumulative_load'] = self.cumulative_load
        
        return components


class CurriculumDQNAgent:
    """DQN agent with curriculum learning support."""
    
    def __init__(self, state_size, action_size, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = learning_rate
        self.batch_size = 128
        
        # Q-table (persists across curriculum stages)
        self.q_table = {}
        self.learn_step = 0
        
        # Track curriculum progress
        self.stage_history = []
    
    def _discretize_state(self, state):
        """Discretize state."""
        return tuple(np.round(state, 1))
    
    def get_q_values(self, state):
        """Get Q-values for all actions."""
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key]
    
    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self._discretize_state(state)
            next_state_key = self._discretize_state(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            # Q-learning update
            target = reward
            if not done:
                target += self.gamma * np.max(self.q_table[next_state_key])
            
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] += self.learning_rate * (target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.learn_step += 1
    
    def reset_epsilon(self, new_epsilon=None):
        """Reset exploration rate for new curriculum stage."""
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        else:
            # Increase epsilon slightly for new stage (but not back to 1.0)
            self.epsilon = min(self.epsilon * 1.5, 0.3)


def train_curriculum_stage(agent, env, stage_name, episodes, visualize_every=None):
    """
    Train agent on one curriculum stage.
    
    Parameters:
    -----------
    agent : CurriculumDQNAgent
        The RL agent (Q-table persists across stages)
    env : BiologicallyRealisticHPAEnvironment
        Environment for this stage
    stage_name : str
        Name of developmental stage
    episodes : int
        Number of episodes to train
    visualize_every : int or None
        Visualize every N episodes (None = no visualization)
    
    Returns:
    --------
    scores : list
        Episode scores for this stage
    """
    
    print(f"\n{'='*70}")
    print(f"Curriculum stage: {stage_name.upper()}")
    print(f"{'='*70}")
    print(f"Episode length: {env.max_steps * env.dt:.0f} hours ({env.max_steps * env.dt / 24:.1f} days)")
    print(f"Feedback maturity: {env.feedback_maturity:.1%}")
    print(f"Starting epsilon: {agent.epsilon:.4f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"{'='*70}\n")
    
    scores = []
    avg_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_allocastic_load = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            # Track allostatic load
            allostatic_load = 5.0 - reward  # Convert reward back to load
            total_allocastic_load += allostatic_load
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            state = next_state
            step_count += 1
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        avg_scores.append(avg_score)
        
        avg_load_per_hour = total_allocastic_load / (env.max_steps * env.dt)
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{episodes} | "
                  f"Score: {total_reward:.1f} | "
                  f"Avg: {avg_score:.1f} | "
                  f"Load/hr: {avg_load_per_hour:.2f} | "
                  f"ε: {agent.epsilon:.4f}")
    
    # Store stage results
    agent.stage_history.append({
        'stage': stage_name,
        'episodes': episodes,
        'scores': scores,
        'avg_scores': avg_scores,
        'final_epsilon': agent.epsilon,
        'final_q_size': len(agent.q_table)
    })
    
    print(f"\n{stage_name.capitalize()} stage complete!")
    print(f"  Final avg score: {avg_scores[-1]:.1f}")
    print(f"  Q-table size: {len(agent.q_table)} states")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    
    return scores


def plot_curriculum_progress(agent):
    """Plot learning progress across all curriculum stages."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Curriculum Learning Progress', fontsize=14, fontweight='bold')
    
    # Collect data from all stages
    all_scores = []
    all_avgs = []
    stage_boundaries = [0]
    stage_labels = []
    
    for stage_data in agent.stage_history:
        all_scores.extend(stage_data['scores'])
        all_avgs.extend(stage_data['avg_scores'])
        stage_boundaries.append(len(all_scores))
        stage_labels.append(stage_data['stage'])
    
    episodes = range(1, len(all_scores) + 1)
    
    # Plot 1: Scores across all stages
    ax1 = axes[0, 0]
    ax1.plot(episodes, all_scores, alpha=0.3, color='blue', label='Episode Score')
    ax1.plot(episodes, all_avgs, linewidth=2, color='red', label='Avg (50 ep)')
    
    # Mark stage boundaries
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax1.axvline(x=boundary, color='green', linestyle='--', alpha=0.5)
        ax1.text(boundary, ax1.get_ylim()[1] * 0.9, stage_labels[i-1], 
                rotation=90, va='top')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Scores Across Curriculum Stages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Q-table growth
    ax2 = axes[0, 1]
    q_sizes = [stage['final_q_size'] for stage in agent.stage_history]
    stage_names = [stage['stage'] for stage in agent.stage_history]
    ax2.bar(stage_names, q_sizes, color=['lightblue', 'skyblue', 'steelblue', 'darkblue'][:len(stage_names)])
    ax2.set_ylabel('Q-table Size (states)')
    ax2.set_title('Knowledge Accumulation')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Epsilon decay across stages
    ax3 = axes[1, 0]
    epsilon_values = []
    for stage_data in agent.stage_history:
        # Approximate epsilon trajectory for this stage
        initial_eps = 1.0 if not epsilon_values else epsilon_values[-1]
        final_eps = stage_data['final_epsilon']
        stage_eps = np.linspace(initial_eps, final_eps, stage_data['episodes'])
        epsilon_values.extend(stage_eps)
    
    ax3.plot(episodes, epsilon_values, linewidth=2, color='orange')
    for i, boundary in enumerate(stage_boundaries[1:-1], 1):
        ax3.axvline(x=boundary, color='green', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_title('Exploration Strategy')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stage comparison
    ax4 = axes[1, 1]
    stage_final_scores = []
    for stage_data in agent.stage_history:
        final_avg = np.mean(stage_data['scores'][-20:])  # Last 20 episodes
        stage_final_scores.append(final_avg)
    
    colors = ['lightgreen', 'yellowgreen', 'green', 'darkgreen'][:len(stage_names)]
    bars = ax4.bar(stage_names, stage_final_scores, color=colors)
    ax4.set_ylabel('Average Score (last 20 episodes)')
    ax4.set_title('Performance by Stage')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, stage_final_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('C:/Users/ibriy/code/RL/overall_hpa_curriculum_training_stats.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("\nCurriculum training plot saved!")


def test_agent_on_stage(agent, stage_name, n_episodes=3):
    """Test trained agent on a specific stage."""
    
    env = HPAEnvironment(developmental_stage=stage_name)
    
    print(f"\n{'='*70}")
    print(f"Testing {stage_name.upper()} stage")
    print(f"{'='*70}")
    
    test_scores = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Greedy policy
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_allocastic_load = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            allostatic_load = 5.0 - reward
            total_allocastic_load += allostatic_load
            
            total_reward += reward
            state = next_state
        
        test_scores.append(total_reward)
        avg_load_per_hour = total_allocastic_load / (env.max_steps * env.dt)
        
        print(f"  Test {episode + 1}: Score = {total_reward:.1f} | "
              f"Load/hr = {avg_load_per_hour:.2f}")
    
    agent.epsilon = original_epsilon
    
    avg_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    
    print(f"\n  Test Average: {avg_score:.1f} ± {std_score:.1f}")
    print(f"{'='*70}")
    
    return test_scores


def train_with_curriculum(visualize=True):
    """
    Train HPA agent using curriculum learning through developmental stages.
    
    Progression:
    1. Child stage (24 hours, weak feedback) - Learn basics
    2. Adolescent stage (72 hours, developing feedback) - Intermediate
    3. Adult stage (240 hours, full feedback) - Final complexity
    
    Returns:
    --------
    agent : Trained agent with accumulated knowledge
    """
    
    print("\n" + "="*70)
    print("Curriculum Training")
    print("="*70)
    print("\nTraining progression:")
    print("  Stage 1: Child       ->  24 hours/episode  (simple)")
    print("  Stage 2: Adolescent  ->  72 hours/episode  (intermediate)")
    print("  Stage 3: Adult       ->  240 hours/episode (full complexity)")
    print("\nQ-table transfers between stages (transfer learning)")
    print("="*70)
    
    # Initialize agent (state size = 12, action size = 9)
    agent = CurriculumDQNAgent(state_size=12, action_size=9)
    
    env_child = HPAEnvironment(developmental_stage="child")
    print("\nStarting CHILD stage...")
    train_curriculum_stage(
        agent=agent,
        env=env_child,
        stage_name="child",
        episodes=100,  # More episodes at easier difficulty
        visualize_every=None
    )
    # Slight exploration boost for next stage
    agent.reset_epsilon(new_epsilon=0.3)

    env_adolescent = HPAEnvironment(developmental_stage="adolescent")    
    print("\nStarting ADOLESCENT stage...")
    train_curriculum_stage(
        agent=agent,
        env=env_adolescent,
        stage_name="adolescent",
        episodes=150,
        visualize_every=None
    )

    agent.reset_epsilon(new_epsilon=0.2)
    
    env_adult = HPAEnvironment(developmental_stage="adult")
    print("\nStarting ADULT stage...")
    train_curriculum_stage(
        agent=agent,
        env=env_adult,
        stage_name="adult",
        episodes=200,
        visualize_every=None
    )

    if visualize:
        plot_curriculum_progress(agent)
    
    test_agent_on_stage(agent, "child", n_episodes=3)
    test_agent_on_stage(agent, "adolescent", n_episodes=3)
    test_agent_on_stage(agent, "adult", n_episodes=3)
    
    return agent


if __name__ == "__main__":
    agent = train_with_curriculum(visualize=True)
    
    print(f"Final Q-table size: {len(agent.q_table)} states")
    print(f"Stages completed: {len(agent.stage_history)}")
    print("\nKnowledge progression:")
    for i, stage in enumerate(agent.stage_history, 1):
        print(f"  {i}. {stage['stage'].capitalize():12} - "
              f"{stage['final_q_size']:6} states, "
              f"Final score: {stage['avg_scores'][-1]:8.1f}")
    print("=" * 70)
