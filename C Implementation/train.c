#include "hpa.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Stage configuration */
typedef struct {
    const char* name;
    DevelopmentalStage stage;
    int episodes;
    double epsilon_boost;
} StageConfig;

/* Train one curriculum stage */
void train_stage(Agent* agent, const StageConfig* config) {
    printf("\n");
    printf("========================================================================\n");
    printf("Curriculum stage: %s\n", config->name);
    printf("========================================================================\n");
    
    /* Create environment for this stage */
    HPA env;
    HPA_init(&env, 0.1, config->stage);
    
    printf("Episode length:     %d steps (%.0f hours = %.1f days)\n",
           env.max_steps, env.max_steps * env.dt, env.max_steps * env.dt / 24.0);
    printf("Feedback maturity:  %.0f%%\n", env.feedback_maturity * 100);
    printf("Stress resilience:  %.0f%%\n", env.stress_resilience * 100);
    printf("Starting epsilon:   %.4f\n", agent->epsilon);
    printf("Q-table size:       %d states\n", Agent_get_qtable_size(agent));
    printf("Training episodes:  %d\n", config->episodes);
    printf("========================================================================\n\n");
    
    /* Training statistics */
    double* scores = (double*)malloc(config->episodes * sizeof(double));
    if (!scores) {
        printf("ERROR: Failed to allocate memory for scores\n");
        return;
    }
    
    /* Training loop */
    for (int episode = 0; episode < config->episodes; episode++) {
        /* Reset environment */
        double state[HPA_STATE_SIZE];
        HPA_reset(&env, state);
        
        double total_reward = 0.0;
        double total_load = 0.0;
        int done = 0;
        int steps = 0;
        
        /* Episode loop */
        while (!done) {
            /* Agent acts */
            int action = Agent_act(agent, state);
            
            /* Environment steps */
            double next_state[HPA_STATE_SIZE];
            double reward;
            HPA_step(&env, action, next_state, &reward, &done);
            
            /* Remember and learn */
            Agent_remember(agent, state, action, reward, next_state, done);
            Agent_replay(agent);
            
            /* Statistics */
            total_reward += reward;
            total_load += (5.0 - reward);
            steps++;
            
            /* Update state */
            memcpy(state, next_state, sizeof(state));
        }
        
        scores[episode] = total_reward;
        
        /* Print progress every 10 episodes */
        if ((episode + 1) % 10 == 0 || episode == config->episodes - 1) {
            /* Calculate average over last 50 episodes (or less) */
            int window = 50;
            int start = (episode >= window - 1) ? episode - window + 1 : 0;
            double avg_score = 0.0;
            for (int i = start; i <= episode; i++) {
                avg_score += scores[i];
            }
            avg_score /= (episode - start + 1);
            
            double avg_load_per_hour = total_load / (env.max_steps * env.dt);
            
            printf("  Episode %3d/%d | Score: %7.1f | Avg: %7.1f | Load/hr: %.2f | ε: %.4f | Q-size: %d\n",
                   episode + 1, config->episodes, total_reward, avg_score,
                   avg_load_per_hour, agent->epsilon, Agent_get_qtable_size(agent));
        }
    }
    
    /* Calculate final statistics */
    int window = (config->episodes >= 20) ? 20 : config->episodes;
    double final_avg = 0.0;
    for (int i = config->episodes - window; i < config->episodes; i++) {
        final_avg += scores[i];
    }
    final_avg /= window;
    
    printf("\n%s stage complete!\n", config->name);
    printf("  Final avg score (last %d episodes): %.1f\n", window, final_avg);
    printf("  Q-table size: %d states\n", Agent_get_qtable_size(agent));
    printf("  Final epsilon: %.4f\n", agent->epsilon);
    
    free(scores);
    HPA_destroy(&env);
}

/* Test agent on a specific stage */
void test_stage(Agent* agent, const char* stage_name, DevelopmentalStage stage, int n_tests) {
    printf("\n");
    printf("========================================================================\n");
    printf("Testing on %s stage\n", stage_name);
    printf("========================================================================\n");
    
    /* Create environment */
    HPA env;
    HPA_init(&env, 0.1, stage);
    
    /* Save and disable exploration */
    double original_epsilon = agent->epsilon;
    agent->epsilon = 0.0;

	/* Check if n_tests is valid */
    if (n_tests <= 0) {
        printf("ERROR: n_tests must be > 0\n");
        agent->epsilon = original_epsilon;
        HPA_destroy(&env);
        return;
    }
    
    double* test_scores = (double*)malloc(n_tests * sizeof(double));

	/* Check if memory allocation succeeded */
    if (!test_scores) {
        printf("ERROR: Failed to allocate memory for test scores (%d tests)\n", n_tests);
        agent->epsilon = original_epsilon;
        HPA_destroy(&env);
        return;
    }
    
    for (int test = 0; test < n_tests; test++) {
        double state[HPA_STATE_SIZE];
        HPA_reset(&env, state);
        
        double total_reward = 0.0;
        double total_load = 0.0;
        int done = 0;
        int steps = 0;
        
        while (!done) {
            int action = Agent_act(agent, state);
            
            double next_state[HPA_STATE_SIZE];
            double reward;
            HPA_step(&env, action, next_state, &reward, &done);
            
            total_reward += reward;
            total_load += (5.0 - reward);
            steps++;
            
            memcpy(state, next_state, sizeof(state));
        }
        
        test_scores[test] = total_reward;
        double avg_load_per_hour = total_load / (env.max_steps * env.dt);
        
        printf("  Test %d: Score = %7.1f | Load/hr = %.2f\n",
               test + 1, total_reward, avg_load_per_hour);
    }
    
    /* Calculate average and std dev */
    double avg = 0.0, variance = 0.0;
    for (int i = 0; i < n_tests; i++) {
        avg += test_scores[i];
    }
    avg /= n_tests;
    
    for (int i = 0; i < n_tests; i++) {
        double diff = test_scores[i] - avg;
        variance += diff * diff;
    }
    variance /= n_tests;
    double stddev = sqrt(variance);
    
    printf("\n  Test Average: %.1f (+- %.1f)\n", avg, stddev);
    printf("========================================================================\n");
    
    free(test_scores);
    agent->epsilon = original_epsilon;
    HPA_destroy(&env);
}

int main() {
    printf("\n");
    printf("========================================================================\n");
    printf("HPA Axis Curriculum training\n");
    printf("========================================================================\n");
    printf("\nTraining progression:\n");
	printf("  Stage 1: Child       -  96 hours/episode  (4 days)\n"); // Early-life stress adaptations have lasting effects, so we keep episodes shorter to focus on learning basic regulation without overwhelming the agent
	printf("  Stage 2: Adolescent  -  168 hours/episode  (1 week)\n"); // Adolescence is a critical period for HPA axis maturation, so we increase episode length to allow the agent to learn more complex dynamics and longer-term regulation strategies
	printf("  Stage 3: Adult       -  336 hours/episode (2 weeks)\n"); // Adult stage represents the full complexity of the HPA axis, so we further increase episode length to allow the agent to master long-term regulation and adapt to a wider range of stressors
    printf("\nQ-table transfers between stages (transfer learning)\n");
    printf("========================================================================\n");
    
    /* Initialize random */
    srand((unsigned int)time(NULL));
    HPA_initialize_random();
    
    /* Create agent */
    printf("\nCreating agent...\n");
    Agent* agent = Agent_create(0.0005, 0.98);
    if (!agent) {
        printf("ERROR: Failed to create agent\n");
        return 1;
    }
    printf("  Learning rate: %.4f\n", agent->learning_rate);
    printf("  Discount (gamma): %.2f\n", agent->gamma);
    printf("  Initial epsilon: %.2f\n", agent->epsilon);
    
    /* Define curriculum stages */
    StageConfig stages[3] = {
        {"CHILD",      STAGE_CHILD,      100, 0.3},  /* 100 episodes, boost epsilon to 0.3 to learn more aggressively */
        {"ADOLESCENT", STAGE_ADOLESCENT, 150, 0.2},  /* 150 episodes, boost epsilon to 0.2 */
        {"ADULT",      STAGE_ADULT,      200, -1.0}  /* 200 episodes, no epsilon boost */
    };
    
    // ========================================================
	// Stage 1: CHILD
    // ========================================================
    printf("\nStarting CHILD stage...\n");
    train_stage(agent, &stages[0]);
    
    /* Boost epsilon for next stage */
    Agent_reset_epsilon(agent, stages[1].epsilon_boost);
    
    // ========================================================
	// Stage 2: ADOLESCENT
    // ========================================================
    printf("\nStarting ADOLESCENT stage...\n");
    train_stage(agent, &stages[1]);
    
    /* Boost epsilon for final stage */
    Agent_reset_epsilon(agent, stages[2].epsilon_boost);
    
    // ========================================================
	// Stage 3: ADULT
    // ========================================================
    printf("\nStarting ADULT stage...\n");
    train_stage(agent, &stages[2]);
    
    // ========================================================
    // Test on all stages
    // ========================================================
    printf("\n\n");
    printf("========================================================================\n");
    printf("Testing...\n");
    printf("========================================================================\n");
    
    test_stage(agent, "CHILD", STAGE_CHILD, 3);
    test_stage(agent, "ADOLESCENT", STAGE_ADOLESCENT, 3);
    test_stage(agent, "ADULT", STAGE_ADULT, 3);
    
    // ========================================================
	// Final summary and save Q-table
    // ========================================================
    printf("\n");
    printf("========================================================================\n");
    printf("Training complete!\n");
    printf("========================================================================\n");
    printf("Final Q-table size: %d states\n", Agent_get_qtable_size(agent));
    printf("Total learning steps: %d\n", agent->learn_step);
    printf("Final epsilon: %.4f\n", agent->epsilon);
    
    printf("\nKnowledge progression:\n");
    printf("  1. Child stage      - Learned basic regulation\n");
    printf("  2. Adolescent stage - Refined control, longer episodes\n");
    printf("  3. Adult stage      - Mastered full complexity\n");
    printf("========================================================================\n");
    
    /* Save Q-table */
    printf("\nSaving Q-table to 'curriculum_qtable.dat'...\n");
    if (Agent_save_qtable(agent, "curriculum_qtable.dat")) {
        printf("  Q-table saved successfully!\n");
    } else {
        printf("  Failed to save Q-table\n");
    }
    
    /* Cleanup */
    Agent_destroy(agent);
    
    printf("\n✓ Training complete!\n\n");
    
    return 0;
}
