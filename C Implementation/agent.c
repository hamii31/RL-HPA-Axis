#include "agent.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

 // ========================================================
 // Utility Functions for State Discretization and Hashing
 // ========================================================

 /* Discretize continuous state to integer key (round to 1 decimal place) */
void discretize_state(const double* state, StateKey* key) {
    for (int i = 0; i < HPA_STATE_SIZE; i++) {
        /* Round to 1 decimal place: multiply by 10, round, store as int */
        key->values[i] = (int)round(state[i] * 10.0);
    }
}

/* Hash function for state key (FNV-1a hash) */
unsigned int hash_state(const StateKey* state, int capacity) {
    unsigned int hash = 2166136261u;  /* FNV offset basis */

    for (int i = 0; i < HPA_STATE_SIZE; i++) {
        hash ^= (unsigned int)state->values[i];
        hash *= 16777619u;  /* FNV prime */
    }

    return hash % capacity;
}

/* Compare two state keys for equality */
int state_keys_equal(const StateKey* a, const StateKey* b) {
    for (int i = 0; i < HPA_STATE_SIZE; i++) {
        if (a->values[i] != b->values[i]) {
            return 0;
        }
    }
    return 1;
}

/* Random integer in range [0, max) */
static int rand_int(int max) {
    return rand() % max;
}

/* Random double in range [0, 1] */
static double rand_double() {
    return (double)rand() / (double)RAND_MAX;
}

// ========================================================
// Q-Table Implementation (Hash Table with Chaining)
// ========================================================

/* Create Q-table with initial capacity */
QTable* QTable_create(int initial_capacity) {
    QTable* qtable = (QTable*)malloc(sizeof(QTable));
    if (!qtable) return NULL;

    qtable->capacity = initial_capacity;
    qtable->size = 0;

    /* Allocate buckets (array of pointers) */
    qtable->buckets = (QTableEntry**)calloc(initial_capacity, sizeof(QTableEntry*));
    if (!qtable->buckets) {
        free(qtable);
        return NULL;
    }

    return qtable;
}

/* Destroy Q-table and free all memory */
void QTable_destroy(QTable* qtable) {
    if (!qtable) return;

    /* Free all entries in each bucket */
    for (int i = 0; i < qtable->capacity; i++) {
        QTableEntry* entry = qtable->buckets[i];
        while (entry) {
            QTableEntry* next = entry->next;
            free(entry);
            entry = next;
        }
    }

    free(qtable->buckets);
    free(qtable);
}

/* Rehash Q-table when load factor exceeded */
static void QTable_rehash(QTable* qtable) {
    int old_capacity = qtable->capacity;
    QTableEntry** old_buckets = qtable->buckets;

    /* Double capacity */
    qtable->capacity *= 2;
    qtable->buckets = (QTableEntry**)calloc(qtable->capacity, sizeof(QTableEntry*));
    if (!qtable->buckets) {
        /* Rehash failed - restore old buckets */
        qtable->capacity = old_capacity;
        qtable->buckets = old_buckets;
        return;
    }

    qtable->size = 0;

    /* Rehash all entries */
    for (int i = 0; i < old_capacity; i++) {
        QTableEntry* entry = old_buckets[i];
        while (entry) {
            QTableEntry* next = entry->next;

            /* Find new bucket */
            unsigned int hash = hash_state(&entry->state, qtable->capacity);

            /* Insert at head of new bucket */
            entry->next = qtable->buckets[hash];
            qtable->buckets[hash] = entry;
            qtable->size++;

            entry = next;
        }
    }

    free(old_buckets);
}

/* Get Q-values for a state (creates entry with zeros if not exists) */
double* QTable_get(QTable* qtable, const StateKey* state) {
    unsigned int hash = hash_state(state, qtable->capacity);

    /* Search for existing entry */
    QTableEntry* entry = qtable->buckets[hash];
    while (entry) {
        if (state_keys_equal(&entry->state, state)) {
            return entry->q_values;
        }
        entry = entry->next;
    }

    /* Entry not found - create new one */
    entry = (QTableEntry*)malloc(sizeof(QTableEntry));
    if (!entry) return NULL;

    entry->state = *state;
    memset(entry->q_values, 0, sizeof(entry->q_values));

    /* Insert at head of bucket */
    entry->next = qtable->buckets[hash];
    qtable->buckets[hash] = entry;
    qtable->size++;

    /* Check if rehash needed */
    double load_factor = (double)qtable->size / (double)qtable->capacity;
    if (load_factor > QTABLE_LOAD_FACTOR) {
        QTable_rehash(qtable);
    }

    return entry->q_values;
}

/* Update Q-value for state-action pair */
void QTable_set(QTable* qtable, const StateKey* state, int action, double value) {
    double* q_values = QTable_get(qtable, state);
    if (q_values) {
        q_values[action] = value;
    }
}

// ========================================================
// Replay Buffer Implementation (Circular Buffer)
// ========================================================

/* Create replay buffer */
ReplayBuffer* ReplayBuffer_create(int capacity) {
    ReplayBuffer* buffer = (ReplayBuffer*)malloc(sizeof(ReplayBuffer));
    if (!buffer) return NULL;

    buffer->buffer = (Experience*)malloc(capacity * sizeof(Experience));
    if (!buffer->buffer) {
        free(buffer);
        return NULL;
    }

    buffer->capacity = capacity;
    buffer->size = 0;
    buffer->index = 0;

    return buffer;
}

/* Destroy replay buffer */
void ReplayBuffer_destroy(ReplayBuffer* buffer) {
    if (!buffer) return;
    free(buffer->buffer);
    free(buffer);
}

/* Add experience to buffer (circular buffer) */
void ReplayBuffer_add(ReplayBuffer* buffer, const Experience* exp) {
    buffer->buffer[buffer->index] = *exp;
    buffer->index = (buffer->index + 1) % buffer->capacity;

    if (buffer->size < buffer->capacity) {
        buffer->size++;
    }
}

/* Sample random batch from buffer */
void ReplayBuffer_sample(ReplayBuffer* buffer, Experience* batch, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        int idx = rand_int(buffer->size);
        batch[i] = buffer->buffer[idx];
    }
}

/* Get current size of buffer */
int ReplayBuffer_size(ReplayBuffer* buffer) {
    return buffer->size;
}

// ========================================================
// Agent Implementation
// ========================================================

/* Create and initialize agent */
Agent* Agent_create(double learning_rate, double gamma) {
    Agent* agent = (Agent*)malloc(sizeof(Agent));
    if (!agent) return NULL;

    /* Create Q-table */
    agent->q_table = QTable_create(QTABLE_INITIAL_CAPACITY);
    if (!agent->q_table) {
        free(agent);
        return NULL;
    }

    /* Create replay buffer */
    agent->replay_buffer = ReplayBuffer_create(REPLAY_BUFFER_SIZE);
    if (!agent->replay_buffer) {
        QTable_destroy(agent->q_table);
        free(agent);
        return NULL;
    }

    /* Set hyperparameters */
    agent->learning_rate = learning_rate;
    agent->gamma = gamma;
    agent->epsilon = 1.0;           /* Start with full exploration */
    agent->epsilon_min = 0.01;      /* Minimum exploration */
    agent->epsilon_decay = 0.9995;  /* Decay per learning step */

    /* Initialize statistics */
    agent->learn_step = 0;
    agent->total_steps = 0;
    agent->current_stage = 0;  /* Start at child stage */

    return agent;
}

/* Destroy agent and free memory */
void Agent_destroy(Agent* agent) {
    if (!agent) return;

    QTable_destroy(agent->q_table);
    ReplayBuffer_destroy(agent->replay_buffer);
    free(agent);
}

/* Get Q-values for a state */
void Agent_get_q_values(Agent* agent, const double* state, double* q_values) {
    StateKey key;
    discretize_state(state, &key);

    double* stored_q = QTable_get(agent->q_table, &key);
    if (stored_q) {
        memcpy(q_values, stored_q, HPA_ACTION_SIZE * sizeof(double));
    }
    else {
        memset(q_values, 0, HPA_ACTION_SIZE * sizeof(double));
    }
}

/* Select action using epsilon-greedy policy */
int Agent_act(Agent* agent, const double* state) {
    /* Epsilon-greedy: explore with probability epsilon */
    if (rand_double() < agent->epsilon) {
        /* Random action (exploration) */
        return rand_int(HPA_ACTION_SIZE);
    }

    /* Greedy action (exploitation) */
    double q_values[HPA_ACTION_SIZE];
    Agent_get_q_values(agent, state, q_values);

    /* Find action with maximum Q-value */
    int best_action = 0;
    double best_q = q_values[0];

    for (int a = 1; a < HPA_ACTION_SIZE; a++) {
        if (q_values[a] > best_q) {
            best_q = q_values[a];
            best_action = a;
        }
    }

    agent->total_steps++;
    return best_action;
}

/* Store experience in replay buffer */
void Agent_remember(Agent* agent, const double* state, int action,
    double reward, const double* next_state, int done) {
    Experience exp;

    memcpy(exp.state, state, HPA_STATE_SIZE * sizeof(double));
    exp.action = action;
    exp.reward = reward;
    memcpy(exp.next_state, next_state, HPA_STATE_SIZE * sizeof(double));
    exp.done = done;

    ReplayBuffer_add(agent->replay_buffer, &exp);
}

/* Learn from batch of experiences (Q-learning update) */
void Agent_replay(Agent* agent) {
    /* Need at least BATCH_SIZE experiences */
    if (ReplayBuffer_size(agent->replay_buffer) < BATCH_SIZE) {
        return;
    }

    /* Allocate batch on heap to avoid large stack usage */
    Experience* batch = (Experience*)malloc(BATCH_SIZE * sizeof(Experience));
    if (!batch) {
        /* Allocation failed — skip learning step */
        return;
    }

    /* Sample random batch */
    ReplayBuffer_sample(agent->replay_buffer, batch, BATCH_SIZE);

    /* Q-learning update for each experience */
    for (int i = 0; i < BATCH_SIZE; i++) {
        Experience* exp = &batch[i];

        /* Discretize states */
        StateKey state_key, next_state_key;
        discretize_state(exp->state, &state_key);
        discretize_state(exp->next_state, &next_state_key);

        /* Get Q-values */
        double* q_values = QTable_get(agent->q_table, &state_key);
        double* next_q_values = QTable_get(agent->q_table, &next_state_key);

        if (!q_values || !next_q_values) continue;

        /* Calculate TD target */
        double target = exp->reward;

        if (!exp->done) {
            /* Find max Q-value for next state */
            double max_next_q = next_q_values[0];
            for (int a = 1; a < HPA_ACTION_SIZE; a++) {
                if (next_q_values[a] > max_next_q) {
                    max_next_q = next_q_values[a];
                }
            }

            target += agent->gamma * max_next_q;
        }

        /* Q-learning update: Q(s,a) ← Q(s,a) + α[target - Q(s,a)] */
        double current_q = q_values[exp->action];
        double new_q = current_q + agent->learning_rate * (target - current_q);

        QTable_set(agent->q_table, &state_key, exp->action, new_q);
    }

    free(batch);

    /* Decay epsilon */
    if (agent->epsilon > agent->epsilon_min) {
        agent->epsilon *= agent->epsilon_decay;
        if (agent->epsilon < agent->epsilon_min) {
            agent->epsilon = agent->epsilon_min;
        }
    }

    agent->learn_step++;
}

/* Reset exploration rate for new curriculum stage */
void Agent_reset_epsilon(Agent* agent, double new_epsilon) {
    if (new_epsilon >= 0) {
        agent->epsilon = new_epsilon;
    }
    else {
        /* Boost epsilon for new stage (but not back to 1.0) */
        agent->epsilon = fmin(agent->epsilon * 1.5, 0.3);
    }
}

/* Get size of Q-table (number of states learned) */
int Agent_get_qtable_size(Agent* agent) {
    return agent->q_table->size;
}

// ========================================================
// Save Q-table to file and Load Q-table from file
// ========================================================

/* Save Q-table to binary file */
int Agent_save_qtable(Agent* agent, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return 0;

    /* Write header */
    int size = agent->q_table->size;
    fwrite(&size, sizeof(int), 1, fp);

    /* Write all Q-table entries */
    for (int i = 0; i < agent->q_table->capacity; i++) {
        QTableEntry* entry = agent->q_table->buckets[i];
        while (entry) {
            fwrite(&entry->state, sizeof(StateKey), 1, fp);
            fwrite(entry->q_values, sizeof(double), HPA_ACTION_SIZE, fp);
            entry = entry->next;
        }
    }

    fclose(fp);
    return 1;
}

/* Load Q-table from binary file */
int Agent_load_qtable(Agent* agent, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return 0;

    /* Read header */
    int size;
    if (fread(&size, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    /* Clear existing Q-table */
    QTable_destroy(agent->q_table);
    agent->q_table = QTable_create(size * 2);  /* Create with enough capacity */

    /* Read all entries */
    for (int i = 0; i < size; i++) {
        StateKey state;
        double q_values[HPA_ACTION_SIZE];

        if (fread(&state, sizeof(StateKey), 1, fp) != 1) break;
        if (fread(q_values, sizeof(double), HPA_ACTION_SIZE, fp) != HPA_ACTION_SIZE) break;

        /* Insert into Q-table */
        for (int a = 0; a < HPA_ACTION_SIZE; a++) {
            QTable_set(agent->q_table, &state, a, q_values[a]);
        }
    }

    fclose(fp);
    return 1;
}
