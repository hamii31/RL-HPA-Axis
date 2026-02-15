#ifndef AGENT_H
#define AGENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hpa.h"

// ========================================================
// Hyperparameters and Data Structures for Q-Learning Agent
// ========================================================

#define QTABLE_INITIAL_CAPACITY 10000   /* Initial hash table capacity */
#define QTABLE_LOAD_FACTOR 0.75         /* Rehash when 75% full */
#define REPLAY_BUFFER_SIZE 20000        /* Experience replay capacity */
#define BATCH_SIZE 128                  /* Mini-batch size for training */

/* Discretized state key (for Q-table lookup) */
    typedef struct {
        int values[HPA_STATE_SIZE];  /* Discretized state values */
    } StateKey;

    /* Q-table entry (state-action values) */
    typedef struct QTableEntry {
        StateKey state;
        double q_values[HPA_ACTION_SIZE];
        struct QTableEntry* next;  /* For hash table chaining */
    } QTableEntry;

    /* Q-table (hash table) */
    typedef struct {
        QTableEntry** buckets;
        int capacity;
        int size;  /* Number of states stored */
    } QTable;

    /* Experience (for replay buffer) */
    typedef struct {
        double state[HPA_STATE_SIZE];
        int action;
        double reward;
        double next_state[HPA_STATE_SIZE];
        int done;
    } Experience;

    /* Experience replay buffer */
    typedef struct {
        Experience* buffer;
        int capacity;
        int size;
        int index;  /* Circular buffer index */
    } ReplayBuffer;

    /* Q-Learning Agent */
    typedef struct {
        /* Q-table */
        QTable* q_table;

        /* Experience replay */
        ReplayBuffer* replay_buffer;

        /* Hyperparameters */
        double gamma;          /* Discount factor (0.98) */
        double learning_rate;  /* Alpha (0.0005) */
        double epsilon;        /* Exploration rate (1.0 → 0.01) */
        double epsilon_min;    /* Minimum epsilon */
        double epsilon_decay;  /* Decay rate per episode */

        /* Training statistics */
        int learn_step;        /* Number of learning updates */
        int total_steps;       /* Total steps taken */

        /* Curriculum tracking */
        int current_stage;     /* 0=child, 1=adolescent, 2=adult */

    } Agent;

// ========================================================
// Agent Function Declarations
// ========================================================

    /* Create and initialize agent */
    Agent* Agent_create(double learning_rate, double gamma);

    /* Destroy agent and free memory */
    void Agent_destroy(Agent* agent);

    /* Select action using epsilon-greedy policy */
    int Agent_act(Agent* agent, const double* state);

    /* Store experience in replay buffer */
    void Agent_remember(Agent* agent, const double* state, int action,
        double reward, const double* next_state, int done);

    /* Learn from batch of experiences (experience replay) */
    void Agent_replay(Agent* agent);

    /* Reset exploration rate for new curriculum stage */
    void Agent_reset_epsilon(Agent* agent, double new_epsilon);

    /* Get Q-values for a state */
    void Agent_get_q_values(Agent* agent, const double* state, double* q_values);

    /* Get size of Q-table (number of states learned) */
    int Agent_get_qtable_size(Agent* agent);

    /* Save Q-table to file */
    int Agent_save_qtable(Agent* agent, const char* filename);

    /* Load Q-table from file */
    int Agent_load_qtable(Agent* agent, const char* filename);

// ========================================================
// Q-Table Function Declarations
// ========================================================

    /* Create Q-table */
    QTable* QTable_create(int initial_capacity);

    /* Destroy Q-table */
    void QTable_destroy(QTable* qtable);

    /* Get Q-values for a state (creates entry if not exists) */
    double* QTable_get(QTable* qtable, const StateKey* state);

    /* Update Q-value for state-action pair */
    void QTable_set(QTable* qtable, const StateKey* state, int action, double value);

// ========================================================
// Replay Buffer Function Declarations
// ========================================================

    /* Create replay buffer */
    ReplayBuffer* ReplayBuffer_create(int capacity);

    /* Destroy replay buffer */
    void ReplayBuffer_destroy(ReplayBuffer* buffer);

    /* Add experience to buffer */
    void ReplayBuffer_add(ReplayBuffer* buffer, const Experience* exp);

    /* Sample random batch from buffer */
    void ReplayBuffer_sample(ReplayBuffer* buffer, Experience* batch, int batch_size);

    /* Get current size of buffer */
    int ReplayBuffer_size(ReplayBuffer* buffer);

// ========================================================
// Utility Function Declarations
// ========================================================

    /* Discretize continuous state to state key */
    void discretize_state(const double* state, StateKey* key);

    /* Hash function for state key */
    unsigned int hash_state(const StateKey* state, int capacity);

    /* Compare two state keys */
    int state_keys_equal(const StateKey* a, const StateKey* b);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_H */
