#ifndef TCW_RUNTIME_H
#define TCW_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================
 * TCW-Runtime v0.6 — real TCW-Life with learnable PHI
 *
 * PHI = residual MLP (1 -> 8 -> 1, tanh activation):
 *   z_i    = tanh(w1_i * x + b1_i)     i = 0..7
 *   PHI(x) = sum(w2_i * z_i) + b2
 *
 * Total weights: 8 + 8 + 8 + 1 = 25 float = 100 bytes.
 * E_topo (metabolism) trains PHI via SGD; C (model) stays frozen.
 * ============================================================ */

#define TCW_MLP_HIDDEN     8
#define TCW_MLP_N_WEIGHTS  25                                   /* 8+8+8+1 */
#define TCW_PHI_SIZE_BYTES (TCW_MLP_N_WEIGHTS * sizeof(float))  /* 100 */

typedef enum {
    TCW_OK = 0,
    TCW_ERR_INVALID_ARG = -1,
    TCW_ERR_CHECKSUM_MISMATCH = -2,
    TCW_ERR_DEAD = -3,
} tcw_status_t;

typedef struct {
    const uint8_t *data;
    size_t         size;
    uint32_t       crc32;
} tcw_genome_t;

typedef struct {
    uint8_t __attribute__((aligned(4))) working[TCW_PHI_SIZE_BYTES];
    uint8_t __attribute__((aligned(4))) backup [TCW_PHI_SIZE_BYTES];
    size_t   size;
    uint16_t rank;
    uint16_t max_rank;
    bool     alive;
    uint32_t checksum;
} tcw_phi_t;

typedef struct {
    float alpha;
    float mutation_rate;
} tcw_gate_t;

typedef struct {
    tcw_genome_t genome;
    tcw_phi_t    phi;
    tcw_gate_t   gate;
} tcw_agent_t;

/* MLP layout — must match byte size of PHI exactly. */
typedef struct {
    float w1[TCW_MLP_HIDDEN];    /* 8 */
    float b1[TCW_MLP_HIDDEN];    /* 8 */
    float w2[TCW_MLP_HIDDEN];    /* 8 */
    float b2;                     /* 1 */
} tcw_mlp_t;

_Static_assert(sizeof(tcw_mlp_t) == TCW_PHI_SIZE_BYTES,
               "tcw_mlp_t must be exactly TCW_PHI_SIZE_BYTES");

/* ------- MLP forward (used by main.c for inference) ------- */
float tcw_mlp_forward(const tcw_mlp_t *m, float x);

/* ------- Lifecycle operations ------- */
tcw_status_t tcw_init(tcw_agent_t *agent,
                      const uint8_t *genome_data, size_t genome_size,
                      uint16_t max_rank);

/* E_topo: metabolism. Train PHI via SGD. C (model) frozen. */
tcw_status_t tcw_etopo(tcw_agent_t *agent,
                       const float *x_data, const float *y_target,
                       size_t n, float lr, int epochs);

/* R: restore PHI from backup (memcpy) */
tcw_status_t tcw_repair(tcw_agent_t *agent);

/* A: aging. Zero smallest-|w| weights, reduce rank, sync backup. */
tcw_status_t tcw_aging(tcw_agent_t *agent, uint16_t delta_rank);

/* M: mutation. Random reinit (new environment). */
tcw_status_t tcw_mutation(tcw_agent_t *agent, uint32_t seed);

/* P: proliferation. Children with per-weight dropout probability. */
tcw_status_t tcw_proliferation(const tcw_agent_t *parent,
                               tcw_agent_t *children, size_t n_children,
                               uint32_t seed, float dropout_p);

/* Damage: XOR bytes in working (test helper). */
tcw_status_t tcw_damage(tcw_agent_t *agent, size_t num_bytes, uint32_t seed);

uint32_t tcw_crc32(const uint8_t *data, size_t len);

#endif /* TCW_RUNTIME_H */
