#ifndef TCW_RUNTIME_H
#define TCW_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define TCW_MLP_HIDDEN     8
#define TCW_MLP_N_WEIGHTS  25
#define TCW_PHI_SIZE_BYTES (TCW_MLP_N_WEIGHTS * sizeof(float))

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
    float alpha;         /* b: gate/expression regulator, multiplies PHI output */
    float mutation_rate;
} tcw_gate_t;

typedef struct {
    tcw_genome_t genome;
    tcw_phi_t    phi;
    tcw_gate_t   gate;
} tcw_agent_t;

typedef struct {
    float w1[TCW_MLP_HIDDEN];
    float b1[TCW_MLP_HIDDEN];
    float w2[TCW_MLP_HIDDEN];
    float b2;
} tcw_mlp_t;

_Static_assert(sizeof(tcw_mlp_t) == TCW_PHI_SIZE_BYTES,
               "tcw_mlp_t must match TCW_PHI_SIZE_BYTES");

float tcw_mlp_forward(const tcw_mlp_t *m, float x);

tcw_status_t tcw_init(tcw_agent_t *agent,
                      const uint8_t *genome_data, size_t genome_size,
                      uint16_t max_rank);

/* E_topo: basic SGD on PHI, C frozen */
tcw_status_t tcw_etopo(tcw_agent_t *agent,
                       const float *x_data, const float *y_target,
                       size_t n, float lr, int epochs);

/* E_topo + EWC: adds penalty λ·F_j·(w_j - θ_old_j) */
tcw_status_t tcw_etopo_ewc(tcw_agent_t *agent,
                            const float *x_data, const float *y_target,
                            size_t n, float lr, int epochs,
                            const float *fisher, const float *theta_old,
                            float lambda);

/* Fisher diagonal: F_j = E[(∂L/∂w_j)^2] over given data */
tcw_status_t tcw_compute_fisher(const tcw_agent_t *agent,
                                 const float *x_data, const float *y_target,
                                 size_t n, float *fisher);

/* Composition (⊕): out = α·A + (1-α)·B (weighted weight average) */
tcw_status_t tcw_composition(tcw_agent_t *out,
                             const tcw_agent_t *a, const tcw_agent_t *b,
                             float alpha);

tcw_status_t tcw_repair(tcw_agent_t *agent);
tcw_status_t tcw_aging(tcw_agent_t *agent, uint16_t delta_rank);
tcw_status_t tcw_mutation(tcw_agent_t *agent, uint32_t seed);

tcw_status_t tcw_proliferation(const tcw_agent_t *parent,
                               tcw_agent_t *children, size_t n_children,
                               uint32_t seed, float dropout_p);

tcw_status_t tcw_damage(tcw_agent_t *agent, size_t num_bytes, uint32_t seed);
uint32_t     tcw_crc32(const uint8_t *data, size_t len);

#endif
