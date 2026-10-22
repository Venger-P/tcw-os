#include "tcw_runtime.h"
#include <string.h>
#include <math.h>

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

uint32_t tcw_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
        }
    }
    return ~crc;
}

static inline tcw_mlp_t *phi_mlp(tcw_agent_t *a) {
    return (tcw_mlp_t *)a->phi.working;
}
static inline const tcw_mlp_t *phi_mlp_const(const tcw_agent_t *a) {
    return (const tcw_mlp_t *)a->phi.working;
}

float tcw_mlp_forward(const tcw_mlp_t *m, float x) {
    float y = m->b2;
    for (int i = 0; i < TCW_MLP_HIDDEN; i++) {
        float z = tanhf(m->w1[i] * x + m->b1[i]);
        y += m->w2[i] * z;
    }
    return y;
}

tcw_status_t tcw_init(tcw_agent_t *agent,
                      const uint8_t *genome_data, size_t genome_size,
                      uint16_t max_rank) {
    if (!agent || !genome_data || genome_size == 0 || max_rank == 0) {
        return TCW_ERR_INVALID_ARG;
    }
    agent->genome.data  = genome_data;
    agent->genome.size  = genome_size;
    agent->genome.crc32 = tcw_crc32(genome_data, genome_size);

    /* Birth: PHI = zero MLP (y_corr = 0, model used as-is) */
    memset(agent->phi.working, 0, TCW_PHI_SIZE_BYTES);
    memset(agent->phi.backup,  0, TCW_PHI_SIZE_BYTES);

    agent->phi.size     = TCW_PHI_SIZE_BYTES;
    agent->phi.rank     = max_rank;
    agent->phi.max_rank = max_rank;
    agent->phi.alive    = true;
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);

    agent->gate.alpha         = 1.0f;
    agent->gate.mutation_rate = 1e-4f;
    return TCW_OK;
}

/* ============================================================
 * E_topo: metabolism. SGD on PHI, C frozen.
 * Loss: L = 0.5 * (y_pred - y_target)^2
 * Grad: dL/dy = (y_pred - y_target)
 * ============================================================ */
tcw_status_t tcw_etopo(tcw_agent_t *agent,
                       const float *x_data, const float *y_target,
                       size_t n, float lr, int epochs) {
    if (!agent || !x_data || !y_target || n == 0) return TCW_ERR_INVALID_ARG;
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    tcw_mlp_t *m = phi_mlp(agent);

    for (int ep = 0; ep < epochs; ep++) {
        for (size_t i = 0; i < n; i++) {
            float x  = x_data[i];
            float yt = y_target[i];

            /* Forward */
            float z[TCW_MLP_HIDDEN];
            float y = m->b2;
            for (int j = 0; j < TCW_MLP_HIDDEN; j++) {
                z[j] = tanhf(m->w1[j] * x + m->b1[j]);
                y += m->w2[j] * z[j];
            }

            /* Backward */
            float dy = y - yt;
            for (int j = 0; j < TCW_MLP_HIDDEN; j++) {
                float dz = dy * m->w2[j] * (1.0f - z[j] * z[j]);
                m->w2[j] -= lr * dy * z[j];
                m->w1[j] -= lr * dz * x;
                m->b1[j] -= lr * dz;
            }
            m->b2 -= lr * dy;
        }
    }

    /* New knowledge is now the "healthy" state */
    memcpy(agent->phi.backup, agent->phi.working, TCW_PHI_SIZE_BYTES);
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

tcw_status_t tcw_repair(tcw_agent_t *agent) {
    if (!agent) return TCW_ERR_INVALID_ARG;
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    uint32_t backup_crc = tcw_crc32(agent->phi.backup, TCW_PHI_SIZE_BYTES);
    memcpy(agent->phi.working, agent->phi.backup, TCW_PHI_SIZE_BYTES);
    agent->phi.checksum = backup_crc;

    if (tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES) != agent->phi.checksum) {
        return TCW_ERR_CHECKSUM_MISMATCH;
    }
    return TCW_OK;
}

/* ============================================================
 * Aging: zero weights with smallest |w|.
 * Number to zero = (1 - rank/max_rank) * N_WEIGHTS (cumulative).
 * After each aging step, backup is synced (irreversible).
 * ============================================================ */
tcw_status_t tcw_aging(tcw_agent_t *agent, uint16_t delta_rank) {
    if (!agent) return TCW_ERR_INVALID_ARG;
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    uint16_t old = agent->phi.rank;
    uint16_t nw  = (old > delta_rank) ? (uint16_t)(old - delta_rank) : 0;
    agent->phi.rank = nw;
    if (nw == 0) agent->phi.alive = false;

    float *w = (float *)agent->phi.working;
    float frac_dead = 1.0f - (float)nw / (float)agent->phi.max_rank;
    int n_to_zero = (int)(frac_dead * (float)TCW_MLP_N_WEIGHTS);
    if (n_to_zero > TCW_MLP_N_WEIGHTS) n_to_zero = TCW_MLP_N_WEIGHTS;

    /* Iteratively zero the smallest-magnitude non-zero weight */
    for (int k = 0; k < n_to_zero; k++) {
        int min_idx = -1;
        float min_abs = 1e30f;
        for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
            if (w[j] != 0.0f && fabsf(w[j]) < min_abs) {
                min_abs = fabsf(w[j]);
                min_idx = j;
            }
        }
        if (min_idx < 0) break;
        w[min_idx] = 0.0f;
    }

    memcpy(agent->phi.backup, agent->phi.working, TCW_PHI_SIZE_BYTES);
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

/* Mutation: random reinit of all weights, small uniform [-0.1, 0.1] */
tcw_status_t tcw_mutation(tcw_agent_t *agent, uint32_t seed) {
    if (!agent) return TCW_ERR_INVALID_ARG;
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    float *w = (float *)agent->phi.working;
    uint32_t rng = seed ? seed : 42u;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
        int32_t r = (int32_t)xorshift32(&rng);
        w[j] = ((float)r / 2147483648.0f) * 0.1f;
    }

    memcpy(agent->phi.backup, agent->phi.working, TCW_PHI_SIZE_BYTES);
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

/* Proliferation: children with per-weight dropout prob p in [0, 1] */
tcw_status_t tcw_proliferation(const tcw_agent_t *parent,
                               tcw_agent_t *children, size_t n_children,
                               uint32_t seed, float dropout_p) {
    if (!parent || !children || n_children == 0) return TCW_ERR_INVALID_ARG;
    if (!parent->phi.alive) return TCW_ERR_DEAD;
    if (dropout_p < 0.0f) dropout_p = 0.0f;
    if (dropout_p > 1.0f) dropout_p = 1.0f;

    uint32_t rng = seed ? seed : 12345u;
    uint32_t thresh = (uint32_t)(dropout_p * 4294967040.0f);  /* avoid UINT32_MAX edge */

    for (size_t i = 0; i < n_children; i++) {
        children[i].genome = parent->genome;
        children[i].gate   = parent->gate;

        memcpy(children[i].phi.working, parent->phi.working, TCW_PHI_SIZE_BYTES);
        memcpy(children[i].phi.backup,  parent->phi.backup,  TCW_PHI_SIZE_BYTES);

        float *w = (float *)children[i].phi.working;
        for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
            if (xorshift32(&rng) < thresh) {
                w[j] = 0.0f;
            }
        }

        children[i].phi.size     = parent->phi.size;
        children[i].phi.rank     = parent->phi.rank;
        children[i].phi.max_rank = parent->phi.max_rank;
        children[i].phi.alive    = true;
        children[i].phi.checksum = tcw_crc32(children[i].phi.working, TCW_PHI_SIZE_BYTES);
    }
    return TCW_OK;
}

tcw_status_t tcw_damage(tcw_agent_t *agent, size_t num_bytes, uint32_t seed) {
    if (!agent) return TCW_ERR_INVALID_ARG;
    if (num_bytes > TCW_PHI_SIZE_BYTES) num_bytes = TCW_PHI_SIZE_BYTES;

    uint32_t rng = seed ? seed : 42u;
    for (size_t i = 0; i < num_bytes; i++) {
        agent->phi.working[i] ^= (uint8_t)xorshift32(&rng);
    }
    return TCW_OK;
}
