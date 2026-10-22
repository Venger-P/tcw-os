#include "tcw_runtime.h"
#include <string.h>

/* ============================================================
 * TCW-Runtime v0.3
 *
 * v0.3 fix vs v0.2:
 *   tcw_aging() now syncs backup with working after aging.
 *   Rationale (biology): a 40-year-old heals in a 40-year-old body,
 *   not in a 20-year-old one. Without this fix, repair would "youthify"
 *   the agent by restoring pre-aged backup, cancelling aging entirely.
 * ============================================================ */

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

tcw_status_t tcw_init(tcw_agent_t *agent,
                      const uint8_t *genome_data, size_t genome_size,
                      uint16_t max_rank) {
    if (!agent || !genome_data || genome_size == 0 || max_rank == 0) {
        return TCW_ERR_INVALID_ARG;
    }
    agent->genome.data  = genome_data;
    agent->genome.size  = genome_size;
    agent->genome.crc32 = tcw_crc32(genome_data, genome_size);

    size_t copy_size = genome_size < TCW_PHI_SIZE_BYTES ? genome_size : TCW_PHI_SIZE_BYTES;
    memset(agent->phi.working, 0, TCW_PHI_SIZE_BYTES);
    memcpy(agent->phi.working, genome_data, copy_size);
    memcpy(agent->phi.backup, agent->phi.working, TCW_PHI_SIZE_BYTES);

    agent->phi.size     = TCW_PHI_SIZE_BYTES;
    agent->phi.rank     = max_rank;
    agent->phi.max_rank = max_rank;
    agent->phi.alive    = true;
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);

    agent->gate.alpha         = 1.0f;
    agent->gate.mutation_rate = 1e-4f;
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

tcw_status_t tcw_aging(tcw_agent_t *agent, uint16_t delta_rank) {
    if (!agent) return TCW_ERR_INVALID_ARG;
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    uint16_t old_rank = agent->phi.rank;
    uint16_t new_rank = (old_rank > delta_rank) ? (uint16_t)(old_rank - delta_rank) : 0;
    uint16_t lost     = (uint16_t)(old_rank - new_rank);

    size_t bytes_to_zero = ((size_t)TCW_PHI_SIZE_BYTES * lost) / agent->phi.max_rank;
    if (bytes_to_zero > TCW_PHI_SIZE_BYTES) bytes_to_zero = TCW_PHI_SIZE_BYTES;

    /* Aging: zero out "high-rank" bytes (simulates lost knowledge) */
    memset(&agent->phi.working[TCW_PHI_SIZE_BYTES - bytes_to_zero], 0, bytes_to_zero);

    agent->phi.rank = new_rank;
    if (new_rank == 0) agent->phi.alive = false;

    /* v0.3 FIX: sync backup with working, so repair restores AGED state,
       not the youthful one. Aging becomes irreversible — as in biology. */
    memcpy(agent->phi.backup, agent->phi.working, TCW_PHI_SIZE_BYTES);

    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

tcw_status_t tcw_mutation(tcw_agent_t *agent,
                          const uint8_t *new_phi, size_t new_size) {
    if (!agent || !new_phi || new_size > TCW_PHI_SIZE_BYTES) {
        return TCW_ERR_INVALID_ARG;
    }
    if (!agent->phi.alive) return TCW_ERR_DEAD;

    memset(agent->phi.working, 0, TCW_PHI_SIZE_BYTES);
    memcpy(agent->phi.working, new_phi, new_size);
    memcpy(agent->phi.backup,  agent->phi.working, TCW_PHI_SIZE_BYTES);
    agent->phi.checksum = tcw_crc32(agent->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

tcw_status_t tcw_proliferation(const tcw_agent_t *parent,
                               tcw_agent_t *children, size_t n_children,
                               uint32_t seed) {
    if (!parent || !children || n_children == 0) return TCW_ERR_INVALID_ARG;
    if (!parent->phi.alive) return TCW_ERR_DEAD;

    uint32_t rng = seed ? seed : 12345u;
    for (size_t i = 0; i < n_children; i++) {
        children[i].genome = parent->genome;
        children[i].gate   = parent->gate;

        memcpy(children[i].phi.working, parent->phi.working, TCW_PHI_SIZE_BYTES);
        for (size_t j = 0; j < TCW_PHI_SIZE_BYTES; j++) {
            if ((xorshift32(&rng) & 0x1Fu) == 0) {
                children[i].phi.working[j] = 0;
            }
        }
        memcpy(children[i].phi.backup, children[i].phi.working, TCW_PHI_SIZE_BYTES);
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
