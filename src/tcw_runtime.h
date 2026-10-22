#ifndef TCW_RUNTIME_H
#define TCW_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================
 * TCW-Life Runtime — Формальная алгебра жизненного цикла весов
 *
 *   W = (C, Phi, b)
 *     C   — геном (Read-Only, инвариант)
 *     Phi — фенотип (Read/Write, адаптируемый)
 *     b   — гейтинг (регулятор экспрессии)
 *
 * Операции:
 *   R (Repair)        — регенерация Phi из backup
 *   A (Aging)         — усечение Phi (rank reduction)
 *   M (Mutation)      — замена Phi (адаптация)
 *   P (Proliferation) — копия Phi с маской
 * ============================================================ */

#define TCW_PHI_SIZE_BYTES  8
#define TCW_MAX_CHILDREN    4

typedef enum {
    TCW_OK = 0,
    TCW_ERR_INVALID_ARG = -1,
    TCW_ERR_CHECKSUM_MISMATCH = -2,
    TCW_ERR_DEAD = -3,
} tcw_status_t;

/* "Геном" — инвариант, только для чтения */
typedef struct {
    const uint8_t *data;
    size_t         size;
    uint32_t       crc32;
} tcw_genome_t;

/* "Фенотип" — адаптируемая часть */
typedef struct {
    uint8_t  working[TCW_PHI_SIZE_BYTES];
    uint8_t  backup[TCW_PHI_SIZE_BYTES];
    size_t   size;
    uint16_t rank;
    uint16_t max_rank;
    bool     alive;
    uint32_t checksum;
} tcw_phi_t;

/* "Гейтинг" — регулятор */
typedef struct {
    float alpha;
    float mutation_rate;
} tcw_gate_t;

/* Полный агент */
typedef struct {
    tcw_genome_t genome;
    tcw_phi_t    phi;
    tcw_gate_t   gate;
} tcw_agent_t;

/* ---------- API ---------- */
tcw_status_t tcw_init(tcw_agent_t *agent,
                      const uint8_t *genome_data, size_t genome_size,
                      uint16_t max_rank);

tcw_status_t tcw_repair(tcw_agent_t *agent);
tcw_status_t tcw_aging(tcw_agent_t *agent, uint16_t delta_rank);
tcw_status_t tcw_mutation(tcw_agent_t *agent,
                          const uint8_t *new_phi, size_t new_size);
tcw_status_t tcw_proliferation(const tcw_agent_t *parent,
                               tcw_agent_t *children, size_t n_children,
                               uint32_t seed);

/* Для тестов */
tcw_status_t tcw_damage(tcw_agent_t *agent, size_t num_bytes, uint32_t seed);
uint32_t     tcw_crc32(const uint8_t *data, size_t len);

#endif /* TCW_RUNTIME_H */