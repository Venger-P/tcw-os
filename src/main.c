#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <string.h>
#include "tcw_runtime.h"

/* ============================================================
 * TCW-OS v0.3 — Full Lifecycle
 *   Birth -> Aging -> Mutation -> Damage -> Repair -> Death
 *
 * Threads (priority; lower number = higher priority):
 *   repair    (1) — restores PHI on damage; preempts everything
 *   aging     (2) — rank -= 2 every 600ms
 *   control   (3) — aggregates events, prints lifecycle summary
 *   mutation  (4) — replaces PHI every 1400ms
 *   inference (5) — damages PHI every 5th iteration
 *
 * Shared state: agent.phi protected by k_mutex (priority inheritance).
 * Why mutex, not semaphore: repair on prio 1 can wait on a mutex held
 * by inference on prio 5 -> priority inheritance temporarily boosts
 * inference, preventing unbounded priority inversion.
 * ============================================================ */

#define STACK_SIZE          2048
#define PRIO_REPAIR         1
#define PRIO_AGING          2
#define PRIO_CONTROL        3
#define PRIO_MUTATION       4
#define PRIO_INFERENCE      5

#define DEMO_ITERATIONS     30
#define DAMAGE_EVERY_N      5
#define AGING_EVERY_MS      600
#define MUTATION_EVERY_MS   1400
#define AGING_STEP          2

enum msg_type {
    MSG_INFERENCE_RESULT,
    MSG_REPAIR_NEEDED,
    MSG_REPAIR_COMPLETE,
    MSG_REPAIR_FAILED,
    MSG_AGING_DONE,
    MSG_MUTATION_DONE,
};

struct msg {
    enum msg_type type;
    uint32_t      value;
    uint32_t      checksum;
};

K_MSGQ_DEFINE(queue_to_control, sizeof(struct msg), 32, 4);
K_MSGQ_DEFINE(queue_to_repair,  sizeof(struct msg), 8,  4);

/* Mutex, NOT semaphore — priority inheritance matters here */
K_MUTEX_DEFINE(phi_lock);

static tcw_agent_t agent;
static uint8_t     fake_genome[512];

/* ================================================================
 * inference (prio 5): the "body" that lives and gets damaged
 * ================================================================ */
static void inference_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    for (int iter = 1; iter <= DEMO_ITERATIONS; iter++) {
        struct msg m = { .type = MSG_INFERENCE_RESULT, .value = (uint32_t)iter };

        if (!agent.phi.alive) {
            m.checksum = 0;
            printk("[INFERENCE] iter=%d: agent is DEAD, skipping\n", iter);
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
            k_sleep(K_MSEC(200));
            continue;
        }

        if (iter % DAMAGE_EVERY_N == 0) {
            k_mutex_lock(&phi_lock, K_FOREVER);
            tcw_damage(&agent, 100, (uint32_t)iter);
            k_mutex_unlock(&phi_lock);

            m.checksum = 0;
            printk("[INFERENCE] iter=%d -- PHI CORRUPTED (100 bytes)\n", iter);
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);

            struct msg r = { .type = MSG_REPAIR_NEEDED, .value = (uint32_t)iter };
            k_msgq_put(&queue_to_repair, &r, K_NO_WAIT);
        } else {
            k_mutex_lock(&phi_lock, K_FOREVER);
            m.checksum = agent.phi.checksum;
            uint16_t rank = agent.phi.rank;
            k_mutex_unlock(&phi_lock);

            printk("[INFERENCE] iter=%d: phi OK (crc=0x%08x, rank=%u)\n",
                   iter, m.checksum, rank);
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }

        k_sleep(K_MSEC(200));
    }

    printk("\n[INFERENCE] Demo complete\n");
    k_sleep(K_SECONDS(2));
}

/* ================================================================
 * repair (prio 1): the highest-priority healer
 * ================================================================ */
static void repair_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_repair, &m, K_FOREVER);

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_repair(&agent);
        uint32_t crc   = agent.phi.checksum;
        uint16_t rank  = agent.phi.rank;
        k_mutex_unlock(&phi_lock);

        struct msg resp = {
            .type = (s == TCW_OK) ? MSG_REPAIR_COMPLETE : MSG_REPAIR_FAILED,
            .value = m.value,
            .checksum = (s == TCW_OK) ? crc : 0,
        };

        if (s == TCW_OK) {
            printk("[REPAIR   ] SUCCESS iter=%u (rank=%u, crc=0x%08x)\n",
                   m.value, rank, crc);
        } else {
            printk("[REPAIR   ] FAILED iter=%u (code %d, agent dead)\n",
                   m.value, s);
        }
        k_msgq_put(&queue_to_control, &resp, K_NO_WAIT);
    }
}

/* ================================================================
 * aging (prio 2): rank decreases every 600ms — irreversible
 * ================================================================ */
static void aging_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        k_sleep(K_MSEC(AGING_EVERY_MS));

        if (!agent.phi.alive) {
            printk("[AGING    ] agent already dead, stopping\n");
            return;
        }

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_aging(&agent, AGING_STEP);
        uint16_t new_rank = agent.phi.rank;
        bool alive = agent.phi.alive;
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[AGING    ] rank -> %u%s\n",
                   new_rank, alive ? "" : "  *** DEAD ***");

            struct msg m = {
                .type = MSG_AGING_DONE,
                .value = new_rank,
                .checksum = alive ? 1 : 0,
            };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);

            if (!alive) return;
        }
    }
}

/* ================================================================
 * mutation (prio 4): replaces PHI every 1400ms
 * ================================================================ */
static void mutation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    uint8_t new_phi[TCW_PHI_SIZE_BYTES];
    int mutation_counter = 0;

    while (1) {
        k_sleep(K_MSEC(MUTATION_EVERY_MS));

        if (!agent.phi.alive) {
            printk("[MUTATION ] agent already dead, stopping\n");
            return;
        }

        mutation_counter++;
        memset(new_phi, 0x10 * mutation_counter, TCW_PHI_SIZE_BYTES);

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_mutation(&agent, new_phi, TCW_PHI_SIZE_BYTES);
        uint16_t rank = agent.phi.rank;
        uint32_t crc  = agent.phi.checksum;
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[MUTATION ] PHI replaced (env #%d, rank=%u, crc=0x%08x)\n",
                   mutation_counter, rank, crc);
            struct msg m = {
                .type = MSG_MUTATION_DONE,
                .value = (uint32_t)mutation_counter,
                .checksum = crc,
            };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
}

/* ================================================================
 * control (prio 3): observes everything, prints final summary
 * ================================================================ */
static void control_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int damages = 0, repairs_ok = 0, repairs_failed = 0;
    int agings = 0, mutations = 0;

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_control, &m, K_FOREVER);

        switch (m.type) {
        case MSG_INFERENCE_RESULT:
            if (m.checksum == 0) damages++;
            break;

        case MSG_REPAIR_COMPLETE:
            repairs_ok++;
            break;

        case MSG_REPAIR_FAILED:
            repairs_failed++;
            break;

        case MSG_AGING_DONE:
            agings++;
            if (m.checksum == 0) {
                printk("[CONTROL  ] *** AGENT DIED OF OLD AGE (rank=0) ***\n");
                printk("[CONTROL  ] LIFECYCLE: damages=%d, repairs_ok=%d, "
                       "repairs_failed=%d, agings=%d, mutations=%d\n\n",
                       damages, repairs_ok, repairs_failed, agings, mutations);
            }
            break;

        case MSG_MUTATION_DONE:
            mutations++;
            break;

        default:
            break;
        }
    }
}

/* ================================================================
 * Stacks & TCBs
 * ================================================================ */
K_THREAD_STACK_DEFINE(inference_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(repair_stack,    STACK_SIZE);
K_THREAD_STACK_DEFINE(aging_stack,     STACK_SIZE);
K_THREAD_STACK_DEFINE(mutation_stack,  STACK_SIZE);
K_THREAD_STACK_DEFINE(control_stack,   STACK_SIZE);

static struct k_thread inference_tcb;
static struct k_thread repair_tcb;
static struct k_thread aging_tcb;
static struct k_thread mutation_tcb;
static struct k_thread control_tcb;

/* ================================================================
 * Entry point
 * ================================================================ */
int main(void)
{
    printk("\n");
    printk("==================================================\n");
    printk("  TCW-OS v0.3: Full Lifecycle\n");
    printk("  Birth -> Damage -> Repair -> Aging -> Mutation -> Death\n");
    printk("==================================================\n\n");

    for (size_t i = 0; i < sizeof(fake_genome); i++)
        fake_genome[i] = (uint8_t)(i * 7 + 3);

    tcw_status_t s = tcw_init(&agent, fake_genome, sizeof(fake_genome), 16);
    if (s != TCW_OK) {
        printk("FATAL: tcw_init failed\n");
        return 0;
    }
    printk("Agent born: rank=%u, crc=0x%08x\n\n",
           agent.phi.rank, agent.phi.checksum);

    k_thread_create(&repair_tcb, repair_stack, STACK_SIZE,
                    repair_thread, NULL, NULL, NULL,
                    PRIO_REPAIR, 0, K_NO_WAIT);
    k_thread_name_set(&repair_tcb, "tcw_repair");

    k_thread_create(&aging_tcb, aging_stack, STACK_SIZE,
                    aging_thread, NULL, NULL, NULL,
                    PRIO_AGING, 0, K_NO_WAIT);
    k_thread_name_set(&aging_tcb, "tcw_aging");

    k_thread_create(&control_tcb, control_stack, STACK_SIZE,
                    control_thread, NULL, NULL, NULL,
                    PRIO_CONTROL, 0, K_NO_WAIT);
    k_thread_name_set(&control_tcb, "tcw_control");

    k_thread_create(&mutation_tcb, mutation_stack, STACK_SIZE,
                    mutation_thread, NULL, NULL, NULL,
                    PRIO_MUTATION, 0, K_NO_WAIT);
    k_thread_name_set(&mutation_tcb, "tcw_mutation");

    k_thread_create(&inference_tcb, inference_stack, STACK_SIZE,
                    inference_thread, NULL, NULL, NULL,
                    PRIO_INFERENCE, 0, K_NO_WAIT);
    k_thread_name_set(&inference_tcb, "tcw_inference");

    return 0;
}
