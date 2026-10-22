#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include <string.h>
#include "tcw_runtime.h"
#include "tcw_ai.h"

/* ============================================================
 * TCW-OS v0.4: Living Neural Network (TFLite Micro)
 *   Birth -> Aging -> Mutation -> Damage -> Repair -> Death
 *
 * PHI = 8 bytes = two floats (alpha, beta).
 * Inference: y_final = alpha * y_model(x) + beta
 * Metric:    MSE vs sinf(x)
 *
 * Damage XORs 8 bytes; sanitize rejects NaN/Inf/|x|>100.
 * Aging:    alpha = rank / max_rank.
 * Mutation: alpha, beta <- new env.
 * ============================================================ */

#define STACK_SIZE          4096
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
    MSG_REPAIR_COMPLETE,
    MSG_REPAIR_FAILED,
    MSG_AGING_DONE,
    MSG_MUTATION_DONE,
};

struct msg {
    enum msg_type type;
    uint32_t      iter;
    float         alpha;
    float         beta;
    float         mse;
    uint16_t      rank;
};

K_MSGQ_DEFINE(queue_to_control, sizeof(struct msg), 32, 4);
K_MSGQ_DEFINE(queue_to_repair,  sizeof(struct msg), 8,  4);
K_MUTEX_DEFINE(phi_lock);

static tcw_agent_t agent;

/* ------- calibration helpers (read/write PHI as two floats) ------- */
static void read_calib(float *a, float *b) {
    memcpy(a, agent.phi.working,     4);
    memcpy(b, agent.phi.working + 4, 4);
}
static void write_calib(float a, float b) {
    memcpy(agent.phi.working,     &a, 4);
    memcpy(agent.phi.working + 4, &b, 4);
}

/* Reject NaN, Inf, AND unreasonably large values */
static void sanitize_calib(void) {
    float a, b;
    read_calib(&a, &b);
    if (!isfinite(a) || !isfinite(b) || fabsf(a) > 100.0f || fabsf(b) > 100.0f) {
        write_calib(5.0f, -2.0f);
    }
}

/* Semantic aging: alpha = rank / max_rank, backup synced */
static tcw_status_t aging_calib(tcw_agent_t *ag, uint16_t delta) {
    if (!ag || !ag->phi.alive) return TCW_ERR_DEAD;

    uint16_t old = ag->phi.rank;
    uint16_t nw  = (old > delta) ? (uint16_t)(old - delta) : 0;
    ag->phi.rank = nw;
    if (nw == 0) ag->phi.alive = false;

    float a = (float)nw / (float)ag->phi.max_rank;
    float b;
    memcpy(&b, ag->phi.working + 4, 4);

    write_calib(a, b);
    memcpy(ag->phi.backup, ag->phi.working, TCW_PHI_SIZE_BYTES);
    ag->phi.checksum = tcw_crc32(ag->phi.working, TCW_PHI_SIZE_BYTES);
    return TCW_OK;
}

/* ================== INFERENCE (prio 5) ================== */
static void inference_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    for (int iter = 1; iter <= DEMO_ITERATIONS; iter++) {
        if (!agent.phi.alive) {
            printk("[INFERENCE] iter=%d: agent is DEAD, skipping\n", iter);
            k_sleep(K_MSEC(200));
            continue;
        }

        if (iter % DAMAGE_EVERY_N == 0) {
            k_mutex_lock(&phi_lock, K_FOREVER);
            tcw_damage(&agent, TCW_PHI_SIZE_BYTES, (uint32_t)iter);
            sanitize_calib();
            float a, b;
            read_calib(&a, &b);
            k_mutex_unlock(&phi_lock);

            printk("[INFERENCE] iter=%d -- PHI CORRUPTED (a=%.2f, b=%.2f)\n",
                   iter, (double)a, (double)b);

            struct msg r = { .type = MSG_INFERENCE_RESULT, .iter = iter };
            k_msgq_put(&queue_to_repair, &r, K_NO_WAIT);
            k_sleep(K_MSEC(200));
            continue;
        }

        float x = ((float)iter / (float)DEMO_ITERATIONS) * (2.0f * 3.14159265f);

        k_mutex_lock(&phi_lock, K_FOREVER);
        float y_model = tcw_ai_infer_raw(x);
        float alpha, beta;
        read_calib(&alpha, &beta);
        k_mutex_unlock(&phi_lock);

        float y_pred = alpha * y_model + beta;
        float y_true = sinf(x);
        float err    = (y_pred - y_true) * (y_pred - y_true);

        printk("[INFERENCE] iter=%d x=%.2f y_pred=%.3f y_true=%.3f err=%.4f a=%.2f b=%.2f rank=%u\n",
               iter, (double)x, (double)y_pred, (double)y_true,
               (double)err, (double)alpha, (double)beta, agent.phi.rank);

        struct msg m = {
            .type = MSG_INFERENCE_RESULT,
            .iter = iter,
            .alpha = alpha, .beta = beta,
            .mse = err, .rank = agent.phi.rank,
        };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        k_sleep(K_MSEC(200));
    }

    printk("\n[INFERENCE] Demo complete\n");
    k_sleep(K_SECONDS(2));
}

/* ================== REPAIR (prio 1) ================== */
static void repair_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_repair, &m, K_FOREVER);

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_repair(&agent);
        float alpha, beta;
        read_calib(&alpha, &beta);
        uint16_t rank = agent.phi.rank;
        k_mutex_unlock(&phi_lock);

        struct msg resp = {
            .type = (s == TCW_OK) ? MSG_REPAIR_COMPLETE : MSG_REPAIR_FAILED,
            .iter = m.iter, .alpha = alpha, .beta = beta, .rank = rank,
        };

        if (s == TCW_OK) {
            printk("[REPAIR   ] SUCCESS iter=%u -> a=%.2f b=%.2f rank=%u\n",
                   m.iter, (double)alpha, (double)beta, rank);
        } else {
            printk("[REPAIR   ] FAILED iter=%u (code %d)\n", m.iter, s);
        }
        k_msgq_put(&queue_to_control, &resp, K_NO_WAIT);
    }
}

/* ================== AGING (prio 2) ================== */
static void aging_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        k_sleep(K_MSEC(AGING_EVERY_MS));
        if (!agent.phi.alive) return;

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = aging_calib(&agent, AGING_STEP);
        float alpha, beta;
        read_calib(&alpha, &beta);
        uint16_t rank = agent.phi.rank;
        bool alive = agent.phi.alive;
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[AGING    ] rank->%u a=%.3f%s\n",
                   rank, (double)alpha, alive ? "" : "  *** DEAD ***");
            struct msg m = {
                .type = MSG_AGING_DONE,
                .iter = 0, .alpha = alpha, .beta = beta, .rank = rank,
            };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
            if (!alive) return;
        }
    }
}

/* ================== MUTATION (prio 4) ================== */
static void mutation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int count = 0;
    uint8_t new_phi[TCW_PHI_SIZE_BYTES];

    while (1) {
        k_sleep(K_MSEC(MUTATION_EVERY_MS));
        if (!agent.phi.alive) return;

        count++;
        float na = 1.0f + 0.05f * (float)count;
        float nb = 0.02f * (float)count;
        memcpy(new_phi,     &na, 4);
        memcpy(new_phi + 4, &nb, 4);

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_mutation(&agent, new_phi, TCW_PHI_SIZE_BYTES);
        uint16_t rank = agent.phi.rank;
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[MUTATION ] env #%d -> a=%.2f b=%.2f rank=%u\n",
                   count, (double)na, (double)nb, rank);
            struct msg m = {
                .type = MSG_MUTATION_DONE, .iter = (uint32_t)count,
                .alpha = na, .beta = nb, .rank = rank,
            };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
}

/* ================== CONTROL (prio 3) ================== */
static void control_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int damages = 0, repairs_ok = 0, repairs_failed = 0;
    int agings = 0, mutations = 0;
    float mse_sum = 0.0f;
    int mse_cnt = 0;
    float mse_min = 1e9f, mse_max = 0.0f;

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_control, &m, K_FOREVER);

        switch (m.type) {
        case MSG_INFERENCE_RESULT:
            if (m.mse > 0.0f && m.mse < 1e3f) {
                mse_sum += m.mse; mse_cnt++;
                if (m.mse < mse_min) mse_min = m.mse;
                if (m.mse > mse_max) mse_max = m.mse;
            }
            break;

        case MSG_REPAIR_COMPLETE: repairs_ok++; damages++; break;
        case MSG_REPAIR_FAILED:   repairs_failed++; break;
        case MSG_AGING_DONE:
            agings++;
            if (m.rank == 0) {
                printk("\n[CONTROL  ] *** AGENT DIED OF OLD AGE (rank=0, a=0) ***\n");
                printk("[CONTROL  ] LIFECYCLE: damages=%d, repairs_ok=%d, "
                       "repairs_failed=%d, agings=%d, mutations=%d\n",
                       damages, repairs_ok, repairs_failed, agings, mutations);
                if (mse_cnt > 0) {
                    printk("[CONTROL  ] MSE: min=%.4f max=%.4f avg=%.4f\n\n",
                           (double)mse_min, (double)mse_max,
                           (double)(mse_sum / mse_cnt));
                }
            }
            break;
        case MSG_MUTATION_DONE: mutations++; break;
        default: break;
        }
    }
}

/* ================== STACKS & TCBs ================== */
K_THREAD_STACK_DEFINE(inference_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(repair_stack,    STACK_SIZE);
K_THREAD_STACK_DEFINE(aging_stack,     STACK_SIZE);
K_THREAD_STACK_DEFINE(mutation_stack,  STACK_SIZE);
K_THREAD_STACK_DEFINE(control_stack,   STACK_SIZE);

static struct k_thread inference_tcb, repair_tcb, aging_tcb, mutation_tcb, control_tcb;

/* ================== ENTRY ================== */
int main(void)
{
    printk("\n");
    printk("==================================================\n");
    printk("  TCW-OS v0.4: Living Neural Network (TFLite Micro)\n");
    printk("  PHI = (alpha, beta), y = alpha * y_model + beta\n");
    printk("==================================================\n\n");

    tcw_ai_init();
    printk("TFLite Micro initialized (sine-wave model, INT8)\n");

    uint8_t genome[TCW_PHI_SIZE_BYTES];
    float a0 = 1.0f, b0 = 0.0f;
    memcpy(genome,     &a0, 4);
    memcpy(genome + 4, &b0, 4);

    tcw_status_t s = tcw_init(&agent, genome, TCW_PHI_SIZE_BYTES, 16);
    if (s != TCW_OK) { printk("FATAL: tcw_init failed\n"); return 0; }
    printk("Agent born: rank=%u, a=%.2f b=%.2f (perfect baseline)\n\n",
           agent.phi.rank, (double)a0, (double)b0);

    k_thread_create(&repair_tcb, repair_stack, STACK_SIZE,
                    repair_thread, NULL, NULL, NULL, PRIO_REPAIR, 0, K_NO_WAIT);
    k_thread_name_set(&repair_tcb, "tcw_repair");

    k_thread_create(&aging_tcb, aging_stack, STACK_SIZE,
                    aging_thread, NULL, NULL, NULL, PRIO_AGING, 0, K_NO_WAIT);
    k_thread_name_set(&aging_tcb, "tcw_aging");

    k_thread_create(&control_tcb, control_stack, STACK_SIZE,
                    control_thread, NULL, NULL, NULL, PRIO_CONTROL, 0, K_NO_WAIT);
    k_thread_name_set(&control_tcb, "tcw_control");

    k_thread_create(&mutation_tcb, mutation_stack, STACK_SIZE,
                    mutation_thread, NULL, NULL, NULL, PRIO_MUTATION, 0, K_NO_WAIT);
    k_thread_name_set(&mutation_tcb, "tcw_mutation");

    k_thread_create(&inference_tcb, inference_stack, STACK_SIZE,
                    inference_thread, NULL, NULL, NULL, PRIO_INFERENCE, 0, K_NO_WAIT);
    k_thread_name_set(&inference_tcb, "tcw_inference");

    return 0;
}
