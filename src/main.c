#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include <string.h>
#include "tcw_runtime.h"
#include "tcw_ai.h"

/* ============================================================
 * TCW-OS v0.6 — Full TCW-Life with learnable PHI
 *
 * PHI = residual MLP (1->8->1, tanh, 25 weights = 100 bytes).
 * y_pred(x) = y_model(x) + PHI(x / 2pi)
 *
 * 7 threads (priority; lower = higher):
 *   repair         (1) : restore PHI from backup
 *   etopo          (2) : SGD train PHI on D_train (metabolism)
 *   aging          (3) : zero smallest-|w| weights, rank-=2
 *   control        (4) : aggregates lifecycle + prints homeostasis
 *   mutation       (5) : random reinit PHI (new environment)
 *   inference      (6) : y_pred, MSE, damage every 5th iter
 *   proliferation  (7) : 3 children with 25% dropout
 *
 * Homeostasis metric: held-out MSE (10 clean points, unseen by etopo)
 * after each operation. Should stay low across R/A/M/P.
 * E_topo is the ONE operation that can drift it (by design — learning).
 * ============================================================ */

#define STACK_SIZE               4096
#define PRIO_REPAIR              1
#define PRIO_ETOPO               2
#define PRIO_AGING               3
#define PRIO_CONTROL             4
#define PRIO_MUTATION            5
#define PRIO_INFERENCE           6
#define PRIO_PROLIFERATION       7

#define DEMO_ITERATIONS          30
#define DAMAGE_EVERY_N           5
#define AGING_EVERY_MS           600
#define MUTATION_EVERY_MS        1400
#define ETOPO_EVERY_MS           2500
#define PROLIFERATION_EVERY_MS   3000
#define AGING_STEP               2
#define N_CHILDREN               3
#define PROLIF_DROPOUT_P         0.25f

#define ETOPO_EPOCHS             30
#define ETOPO_LR                 0.05f
#define ETOPO_BATCH              20

#define TWO_PI                   6.28318530718f

enum msg_type {
    MSG_INFERENCE_RESULT,
    MSG_REPAIR_COMPLETE,
    MSG_AGING_DONE,
    MSG_MUTATION_DONE,
    MSG_PROGENY_DONE,
    MSG_ETOPO_DONE,
    MSG_HELDOUT,
};

struct msg {
    enum msg_type type;
    uint32_t      iter;
    float         mse;
    float         heldout_mse;
    uint16_t      rank;
};

K_MSGQ_DEFINE(queue_to_control, sizeof(struct msg), 64, 4);
K_MSGQ_DEFINE(queue_to_repair,  sizeof(struct msg), 8,  4);
K_MUTEX_DEFINE(phi_lock);

static tcw_agent_t agent;

/* === Homeostasis: held-out MSE (10 clean points, always) === */
static float compute_heldout_mse(void) {
    float sum = 0.0f;
    for (int k = 0; k < 10; k++) {
        float x  = ((float)k + 0.5f) / 10.0f * TWO_PI;
        float xn = x / TWO_PI;
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)agent.phi.working, xn);
        float yp = ym + yc;
        float yt = sinf(x);
        float e  = yp - yt;
        sum += e * e;
    }
    return sum / 10.0f;
}

static void sanitize_phi(void) {
    float *w = (float *)agent.phi.working;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
        if (!isfinite(w[j]) || fabsf(w[j]) > 100.0f) {
            w[j] = 0.5f;
        }
    }
}

/* ================== INFERENCE (prio 6) ================== */
static void inference_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    for (int iter = 1; iter <= DEMO_ITERATIONS; iter++) {
        if (!agent.phi.alive) {
            printk("[INFERENCE] iter=%d: DEAD, skipping\n", iter);
            k_sleep(K_MSEC(200));
            continue;
        }

        if (iter % DAMAGE_EVERY_N == 0) {
            k_mutex_lock(&phi_lock, K_FOREVER);
            tcw_damage(&agent, TCW_PHI_SIZE_BYTES, (uint32_t)iter);
            sanitize_phi();
            k_mutex_unlock(&phi_lock);
            printk("[INFERENCE] iter=%d -- PHI CORRUPTED\n", iter);
            struct msg r = { .type = MSG_INFERENCE_RESULT, .iter = iter };
            k_msgq_put(&queue_to_repair, &r, K_NO_WAIT);
            k_sleep(K_MSEC(200));
            continue;
        }

        float x  = ((float)iter / (float)DEMO_ITERATIONS) * TWO_PI;
        float xn = x / TWO_PI;

        k_mutex_lock(&phi_lock, K_FOREVER);
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)agent.phi.working, xn);
        uint16_t rank = agent.phi.rank;
        k_mutex_unlock(&phi_lock);

        float yp = ym + yc;
        float yt = sinf(x);
        float err = (yp - yt) * (yp - yt);

        printk("[INFERENCE] iter=%d x=%.2f y=%.3f y_true=%.3f err=%.5f rank=%u\n",
               iter, (double)x, (double)yp, (double)yt, (double)err, rank);

        struct msg m = { .type = MSG_INFERENCE_RESULT, .iter = iter,
                         .mse = err, .rank = rank };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        k_sleep(K_MSEC(200));
    }
    printk("\n[INFERENCE] Demo complete\n");
    k_sleep(K_SECONDS(2));
}

/* ================== ETOPO (prio 2): metabolism ================== */
static void etopo_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        k_sleep(K_MSEC(ETOPO_EVERY_MS));
        if (!agent.phi.alive) {
            printk("[ETOPO    ] parent dead, stopping\n");
            return;
        }

        /* D_train: 20 points of sine + tiny high-freq noise.
         * Target for PHI = y_true(x) - y_model(x). */
        float x_train[ETOPO_BATCH], y_train[ETOPO_BATCH];
        for (int k = 0; k < ETOPO_BATCH; k++) {
            float x = (float)k / (float)ETOPO_BATCH * TWO_PI;
            float y_noisy = sinf(x) + 0.05f * sinf(5.0f * x);
            x_train[k] = x / TWO_PI;
            y_train[k] = y_noisy - tcw_ai_infer_raw(x);
        }

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_etopo(&agent, x_train, y_train,
                                    ETOPO_BATCH, ETOPO_LR, ETOPO_EPOCHS);
        float ho = compute_heldout_mse();
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[ETOPO    ] trained %d epochs on %d noisy samples, held-out MSE=%.5f\n",
                   ETOPO_EPOCHS, ETOPO_BATCH, (double)ho);
            struct msg m = { .type = MSG_ETOPO_DONE, .iter = 0,
                             .heldout_mse = ho, .rank = agent.phi.rank };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
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
        float ho = compute_heldout_mse();
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[REPAIR   ] SUCCESS iter=%u, held-out MSE=%.5f\n",
                   m.iter, (double)ho);
        } else {
            printk("[REPAIR   ] FAILED iter=%u (code %d)\n", m.iter, s);
        }
        struct msg r = { .type = MSG_REPAIR_COMPLETE, .iter = m.iter,
                         .heldout_mse = ho, .rank = agent.phi.rank };
        k_msgq_put(&queue_to_control, &r, K_NO_WAIT);
    }
}

/* ================== AGING (prio 3) ================== */
static void aging_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        k_sleep(K_MSEC(AGING_EVERY_MS));
        if (!agent.phi.alive) return;

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_aging(&agent, AGING_STEP);
        float ho = compute_heldout_mse();
        uint16_t rank = agent.phi.rank;
        bool alive = agent.phi.alive;
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[AGING    ] rank->%u held-out MSE=%.5f%s\n",
                   rank, (double)ho, alive ? "" : "  *** DEAD ***");
            struct msg m = { .type = MSG_AGING_DONE, .rank = rank,
                             .heldout_mse = ho };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
            if (!alive) return;
        }
    }
}

/* ================== MUTATION (prio 5) ================== */
static void mutation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int count = 0;
    while (1) {
        k_sleep(K_MSEC(MUTATION_EVERY_MS));
        if (!agent.phi.alive) return;

        count++;
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_mutation(&agent, (uint32_t)(1000 + count));
        float ho = compute_heldout_mse();
        k_mutex_unlock(&phi_lock);

        if (s == TCW_OK) {
            printk("[MUTATION ] env #%d random reinit, held-out MSE=%.5f\n",
                   count, (double)ho);
            struct msg m = { .type = MSG_MUTATION_DONE, .iter = (uint32_t)count,
                             .heldout_mse = ho, .rank = agent.phi.rank };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
}

/* ================== PROLIFERATION (prio 7) ================== */
static void proliferation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    static tcw_agent_t children[N_CHILDREN];
    int generation = 0;

    while (1) {
        k_sleep(K_MSEC(PROLIFERATION_EVERY_MS));
        if (!agent.phi.alive) {
            printk("[PROLIF   ] parent dead, stopping\n");
            return;
        }

        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_agent_t snap = agent;
        k_mutex_unlock(&phi_lock);

        tcw_status_t s = tcw_proliferation(&snap, children, N_CHILDREN,
                                            (uint32_t)(generation + 1),
                                            PROLIF_DROPOUT_P);
        if (s != TCW_OK) continue;
        generation++;

        printk("[PROLIF   ] gen #%d: spawned %d children (dropout %.0f%%)\n",
               generation, N_CHILDREN, (double)(PROLIF_DROPOUT_P * 100.0f));

        for (int i = 0; i < N_CHILDREN; i++) {
            float mse = 0.0f;
            for (int k = 0; k < 10; k++) {
                float x  = ((float)k + 0.5f) / 10.0f * TWO_PI;
                float xn = x / TWO_PI;
                float ym = tcw_ai_infer_raw(x);
                float yc = tcw_mlp_forward((const tcw_mlp_t *)children[i].phi.working, xn);
                float yp = ym + yc;
                float e  = yp - sinf(x);
                mse += e * e;
            }
            mse /= 10.0f;
            printk("[PROLIF   ]   child#%d held-out MSE=%.5f\n",
                   i + 1, (double)mse);
        }

        struct msg m = { .type = MSG_PROGENY_DONE, .iter = (uint32_t)generation };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }
}

/* ================== CONTROL (prio 4) ================== */
static void control_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int damages = 0, repairs = 0, agings = 0, mutations = 0,
        proliferations = 0, etopos = 0;
    float mse_sum = 0.0f, mse_min = 1e9f, mse_max = 0.0f;
    int mse_cnt = 0;
    float ho_sum = 0.0f, ho_min = 1e9f, ho_max = 0.0f;
    int ho_cnt = 0;

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_control, &m, K_FOREVER);

        if (m.type == MSG_INFERENCE_RESULT) {
            if (m.mse > 0.0f && m.mse < 1e3f) {
                mse_sum += m.mse; mse_cnt++;
                if (m.mse < mse_min) mse_min = m.mse;
                if (m.mse > mse_max) mse_max = m.mse;
            }
            damages++;
            continue;
        }

        if (m.heldout_mse > 0.0f) {
            ho_sum += m.heldout_mse; ho_cnt++;
            if (m.heldout_mse < ho_min) ho_min = m.heldout_mse;
            if (m.heldout_mse > ho_max) ho_max = m.heldout_mse;
        }

        switch (m.type) {
        case MSG_REPAIR_COMPLETE: repairs++; break;
        case MSG_AGING_DONE:
            agings++;
            if (m.rank == 0) {
                printk("\n[CONTROL  ] *** AGENT DIED OF OLD AGE (rank=0) ***\n");
                printk("[CONTROL  ] LIFECYCLE: damages=%d, repairs=%d, "
                       "agings=%d, mutations=%d, proliferations=%d, etopos=%d\n",
                       damages, repairs, agings, mutations, proliferations, etopos);
                if (mse_cnt)
                    printk("[CONTROL  ] infer MSE:   min=%.5f max=%.5f avg=%.5f\n",
                           (double)mse_min, (double)mse_max, (double)(mse_sum/mse_cnt));
                if (ho_cnt)
                    printk("[CONTROL  ] held-out MSE: min=%.5f max=%.5f avg=%.5f\n",
                           (double)ho_min, (double)ho_max, (double)(ho_sum/ho_cnt));
            }
            break;
        case MSG_MUTATION_DONE: mutations++; break;
        case MSG_PROGENY_DONE:  proliferations++; break;
        case MSG_ETOPO_DONE:    etopos++; break;
        default: break;
        }
    }
}

/* ================== STACKS & TCBs ================== */
K_THREAD_STACK_DEFINE(inference_stack,     STACK_SIZE);
K_THREAD_STACK_DEFINE(repair_stack,        STACK_SIZE);
K_THREAD_STACK_DEFINE(aging_stack,         STACK_SIZE);
K_THREAD_STACK_DEFINE(etopo_stack,         STACK_SIZE);
K_THREAD_STACK_DEFINE(mutation_stack,      STACK_SIZE);
K_THREAD_STACK_DEFINE(proliferation_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(control_stack,       STACK_SIZE);

static struct k_thread inference_tcb, repair_tcb, aging_tcb, etopo_tcb,
                      mutation_tcb, proliferation_tcb, control_tcb;

/* ================== ENTRY ================== */
int main(void)
{
    printk("\n");
    printk("==================================================\n");
    printk("  TCW-OS v0.6: Full TCW-Life (learnable PHI)\n");
    printk("  PHI = residual MLP (1->8->1, tanh, 25 weights)\n");
    printk("  E_topo + R + A + M + P on real TFLite model\n");
    printk("==================================================\n\n");

    tcw_ai_init();
    printk("TFLite Micro: sine-wave model (INT8), frozen as C (genome)\n");

    /* Genome: initial PHI = zeros (MLP outputs 0, y_pred = y_model) */
    uint8_t genome[TCW_PHI_SIZE_BYTES];
    memset(genome, 0, TCW_PHI_SIZE_BYTES);

    tcw_status_t s = tcw_init(&agent, genome, TCW_PHI_SIZE_BYTES, 16);
    if (s != TCW_OK) { printk("FATAL: tcw_init failed\n"); return 0; }
    printk("Agent born: rank=%u, PHI=0 (y_pred = y_model)\n", agent.phi.rank);

    float ho = compute_heldout_mse();
    printk("Baseline held-out MSE = %.5f\n\n", (double)ho);

    k_thread_create(&repair_tcb, repair_stack, STACK_SIZE,
                    repair_thread, NULL, NULL, NULL, PRIO_REPAIR, 0, K_NO_WAIT);
    k_thread_name_set(&repair_tcb, "tcw_repair");

    k_thread_create(&etopo_tcb, etopo_stack, STACK_SIZE,
                    etopo_thread, NULL, NULL, NULL, PRIO_ETOPO, 0, K_NO_WAIT);
    k_thread_name_set(&etopo_tcb, "tcw_etopo");

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

    k_thread_create(&proliferation_tcb, proliferation_stack, STACK_SIZE,
                    proliferation_thread, NULL, NULL, NULL, PRIO_PROLIFERATION, 0, K_NO_WAIT);
    k_thread_name_set(&proliferation_tcb, "tcw_prolif");

    return 0;
}
