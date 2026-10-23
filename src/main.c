#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include <string.h>
#include "tcw_runtime.h"
#include "tcw_ai.h"

/* ============================================================
 * TCW-OS v0.9.3 — Single-anchor EWC + larger residual + burst fix
 *
 * 3 fixes over v0.9:
 *   1. Residual target: 0.05*sin(5x) -> 0.20*sin(5x)
 *      PHI learns 4x larger weights -> dropout visible
 *   2. Lambda sweep: 4 trials (no-EWC, 50, 200, 1000) in one run
 *      Prints full tradeoff table
 *   3. Composition burst: 50ms interval (was 200ms), aging delayed
 *      500ms after ontogeny -> all 3 bursts land before first aging
 * ============================================================ */

#define STACK_SIZE               4096
#define PRIO_REPAIR              1
#define PRIO_ETOPO               2
#define PRIO_AGING               3
#define PRIO_CONTROL             4
#define PRIO_MUTATION            5
#define PRIO_INFERENCE           6
#define PRIO_PROLIFERATION       7
#define PRIO_COMPOSITION         8
#define PRIO_HOMEOSTASIS         9

#define DEMO_ITERATIONS          40
#define DAMAGE_EVERY_N           10
#define AGING_EVERY_MS           600
#define AGING_DELAY_AFTER_ONTO   500
#define MUTATION_EVERY_MS        1400
#define PROLIFERATION_EVERY_MS   3000
#define COMPOSITION_BURST_MS     50
#define COMPOSITION_PERIODIC_MS  5000
#define COMPOSITION_BURST_COUNT  3
#define NORMALIZE_EVERY_MS       2200
#define HOMEOSTASIS_EVERY_MS     1500
#define AGING_STEP               2
#define N_CHILDREN               3
#define PROLIF_DROPOUT_P         0.10f

#define ETOPO_EPOCHS             40
#define ETOPO_LR                 0.05f
#define ETOPO_BATCH              20
#define REPLAY_PER_TASK          10
#define REPLAY_MAX               30

#define TWO_PI                   6.28318530718f
#define T1_XMIN                  0.0f
#define T1_XMAX                  (TWO_PI / 4.0f)
#define T2_XMIN                  (TWO_PI / 4.0f)
#define T2_XMAX                  (TWO_PI / 2.0f)
#define T3_XMIN                  (TWO_PI / 2.0f)
#define T3_XMAX                  (3.0f * TWO_PI / 4.0f)

enum msg_type {
    MSG_INFERENCE_RESULT,
    MSG_REPAIR_COMPLETE,
    MSG_AGING_DONE,
    MSG_MUTATION_DONE,
    MSG_PROGENY_DONE,
    MSG_ETOPO_DONE,
    MSG_COMPOSITION_DONE,
    MSG_NORMALIZATION_DONE,
    MSG_ONTogeny_DONE,
    MSG_HOMEOSTASIS_REPORT,
};

struct msg {
    enum msg_type type;
    uint32_t      iter;
    float         mse;
    float         heldout_mse;
    float         gate_alpha;
    uint16_t      rank;
    uint8_t       corrupted;
};

K_MSGQ_DEFINE(queue_to_control, sizeof(struct msg), 64, 4);
K_MSGQ_DEFINE(queue_to_repair,  sizeof(struct msg), 8,  4);
K_MUTEX_DEFINE(phi_lock);

static tcw_agent_t agent;
static volatile bool ontogeny_done = false;

static float y_model_ref[10];
static bool homeo_ref_ready = false;

typedef struct { float x; float y_target; } sample_t;
static sample_t replay_buf[REPLAY_MAX];
static int replay_n = 0;

static void capture_homeostasis_reference(void) {
    for (int i = 0; i < 10; i++) {
        float x = (float)i / 10.0f * TWO_PI;
        y_model_ref[i] = tcw_ai_infer_raw(x);
    }
    homeo_ref_ready = true;
}

static float verify_homeostasis(void) {
    if (!homeo_ref_ready) return 0.0f;
    float max_delta = 0.0f;
    for (int i = 0; i < 10; i++) {
        float x = (float)i / 10.0f * TWO_PI;
        float y = tcw_ai_infer_raw(x);
        float d = fabsf(y - y_model_ref[i]);
        if (d > max_delta) max_delta = d;
    }
    return max_delta;
}

/* v0.9.1: residual 4x larger (0.20 vs 0.05) */
static void generate_task_data(float xmin, float xmax, int n,
                                float *x_out, float *y_out) {
    for (int i = 0; i < n; i++) {
        float x = xmin + (xmax - xmin) * (float)i / (float)(n - 1);
        x_out[i] = x / TWO_PI;
        y_out[i] = (sinf(x) + 0.20f * sinf(5.0f * x)) - tcw_ai_infer_raw(x);
    }
}

static float eval_task(float xmin, float xmax, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float x = xmin + (xmax - xmin) * (float)i / (float)(n - 1);
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)agent.phi.working, x / TWO_PI);
        float yp = ym + agent.gate.alpha * yc;
        float e  = yp - sinf(x);
        sum += e * e;
    }
    return sum / (float)n;
}

static float compute_heldout_mse(void) {
    float sum = 0.0f;
    for (int k = 0; k < 10; k++) {
        float x  = ((float)k + 0.5f) / 10.0f * TWO_PI;
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)agent.phi.working, x / TWO_PI);
        float yp = ym + agent.gate.alpha * yc;
        float e  = yp - sinf(x);
        sum += e * e;
    }
    return sum / 10.0f;
}

static float eval_agent_mse(const tcw_agent_t *ag) {
    float sum = 0.0f;
    for (int k = 0; k < 10; k++) {
        float x  = ((float)k + 0.5f) / 10.0f * TWO_PI;
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)ag->phi.working, x / TWO_PI);
        float yp = ym + ag->gate.alpha * yc;
        float e  = yp - sinf(x);
        sum += e * e;
    }
    return sum / 10.0f;
}

static void sanitize_phi(void) {
    float *w = (float *)agent.phi.working;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
        if (!isfinite(w[j]) || fabsf(w[j]) > 100.0f) w[j] = 0.5f;
    }
}

static void reset_agent_for_ablation(void) {
    /* v0.9.2 fix: random init, not zero.
     * Zero-init causes dead gradient (w2=0 blocks dz). Every trial
     * must start with nonzero weights, or Fisher is 1/25 and EWC
     * cannot work. Different seed per call for independence. */
    static uint32_t seed_ctr = 0;
    seed_ctr++;
    float *w = (float *)agent.phi.working;
    uint32_t rng = 0xCAFEBABE ^ (seed_ctr * 0x9E3779B1u);
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        int32_t r = (int32_t)rng;
        w[j] = ((float)r / 2147483648.0f) * 0.1f;
    }
    memcpy(agent.phi.backup, agent.phi.working, TCW_PHI_SIZE_BYTES);
    agent.phi.rank = agent.phi.max_rank;
    agent.phi.alive = true;
    agent.phi.checksum = tcw_crc32(agent.phi.working, TCW_PHI_SIZE_BYTES);
    agent.gate.alpha = 1.0f;
    replay_n = 0;
}

static void train_no_ewc(float xmin, float xmax) {
    float x_new[ETOPO_BATCH], y_new[ETOPO_BATCH];
    generate_task_data(xmin, xmax, ETOPO_BATCH, x_new, y_new);
    float x_batch[ETOPO_BATCH + REPLAY_MAX];
    float y_batch[ETOPO_BATCH + REPLAY_MAX];
    int nb = 0;
    for (int i = 0; i < ETOPO_BATCH; i++) {
        x_batch[nb] = x_new[i]; y_batch[nb] = y_new[i]; nb++;
    }
    for (int i = 0; i < replay_n; i++) {
        x_batch[nb] = replay_buf[i].x; y_batch[nb] = replay_buf[i].y_target; nb++;
    }
    k_mutex_lock(&phi_lock, K_FOREVER);
    tcw_etopo(&agent, x_batch, y_batch, nb, ETOPO_LR, ETOPO_EPOCHS);
    k_mutex_unlock(&phi_lock);
    for (int i = 0; i < REPLAY_PER_TASK && replay_n < REPLAY_MAX; i++) {
        int idx = i * 2;
        if (idx >= ETOPO_BATCH) idx = ETOPO_BATCH - 1;
        replay_buf[replay_n].x = x_new[idx];
        replay_buf[replay_n].y_target = y_new[idx];
        replay_n++;
    }
}

static void train_with_ewc(float xmin, float xmax,
                            const float *f1, const float *t1,
                            const float *f2, const float *t2,
                            float lambda) {
    float x_new[ETOPO_BATCH], y_new[ETOPO_BATCH];
    generate_task_data(xmin, xmax, ETOPO_BATCH, x_new, y_new);
    float x_batch[ETOPO_BATCH + REPLAY_MAX];
    float y_batch[ETOPO_BATCH + REPLAY_MAX];
    int nb = 0;
    for (int i = 0; i < ETOPO_BATCH; i++) {
        x_batch[nb] = x_new[i]; y_batch[nb] = y_new[i]; nb++;
    }
    for (int i = 0; i < replay_n; i++) {
        x_batch[nb] = replay_buf[i].x; y_batch[nb] = replay_buf[i].y_target; nb++;
    }
    k_mutex_lock(&phi_lock, K_FOREVER);
    tcw_mlp_t *m = (tcw_mlp_t *)agent.phi.working;
    for (int ep = 0; ep < ETOPO_EPOCHS; ep++) {
        for (int i = 0; i < nb; i++) {
            float x  = x_batch[i];
            float yt = y_batch[i];
            float z[TCW_MLP_HIDDEN];
            float y = m->b2;
            for (int j = 0; j < TCW_MLP_HIDDEN; j++) {
                z[j] = tanhf(m->w1[j] * x + m->b1[j]);
                y += m->w2[j] * z[j];
            }
            float dy = y - yt;
            float dw1[TCW_MLP_HIDDEN], db1[TCW_MLP_HIDDEN], dw2[TCW_MLP_HIDDEN];
            for (int j = 0; j < TCW_MLP_HIDDEN; j++) {
                float dz = dy * m->w2[j] * (1.0f - z[j] * z[j]);
                dw1[j] = dz * x;
                db1[j] = dz;
                dw2[j] = dy * z[j];
            }
            int idx = 0;
            for (int j = 0; j < TCW_MLP_HIDDEN; j++, idx++) {
                float ewc = 0.0f;
                if (f1) ewc += lambda * f1[idx] * (m->w1[j] - t1[idx]);
                if (f2) ewc += lambda * f2[idx] * (m->w1[j] - t2[idx]);
                m->w1[j] -= ETOPO_LR * (dw1[j] + ewc);
            }
            for (int j = 0; j < TCW_MLP_HIDDEN; j++, idx++) {
                float ewc = 0.0f;
                if (f1) ewc += lambda * f1[idx] * (m->b1[j] - t1[idx]);
                if (f2) ewc += lambda * f2[idx] * (m->b1[j] - t2[idx]);
                m->b1[j] -= ETOPO_LR * (db1[j] + ewc);
            }
            for (int j = 0; j < TCW_MLP_HIDDEN; j++, idx++) {
                float ewc = 0.0f;
                if (f1) ewc += lambda * f1[idx] * (m->w2[j] - t1[idx]);
                if (f2) ewc += lambda * f2[idx] * (m->w2[j] - t2[idx]);
                m->w2[j] -= ETOPO_LR * (dw2[j] + ewc);
            }
            {
                float ewc = 0.0f;
                if (f1) ewc += lambda * f1[idx] * (m->b2 - t1[idx]);
                if (f2) ewc += lambda * f2[idx] * (m->b2 - t2[idx]);
                m->b2 -= ETOPO_LR * (dy + ewc);
            }
        }
    }
    memcpy(agent.phi.backup, agent.phi.working, TCW_PHI_SIZE_BYTES);
    agent.phi.checksum = tcw_crc32(agent.phi.working, TCW_PHI_SIZE_BYTES);
    k_mutex_unlock(&phi_lock);
    for (int i = 0; i < REPLAY_PER_TASK && replay_n < REPLAY_MAX; i++) {
        int idx = i * 2;
        if (idx >= ETOPO_BATCH) idx = ETOPO_BATCH - 1;
        replay_buf[replay_n].x = x_new[idx];
        replay_buf[replay_n].y_target = y_new[idx];
        replay_n++;
    }
}

static void snapshot_fisher_theta(float xmin, float xmax,
                                    float *fisher, float *theta) {
    float xs[10], ys[10];
    generate_task_data(xmin, xmax, 10, xs, ys);
    k_mutex_lock(&phi_lock, K_FOREVER);
    tcw_compute_fisher(&agent, xs, ys, 10, fisher);
    memcpy(theta, agent.phi.working, TCW_PHI_SIZE_BYTES);
    k_mutex_unlock(&phi_lock);

    /* v0.9.2: direct Fisher diagnostic */
    float fmax = 0.0f, fsum = 0.0f;
    int fnz = 0;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) {
        if (fisher[j] > fmax) fmax = fisher[j];
        fsum += fisher[j];
        if (fisher[j] > 1e-9f) fnz++;
    }
    printk("[FISHER   ] x∈[%.2f,%.2f]: max=%.6e mean=%.6e nonzero=%d/25\n",
           (double)xmin, (double)xmax, (double)fmax,
           (double)(fsum / TCW_MLP_N_WEIGHTS), fnz);
}

/* ============================================================
 * v0.9.1: single trial — T1 -> T2 -> T3, with or without EWC
 * Returns forgetting percentage on T1 (after T3 vs after T1).
 * Caller must reset_agent_for_ablation() between trials.
 * ============================================================ */
static float run_ontogeny_trial(const char *label, float lambda, bool use_ewc) {
    float fisher_T1[TCW_MLP_N_WEIGHTS];
    float theta_T1 [TCW_MLP_N_WEIGHTS];
    float fisher_T2[TCW_MLP_N_WEIGHTS];
    float theta_T2 [TCW_MLP_N_WEIGHTS];

    /* T1 (no EWC — no prior anchors) */
    train_no_ewc(T1_XMIN, T1_XMAX);
    float m1_after_T1 = eval_task(T1_XMIN, T1_XMAX, 10);
    if (use_ewc) snapshot_fisher_theta(T1_XMIN, T1_XMAX, fisher_T1, theta_T1);

    /* T2 */
    if (use_ewc) train_with_ewc(T2_XMIN, T2_XMAX, fisher_T1, theta_T1, NULL, NULL, lambda);
    else         train_no_ewc(T2_XMIN, T2_XMAX);
    float m1_after_T2 = eval_task(T1_XMIN, T1_XMAX, 10);
    if (use_ewc) snapshot_fisher_theta(T2_XMIN, T2_XMAX, fisher_T2, theta_T2);

    /* T3 — v0.9.3: SINGLE anchor (θ_T1 only), no multi-anchor tension */
    if (use_ewc) train_with_ewc(T3_XMIN, T3_XMAX, fisher_T1, theta_T1,
                                 NULL, NULL, lambda);
    else         train_no_ewc(T3_XMIN, T3_XMAX);
    float m1_after_T3 = eval_task(T1_XMIN, T1_XMAX, 10);

    float forgetting = 100.0f * (m1_after_T3 - m1_after_T1) / m1_after_T1;

    printk("[ONTOGENY] %-8s λ=%-6.0f  T1@T1=%.5f T1@T2=%.5f T1@T3=%.5f forget=%+.1f%%\n",
           label, (double)lambda, (double)m1_after_T1, (double)m1_after_T2,
           (double)m1_after_T3, (double)forgetting);

    return forgetting;
}

/* ================== ONTOGENY (prio 2) — v0.9.1 λ sweep ================== */
static void ontogeny_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    printk("\n[ONTOGENY] ================================================\n");
    printk("[ONTOGENY] LAMBDA SWEEP: 4 trials on T1->T2->T3\n");
    printk("[ONTOGENY] ================================================\n");

    /* Trial 0: no EWC */
    reset_agent_for_ablation();
    float f0 = run_ontogeny_trial("NO-EWC", 0.0f, false);
    k_sleep(K_MSEC(100));

    /* Trial 1: λ=50 */
    reset_agent_for_ablation();
    float f50 = run_ontogeny_trial("EWC", 50.0f, true);
    k_sleep(K_MSEC(100));

    /* Trial 2: λ=200 */
    reset_agent_for_ablation();
    float f200 = run_ontogeny_trial("EWC", 200.0f, true);
    k_sleep(K_MSEC(100));

    /* Trial 3: λ=1000 */
    reset_agent_for_ablation();
    float f1000 = run_ontogeny_trial("EWC", 1000.0f, true);
    k_sleep(K_MSEC(100));

    /* Sweep summary */
    printk("\n[ONTOGENY] ============= LAMBDA SWEEP RESULT =============\n");
    printk("[ONTOGENY]   %-8s %-10s %s\n", "lambda", "forget%", "vs no-EWC");
    printk("[ONTOGENY]   %-8s %+.1f%%     baseline\n", "0", (double)f0);
    printk("[ONTOGENY]   %-8.0f %+.1f%%     %+.1f pp\n",
           50.0, (double)f50, (double)(f50 - f0));
    printk("[ONTOGENY]   %-8.0f %+.1f%%     %+.1f pp  ← sweet spot?\n",
           200.0, (double)f200, (double)(f200 - f0));
    printk("[ONTOGENY]   %-8.0f %+.1f%%     %+.1f pp\n",
           1000.0, (double)f1000, (double)(f1000 - f0));
    printk("[ONTOGENY] ================================================\n\n");

    /* Reset and set up final agent (λ=200) for the rest of the demo */
    printk("[ONTOGENY] Building final agent with λ=200 for lifecycle demo...\n");
    reset_agent_for_ablation();
    float fisher_T1[TCW_MLP_N_WEIGHTS], theta_T1[TCW_MLP_N_WEIGHTS];
    float fisher_T2[TCW_MLP_N_WEIGHTS], theta_T2[TCW_MLP_N_WEIGHTS];
    train_no_ewc(T1_XMIN, T1_XMAX);
    snapshot_fisher_theta(T1_XMIN, T1_XMAX, fisher_T1, theta_T1);
    train_with_ewc(T2_XMIN, T2_XMAX, fisher_T1, theta_T1, NULL, NULL, 200.0f);
    snapshot_fisher_theta(T2_XMIN, T2_XMAX, fisher_T2, theta_T2);
    train_with_ewc(T3_XMIN, T3_XMAX, fisher_T1, theta_T1, fisher_T2, theta_T2, 200.0f);

    float ho_final = compute_heldout_mse();
    int nonzero = 0;
    const float *w = (const float *)agent.phi.working;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++) if (w[j] != 0.0f) nonzero++;
    printk("[ONTOGENY] Final agent: held-out=%.5f nonzero=%d/25\n\n",
           (double)ho_final, nonzero);

    ontogeny_done = true;
    struct msg dm = { .type = MSG_ONTogeny_DONE, .heldout_mse = ho_final };
    k_msgq_put(&queue_to_control, &dm, K_NO_WAIT);

    /* Continuous etopo */
    while (1) {
        k_sleep(K_MSEC(1500));
        if (!agent.phi.alive) return;
        float x_all[30], y_all[30];
        int n = 0;
        for (int i = 0; i < 10; i++) {
            float x = T1_XMIN + (T1_XMAX - T1_XMIN) * i / 9.0f;
            x_all[n] = x / TWO_PI; y_all[n] = sinf(x) - tcw_ai_infer_raw(x); n++;
        }
        for (int i = 0; i < 10; i++) {
            float x = T2_XMIN + (T2_XMAX - T2_XMIN) * i / 9.0f;
            x_all[n] = x / TWO_PI; y_all[n] = sinf(x) - tcw_ai_infer_raw(x); n++;
        }
        for (int i = 0; i < 10; i++) {
            float x = T3_XMIN + (T3_XMAX - T3_XMIN) * i / 9.0f;
            x_all[n] = x / TWO_PI; y_all[n] = sinf(x) - tcw_ai_infer_raw(x); n++;
        }
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_etopo(&agent, x_all, y_all, n, ETOPO_LR, ETOPO_EPOCHS);
        float ho = compute_heldout_mse();
        k_mutex_unlock(&phi_lock);
        printk("[ETOPO    ] maintained, held-out=%.5f\n", (double)ho);
        struct msg m = { .type = MSG_ETOPO_DONE, .heldout_mse = ho, .rank = agent.phi.rank };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }
}

static void inference_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    for (int iter = 1; iter <= DEMO_ITERATIONS; iter++) {
        if (!agent.phi.alive) {
            printk("[INFERENCE] iter=%d: DEAD\n", iter);
            k_sleep(K_MSEC(200));
            continue;
        }
        if (iter % DAMAGE_EVERY_N == 0) {
            k_mutex_lock(&phi_lock, K_FOREVER);
            tcw_damage(&agent, TCW_PHI_SIZE_BYTES, (uint32_t)iter);
            sanitize_phi();
            k_mutex_unlock(&phi_lock);
            printk("[INFERENCE] iter=%d -- PHI CORRUPTED\n", iter);
            struct msg c = { .type = MSG_INFERENCE_RESULT, .iter = iter, .corrupted = 1 };
            k_msgq_put(&queue_to_control, &c, K_NO_WAIT);
            struct msg r = { .type = MSG_INFERENCE_RESULT, .iter = iter };
            k_msgq_put(&queue_to_repair, &r, K_NO_WAIT);
            k_sleep(K_MSEC(200));
            continue;
        }
        float x = ((float)iter / (float)DEMO_ITERATIONS) * TWO_PI;
        k_mutex_lock(&phi_lock, K_FOREVER);
        float ym = tcw_ai_infer_raw(x);
        float yc = tcw_mlp_forward((const tcw_mlp_t *)agent.phi.working, x / TWO_PI);
        float ga = agent.gate.alpha;
        uint16_t rank = agent.phi.rank;
        k_mutex_unlock(&phi_lock);
        float yp = ym + ga * yc;
        float yt = sinf(x);
        float err = (yp - yt) * (yp - yt);
        printk("[INFERENCE] iter=%d x=%.2f y=%.3f y_true=%.3f err=%.5f b=%.2f rank=%u\n",
               iter, (double)x, (double)yp, (double)yt, (double)err, (double)ga, rank);
        struct msg m = { .type = MSG_INFERENCE_RESULT, .iter = iter,
                         .mse = err, .rank = rank, .gate_alpha = ga, .corrupted = 0 };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        k_sleep(K_MSEC(200));
    }
    printk("\n[INFERENCE] Demo complete\n");
    k_sleep(K_SECONDS(2));
}

static void repair_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_repair, &m, K_FOREVER);
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_repair(&agent);
        float ho = compute_heldout_mse();
        float hd = verify_homeostasis();
        k_mutex_unlock(&phi_lock);
        if (s == TCW_OK)
            printk("[REPAIR   ] SUCCESS iter=%u held-out=%.5f homeo|Δy|=%.6f\n",
                   m.iter, (double)ho, (double)hd);
        struct msg r = { .type = MSG_REPAIR_COMPLETE, .iter = m.iter,
                         .heldout_mse = ho, .rank = agent.phi.rank };
        k_msgq_put(&queue_to_control, &r, K_NO_WAIT);
    }
}

static void aging_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (!ontogeny_done) k_sleep(K_MSEC(100));
    /* v0.9.1: 500ms grace for composition burst to complete */
    k_sleep(K_MSEC(AGING_DELAY_AFTER_ONTO));
    printk("[AGING    ] starting to age\n");
    while (1) {
        k_sleep(K_MSEC(AGING_EVERY_MS));
        if (!agent.phi.alive) return;
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_aging(&agent, AGING_STEP);
        float ho = compute_heldout_mse();
        float hd = verify_homeostasis();
        uint16_t rank = agent.phi.rank;
        bool alive = agent.phi.alive;
        k_mutex_unlock(&phi_lock);
        if (s == TCW_OK) {
            printk("[AGING    ] rank->%u held-out=%.5f homeo|Δy|=%.6f%s\n",
                   rank, (double)ho, (double)hd, alive ? "" : "  *** DEAD ***");
            struct msg m = { .type = MSG_AGING_DONE, .rank = rank, .heldout_mse = ho };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
            if (!alive) return;
        }
    }
}

static void normalization_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (!ontogeny_done) k_sleep(K_MSEC(100));
    int count = 0;
    while (1) {
        k_sleep(K_MSEC(NORMALIZE_EVERY_MS));
        if (!agent.phi.alive) return;
        count++;
        float alpha = (count % 2 == 1) ? 0.9f : (1.0f / 0.9f);
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_normalization(&agent, alpha);
        float ho = compute_heldout_mse();
        float hd = verify_homeostasis();
        k_mutex_unlock(&phi_lock);
        if (s == TCW_OK) {
            printk("[NORM     ] #%d α=%.2f held-out=%.5f homeo|Δy|=%.6f\n",
                   count, (double)alpha, (double)ho, (double)hd);
            struct msg m = { .type = MSG_NORMALIZATION_DONE, .iter = (uint32_t)count,
                             .heldout_mse = ho, .rank = agent.phi.rank };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
}

static void mutation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (!ontogeny_done) k_sleep(K_MSEC(100));
    int count = 0;
    while (1) {
        k_sleep(K_MSEC(MUTATION_EVERY_MS));
        if (!agent.phi.alive) return;
        count++;
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_status_t s = tcw_mutation(&agent, (uint32_t)(1000 + count));
        float ho = compute_heldout_mse();
        float hd = verify_homeostasis();
        float ga = agent.gate.alpha;
        k_mutex_unlock(&phi_lock);
        if (s == TCW_OK) {
            printk("[MUTATION ] env #%d held-out=%.5f homeo|Δy|=%.6f b=%.2f\n",
                   count, (double)ho, (double)hd, (double)ga);
            struct msg m = { .type = MSG_MUTATION_DONE, .iter = (uint32_t)count,
                             .heldout_mse = ho, .rank = agent.phi.rank, .gate_alpha = ga };
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }
    }
}

static void proliferation_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (!ontogeny_done) k_sleep(K_MSEC(100));
    static tcw_agent_t children[N_CHILDREN];
    int generation = 0;
    while (1) {
        k_sleep(K_MSEC(PROLIFERATION_EVERY_MS));
        if (!agent.phi.alive) { printk("[PROLIF   ] parent dead\n"); return; }
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_agent_t snap = agent;
        k_mutex_unlock(&phi_lock);
        tcw_status_t s = tcw_proliferation(&snap, children, N_CHILDREN,
                                            (uint32_t)(generation + 1), PROLIF_DROPOUT_P);
        if (s != TCW_OK) continue;
        generation++;
        printk("[PROLIF   ] gen #%d: %d children (dropout %.0f%%)\n",
               generation, N_CHILDREN, (double)(PROLIF_DROPOUT_P * 100.0f));
        float x_probe = 1.5f;
        float ym_probe = tcw_ai_infer_raw(x_probe);
        printk("[PROLIF   ] GENETIC PROOF at x=%.2f:\n", (double)x_probe);
        printk("[PROLIF   ]   y_model (C, shared) = %.6f\n", (double)ym_probe);
        for (int i = 0; i < N_CHILDREN; i++) {
            float yc = tcw_mlp_forward((const tcw_mlp_t *)children[i].phi.working,
                                        x_probe / TWO_PI);
            float yp = ym_probe + children[i].gate.alpha * yc;
            float mse = eval_agent_mse(&children[i]);
            printk("[PROLIF   ]   child#%d: y_final=%.6f (Φ differs) held-out=%.5f\n",
                   i + 1, (double)yp, (double)mse);
        }
        struct msg m = { .type = MSG_PROGENY_DONE, .iter = (uint32_t)generation };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }
}

static int count_nonzero(const tcw_agent_t *ag) {
    const float *w = (const float *)ag->phi.working;
    int n = 0;
    for (int j = 0; j < TCW_MLP_N_WEIGHTS; j++)
        if (w[j] != 0.0f) n++;
    return n;
}

static void composition_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (!ontogeny_done) k_sleep(K_MSEC(100));
    static tcw_agent_t kids[2];
    static tcw_agent_t hybrid;
    int gen = 0;

    /* v0.9.1: 3 rapid bursts (50ms apart) — all before first aging */
    for (int burst = 0; burst < COMPOSITION_BURST_COUNT; burst++) {
        k_sleep(K_MSEC(COMPOSITION_BURST_MS));
        if (!agent.phi.alive) return;
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_agent_t snap = agent;
        k_mutex_unlock(&phi_lock);
        tcw_proliferation(&snap, kids, 2, (uint32_t)(gen * 2 + 7), PROLIF_DROPOUT_P);
        gen++;
        tcw_status_t s = tcw_composition(&hybrid, &kids[0], &kids[1], 0.5f);
        if (s != TCW_OK) continue;
        float mp = eval_agent_mse(&snap);
        float m1 = eval_agent_mse(&kids[0]);
        float m2 = eval_agent_mse(&kids[1]);
        float mh = eval_agent_mse(&hybrid);
        printk("[COMPOSE  ] burst #%d: parent=%.5f kid1=%.5f kid2=%.5f hybrid=%.5f\n",
               burst + 1, (double)mp, (double)m1, (double)m2, (double)mh);
        printk("[COMPOSE  ]          nonzero: parent=%d kid1=%d kid2=%d hybrid=%d\n",
               count_nonzero(&snap), count_nonzero(&kids[0]),
               count_nonzero(&kids[1]), count_nonzero(&hybrid));
        struct msg m = { .type = MSG_COMPOSITION_DONE, .iter = (uint32_t)gen,
                         .mse = mh, .heldout_mse = mh };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }

    while (1) {
        k_sleep(K_MSEC(COMPOSITION_PERIODIC_MS));
        if (!agent.phi.alive) { printk("[COMPOSE  ] parent dead\n"); return; }
        k_mutex_lock(&phi_lock, K_FOREVER);
        tcw_agent_t snap = agent;
        k_mutex_unlock(&phi_lock);
        tcw_proliferation(&snap, kids, 2, (uint32_t)(gen * 2 + 7), PROLIF_DROPOUT_P);
        gen++;
        tcw_status_t s = tcw_composition(&hybrid, &kids[0], &kids[1], 0.5f);
        if (s != TCW_OK) continue;
        float mp = eval_agent_mse(&snap);
        float m1 = eval_agent_mse(&kids[0]);
        float m2 = eval_agent_mse(&kids[1]);
        float mh = eval_agent_mse(&hybrid);
        printk("[COMPOSE  ] gen #%d: parent=%.5f kid1=%.5f kid2=%.5f hybrid=%.5f\n",
               gen, (double)mp, (double)m1, (double)m2, (double)mh);
        struct msg m = { .type = MSG_COMPOSITION_DONE, .iter = (uint32_t)gen,
                         .mse = mh, .heldout_mse = mh };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }
}

static void homeostasis_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    while (1) {
        k_sleep(K_MSEC(HOMEOSTASIS_EVERY_MS));
        float hd = verify_homeostasis();
        struct msg m = { .type = MSG_HOMEOSTASIS_REPORT, .mse = hd, .heldout_mse = hd };
        k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
    }
}

static void control_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    int damages = 0, repairs = 0, agings = 0, mutations = 0,
        proliferations = 0, etopos = 0, compositions = 0, normalizations = 0;
    float mse_sum = 0.0f, mse_min = 1e9f, mse_max = 0.0f;
    int mse_cnt = 0;
    float ho_sum = 0.0f, ho_min = 1e9f, ho_max = 0.0f;
    int ho_cnt = 0;
    float homeo_max_seen = 0.0f;

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_control, &m, K_FOREVER);
        if (m.type == MSG_HOMEOSTASIS_REPORT) {
            if (m.mse > homeo_max_seen) homeo_max_seen = m.mse;
            continue;
        }
        if (m.type == MSG_INFERENCE_RESULT) {
            if (m.corrupted) damages++;
            else if (m.mse > 0.0f && m.mse < 1e3f) {
                mse_sum += m.mse; mse_cnt++;
                if (m.mse < mse_min) mse_min = m.mse;
                if (m.mse > mse_max) mse_max = m.mse;
            }
            continue;
        }
        if (m.type == MSG_COMPOSITION_DONE) { compositions++; continue; }
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
                printk("\n[CONTROL  ] *** AGENT DIED OF OLD AGE ***\n");
                printk("[CONTROL  ] LIFECYCLE: damages=%d repairs=%d agings=%d "
                       "mutations=%d prolif=%d etopos=%d compositions=%d norms=%d\n",
                       damages, repairs, agings, mutations, proliferations, etopos, compositions, normalizations);
                if (mse_cnt)
                    printk("[CONTROL  ] infer MSE:    min=%.5f max=%.5f avg=%.5f\n",
                           (double)mse_min, (double)mse_max, (double)(mse_sum/mse_cnt));
                if (ho_cnt)
                    printk("[CONTROL  ] held-out MSE: min=%.5f max=%.5f avg=%.5f\n",
                           (double)ho_min, (double)ho_max, (double)(ho_sum/ho_cnt));
                printk("[CONTROL  ] HOMEOSTASIS: max |Δy_model| = %.8f\n", (double)homeo_max_seen);
                printk("[CONTROL  ]   (%s)\n",
                       homeo_max_seen < 0.0001f ? "PROVEN: C unchanged" : "WARNING: C drifted");
            }
            break;
        case MSG_MUTATION_DONE: mutations++; break;
        case MSG_PROGENY_DONE:  proliferations++; break;
        case MSG_ETOPO_DONE:    etopos++; break;
        case MSG_NORMALIZATION_DONE: normalizations++; break;
        case MSG_ONTogeny_DONE: printk("[CONTROL  ] Ontogeny complete\n"); break;
        default: break;
        }
    }
}

K_THREAD_STACK_DEFINE(inference_stack,     STACK_SIZE);
K_THREAD_STACK_DEFINE(repair_stack,        STACK_SIZE);
K_THREAD_STACK_DEFINE(aging_stack,         STACK_SIZE);
K_THREAD_STACK_DEFINE(etopo_stack,         STACK_SIZE);
K_THREAD_STACK_DEFINE(mutation_stack,      STACK_SIZE);
K_THREAD_STACK_DEFINE(proliferation_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(composition_stack,   STACK_SIZE);
K_THREAD_STACK_DEFINE(normalization_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(homeostasis_stack,   STACK_SIZE);
K_THREAD_STACK_DEFINE(control_stack,       STACK_SIZE);

static struct k_thread inference_tcb, repair_tcb, aging_tcb, etopo_tcb,
                      mutation_tcb, proliferation_tcb, composition_tcb,
                      normalization_tcb, homeostasis_tcb, control_tcb;

int main(void)
{
    printk("\n");
    printk("==================================================\n");
    printk("  TCW-OS v0.9.3 — Single-anchor EWC + larger residual\n");
    printk("  Single-anchor T3, λ ∈ {0,50,200,1000}\n");
    printk("==================================================\n\n");

    tcw_ai_init();
    printk("TFLite Micro: sine-wave model, C frozen\n");

    capture_homeostasis_reference();
    printk("Homeostasis reference captured\n");

    uint8_t genome[TCW_PHI_SIZE_BYTES];
    memset(genome, 0, TCW_PHI_SIZE_BYTES);

    tcw_status_t s = tcw_init(&agent, genome, TCW_PHI_SIZE_BYTES, 16);
    if (s != TCW_OK) { printk("FATAL\n"); return 0; }
    printk("Agent born: rank=%u, b=%.2f\n", agent.phi.rank, (double)agent.gate.alpha);

    float ho = compute_heldout_mse();
    printk("Baseline held-out MSE = %.5f\n\n", (double)ho);

    k_thread_create(&repair_tcb, repair_stack, STACK_SIZE,
                    repair_thread, NULL, NULL, NULL, PRIO_REPAIR, 0, K_NO_WAIT);
    k_thread_name_set(&repair_tcb, "tcw_repair");

    k_thread_create(&etopo_tcb, etopo_stack, STACK_SIZE,
                    ontogeny_thread, NULL, NULL, NULL, PRIO_ETOPO, 0, K_NO_WAIT);
    k_thread_name_set(&etopo_tcb, "tcw_ontogeny");

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

    k_thread_create(&composition_tcb, composition_stack, STACK_SIZE,
                    composition_thread, NULL, NULL, NULL, PRIO_COMPOSITION, 0, K_NO_WAIT);
    k_thread_name_set(&composition_tcb, "tcw_compose");

    k_thread_create(&normalization_tcb, normalization_stack, STACK_SIZE,
                    normalization_thread, NULL, NULL, NULL, PRIO_AGING, 0, K_NO_WAIT);
    k_thread_name_set(&normalization_tcb, "tcw_norm");

    k_thread_create(&homeostasis_tcb, homeostasis_stack, STACK_SIZE,
                    homeostasis_thread, NULL, NULL, NULL, PRIO_HOMEOSTASIS, 0, K_NO_WAIT);
    k_thread_name_set(&homeostasis_tcb, "tcw_homeo");

    return 0;
}
