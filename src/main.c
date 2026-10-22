#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <string.h>
#include "tcw_runtime.h"

#define STACK_SIZE          2048
#define PRIO_REPAIR         1
#define PRIO_CONTROL        3
#define PRIO_INFERENCE      5
#define DEMO_ITERATIONS     20
#define DAMAGE_EVERY_N      5

enum msg_type {
    MSG_INFERENCE_RESULT,
    MSG_REPAIR_NEEDED,
    MSG_REPAIR_COMPLETE,
};

struct msg {
    enum msg_type type;
    uint32_t      value;
    uint32_t      checksum;
};

K_MSGQ_DEFINE(queue_to_control, sizeof(struct msg), 16, 4);
K_MSGQ_DEFINE(queue_to_repair,  sizeof(struct msg), 4,  4);

static tcw_agent_t agent;
static uint8_t     fake_genome[512];

static void inference_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    for (int iter = 1; iter <= DEMO_ITERATIONS; iter++) {
        struct msg m = { .type = MSG_INFERENCE_RESULT, .value = (uint32_t)iter };

        if (iter % DAMAGE_EVERY_N == 0) {
            tcw_damage(&agent, 100, (uint32_t)iter);
            m.checksum = 0;
            printk("[INFERENCE] iter=%d -- PHI CORRUPTED (100 bytes damaged)\n", iter);

            /* Fix: control gets the damage FIRST, then we wake repair */
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);

            struct msg r = { .type = MSG_REPAIR_NEEDED, .value = (uint32_t)iter };
            k_msgq_put(&queue_to_repair, &r, K_NO_WAIT);
        } else {
            m.checksum = agent.phi.checksum;
            printk("[INFERENCE] iter=%d: phi OK (crc=0x%08x)\n", iter, m.checksum);
            k_msgq_put(&queue_to_control, &m, K_NO_WAIT);
        }

        k_sleep(K_MSEC(200));
    }

    printk("\n[INFERENCE] Demo complete: %d iterations, %d damages\n",
           DEMO_ITERATIONS, DEMO_ITERATIONS / DAMAGE_EVERY_N);
    k_sleep(K_SECONDS(1));
}

static void repair_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_repair, &m, K_FOREVER);

        printk("[REPAIR   ] triggered for iter=%u, restoring PHI...\n", m.value);

        uint64_t t0 = k_uptime_get();
        tcw_status_t s = tcw_repair(&agent);
        uint64_t t1 = k_uptime_get();

        struct msg resp = {
            .type = MSG_REPAIR_COMPLETE,
            .value = (uint32_t)(t1 - t0),
            .checksum = (s == TCW_OK) ? agent.phi.checksum : 0,
        };

        if (s == TCW_OK) {
            printk("[REPAIR   ] SUCCESS in %u ms, crc restored to 0x%08x\n",
                   resp.value, resp.checksum);
        } else {
            printk("[REPAIR   ] FAILED (code %d)\n", s);
        }
        k_msgq_put(&queue_to_control, &resp, K_NO_WAIT);
    }
}

static void control_thread(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    int repairs_done = 0, damages_seen = 0;

    while (1) {
        struct msg m;
        k_msgq_get(&queue_to_control, &m, K_FOREVER);

        switch (m.type) {
        case MSG_INFERENCE_RESULT:
            if (m.checksum == 0) {
                damages_seen++;
                printk("[CONTROL  ] iter=%u: DAMAGE detected -> requesting repair\n", m.value);
            } else {
                printk("[CONTROL  ] iter=%u: agent healthy\n", m.value);
            }
            break;

        case MSG_REPAIR_COMPLETE:
            repairs_done++;
            printk("[CONTROL  ] REPAIR #%d confirmed (took %u ms)\n",
                   repairs_done, m.value);
            printk("[CONTROL  ] SUMMARY: %d damages, %d repairs -- agent alive\n\n",
                   damages_seen, repairs_done);
            break;

        default:
            break;
        }
    }
}

K_THREAD_STACK_DEFINE(inference_stack, STACK_SIZE);
K_THREAD_STACK_DEFINE(repair_stack,    STACK_SIZE);
K_THREAD_STACK_DEFINE(control_stack,   STACK_SIZE);

static struct k_thread inference_tcb;
static struct k_thread repair_tcb;
static struct k_thread control_tcb;

int main(void)
{
    printk("\n");
    printk("==============================================\n");
    printk("  TCW-OS v0.2: Multi-threaded Living AI\n");
    printk("  Runtime: Zephyr OS\n");
    printk("==============================================\n\n");

    for (size_t i = 0; i < sizeof(fake_genome); i++)
        fake_genome[i] = (uint8_t)(i * 7 + 3);

    tcw_status_t s = tcw_init(&agent, fake_genome, sizeof(fake_genome), 16);
    if (s != TCW_OK) {
        printk("FATAL: tcw_init failed (code %d)\n", s);
        return 0;
    }
    printk("Agent born: rank=%u, crc=0x%08x\n\n",
           agent.phi.rank, agent.phi.checksum);

    k_thread_create(&repair_tcb, repair_stack, STACK_SIZE,
                    repair_thread, NULL, NULL, NULL,
                    PRIO_REPAIR, 0, K_NO_WAIT);
    k_thread_name_set(&repair_tcb, "tcw_repair");

    k_thread_create(&control_tcb, control_stack, STACK_SIZE,
                    control_thread, NULL, NULL, NULL,
                    PRIO_CONTROL, 0, K_NO_WAIT);
    k_thread_name_set(&control_tcb, "tcw_control");

    k_thread_create(&inference_tcb, inference_stack, STACK_SIZE,
                    inference_thread, NULL, NULL, NULL,
                    PRIO_INFERENCE, 0, K_NO_WAIT);
    k_thread_name_set(&inference_tcb, "tcw_inference");

    return 0;
}
