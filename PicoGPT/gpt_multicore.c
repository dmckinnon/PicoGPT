#include <stdio.h>

#include "pico/multicore.h"
#include "pico/sem.h"
#include "pico/stdlib.h"

#include "gpt_weights/gpt_constants.h"
#include "gpt_utils.h"

#if !GPT_MULTICORE
#error "Only compile gpt_multicore.c for a multicore build"
#endif

typedef enum { JOB_MATMUL, JOB_GELU } JobKind;

typedef struct {
    JobKind kind;
    const float* a;
    const float* b;
    const float* residual;
    float* output;
    size_t rows;
    size_t inner;
    size_t columns;
} CoreJob;

static CoreJob shared_job;
static semaphore_t work_ready;
static semaphore_t work_done;
static bool initialized;
static bool busy;

static void execute(const CoreJob* job) {
    if (job->kind == JOB_GELU) {
        apply_gelu_single_core(job->output, job->rows);
    } else if (job->residual) {
        matmul_with_residual_single_core(job->a, job->rows, job->inner,
                                        job->b, job->columns, job->residual, job->output);
    } else {
        matmul_single_core(job->a, job->rows, job->inner, job->b, job->columns, job->output);
    }
}

static void signal_worker(semaphore_t* semaphore) {
    if (!sem_release(semaphore)) {
        panic("PicoGPT worker semaphore already signaled");
    }
}

static void core1_worker(void) {
    signal_worker(&work_done);
    while (true) {
        sem_acquire_blocking(&work_ready);
        /* Semaphore acquire/release orders the descriptor and tensor accesses. */
        CoreJob job = shared_job;
        execute(&job);
        signal_worker(&work_done);
    }
}

static void require_core0(void) {
    if (get_core_num() != 0 || __get_current_exception() != 0) {
        panic("PicoGPT dispatch requires core 0 foreground context");
    }
}

void gpt_multicore_init(void) {
    require_core0();
    if (initialized) return;
    sem_init(&work_ready, 0, 1);
    sem_init(&work_done, 0, 1);
    multicore_launch_core1(core1_worker);
    sem_acquire_blocking(&work_done);
    initialized = true;
}

static void run_job(CoreJob job) {
    require_core0();
    if (!initialized || busy) {
        panic("PicoGPT worker not initialized or dispatch is reentrant");
    }
    if (job.rows < 2 || (job.kind == JOB_MATMUL && (!job.inner || !job.columns))) {
        execute(&job);
        return;
    }
    busy = true;
    size_t first_rows = job.rows - job.rows / 2;
    CoreJob second = job;
    second.rows -= first_rows;
    if (job.kind == JOB_GELU) {
        second.output += first_rows;
    } else {
        second.a += first_rows * job.inner;
        second.output += first_rows * job.columns;
        if (second.residual) second.residual += first_rows * job.columns;
    }
    shared_job = second;
    signal_worker(&work_ready);
    job.rows = first_rows;
    execute(&job);
    sem_acquire_blocking(&work_done);
    busy = false;
}

void gpt_multicore_matmul(
    const float* a, size_t rows, size_t inner,
    const float* b, size_t columns, const float* residual, float* output
) {
    run_job((CoreJob){JOB_MATMUL, a, b, residual, output, rows, inner, columns});
}

void gpt_multicore_gelu(float* data, size_t length) {
    run_job((CoreJob){JOB_GELU, NULL, NULL, NULL, data, length, 0, 0});
}
