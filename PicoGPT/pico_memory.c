#include <inttypes.h>
#include <malloc.h>
#include <stdio.h>

#include "hardware/regs/addressmap.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"

#include "pico_memory.h"
#include "gpt_multicore.h"

#define STACK_MARKER UINT32_C(0xa5c39e71)

extern char __StackBottom;
extern char __StackTop;
extern char __end__;
extern char __HeapLimit;
#if GPT_MULTICORE
extern char __StackOneBottom;
extern char __StackOneTop;
#endif
extern void* _sbrk(int increment);

static size_t sampled_heap_peak;

static uintptr_t heap_break(void) {
    void* current = _sbrk(0);
    uintptr_t address = (uintptr_t)current;
    if (current == (void*)-1 || address < (uintptr_t)&__end__ ||
        address > (uintptr_t)&__HeapLimit) {
        panic("Invalid heap boundaries in memory diagnostics");
    }
    return address;
}

void pico_memory_sample(void) {
    size_t extent = heap_break() - (uintptr_t)&__end__;
    if (extent > sampled_heap_peak) {
        sampled_heap_peak = extent;
    }
}

__attribute__((noinline)) void pico_memory_init(void) {
    uint32_t interrupts = save_and_disable_interrupts();
    uintptr_t stack_pointer;
    __asm volatile ("mrs %0, msp" : "=r"(stack_pointer));
    uintptr_t bottom = (uintptr_t)&__StackBottom;
    uintptr_t top = (uintptr_t)&__StackTop;
    if (stack_pointer < bottom || stack_pointer > top ||
        (stack_pointer % sizeof(uint32_t)) != 0 || (bottom % sizeof(uint32_t)) != 0) {
        restore_interrupts(interrupts);
        panic("Invalid core 0 stack boundaries");
    }
    /* Only paint unused stack below the current frame; never repaint on reruns. */
    for (volatile uint32_t* word = (volatile uint32_t*)bottom;
         (uintptr_t)word < stack_pointer; word++) {
        *word = STACK_MARKER;
    }
    restore_interrupts(interrupts);
    pico_memory_sample();
}

static size_t stack_watermark(void) {
    uint32_t interrupts = save_and_disable_interrupts();
    uintptr_t top = (uintptr_t)&__StackTop;
    const volatile uint32_t* word = (const volatile uint32_t*)&__StackBottom;
    while ((uintptr_t)word < top && *word == STACK_MARKER) {
        word++;
    }
    size_t observed = top - (uintptr_t)word;
    restore_interrupts(interrupts);
    return observed;
}

void pico_memory_report(const char* label) {
    struct mallinfo heap = mallinfo();
    pico_memory_sample();
    uintptr_t current = heap_break();
    uintptr_t start = (uintptr_t)&__end__;
    uintptr_t limit = (uintptr_t)&__HeapLimit;
    size_t stack_capacity = (uintptr_t)&__StackTop - (uintptr_t)&__StackBottom;
    size_t stack_used = stack_watermark();

    printf("\nMemory snapshot (%s):\n", label);
    printf("  Fixed main RAM before heap: %" PRIuPTR " bytes (includes the entire arena).\n",
           start - SRAM_BASE);
    printf("  Heap extent: %" PRIuPTR " bytes; sampled extent peak: %zu bytes.\n",
           current - start, sampled_heap_peak);
    printf("  Heap allocated chunks: %zu bytes; free inside allocator: %zu bytes.\n",
           heap.uordblks, heap.fordblks);
    printf("  Main RAM not yet claimed by heap: %" PRIuPTR " bytes.\n", limit - current);
    printf("  Core 0 stack observed watermark: %zu/%zu bytes; untouched margin: %zu bytes.\n",
           stack_used, stack_capacity, stack_capacity - stack_used);
    printf("  Stack observation starts in main; excludes earlier startup and is not a worst-case bound.\n");
#if GPT_MULTICORE
    printf("  Core 1 stack reservation: %" PRIuPTR " bytes in scratch X; guard enabled, peak not measured.\n",
           (uintptr_t)&__StackOneTop - (uintptr_t)&__StackOneBottom);
#endif
    printf("  Arena/stack reservations are already counted; do not add their observed usage again.\n");
}
