#ifndef GPT_PROFILE_H
#define GPT_PROFILE_H

#include <stdint.h>

#ifndef GPT_PROFILE_ENABLED
#define GPT_PROFILE_ENABLED 0
#endif

typedef enum {
    GPT_STAGE_EMBEDDINGS,
    GPT_STAGE_PRE_NORM,
    GPT_STAGE_QKV,
    GPT_STAGE_SCORES,
    GPT_STAGE_MASK_SOFTMAX,
    GPT_STAGE_WEIGHTED_VALUES,
    GPT_STAGE_ATTN_PROJECTION,
    GPT_STAGE_ATTN_RESIDUAL,
    GPT_STAGE_FFN_UP,
    GPT_STAGE_GELU,
    GPT_STAGE_FFN_DOWN_RESIDUAL,
    GPT_STAGE_FINAL_NORM,
    GPT_STAGE_LOGITS,
    GPT_STAGE_COUNT
} GPTProfileStage;

#if GPT_PROFILE_ENABLED
uint64_t gpt_profile_now(void);
void gpt_profile_record(GPTProfileStage stage, uint64_t elapsed_us);
#define GPT_PROFILE_BEGIN(stage) uint64_t gpt_profile_start_##stage = gpt_profile_now()
#define GPT_PROFILE_END(stage) \
    gpt_profile_record(stage, gpt_profile_now() - gpt_profile_start_##stage)
#else
#define GPT_PROFILE_BEGIN(stage) ((void)0)
#define GPT_PROFILE_END(stage) ((void)0)
#endif

#endif // GPT_PROFILE_H
