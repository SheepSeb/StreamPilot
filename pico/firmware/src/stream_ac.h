/* Stream AC(lambda) with ObGD (Elsayed, Vasan & Mahmood, 2024) in plain C99 + libm.
 *
 * A port of streampilot/stream_x/agents.py and wrappers.py: the actor and critic MLPs
 * (Linear -> LayerNorm without affine -> LeakyReLU, twice, then linear heads), their hand-written
 * backward passes, ObGD with eligibility traces, the observation history and the online
 * observation/reward normalization. No heap: everything lives in one static arena.
 *
 * Parameter layout of each network (row-major, float32), which the host mirrors:
 *   W1[hidden][in], b1[hidden], W2[hidden][hidden], b2[hidden], W3[out][hidden], b3[out]
 * The actor's head is heads.0 (mean) stacked on heads.1 (pre-softplus std): out = 2 * act_dim.
 * The critic's head has out = 1.
 */
#ifndef STREAM_AC_H
#define STREAM_AC_H

#include <stdint.h>

#ifndef SAC_ARENA_BYTES
#define SAC_ARENA_BYTES (384u * 1024u)
#endif

typedef struct {
    int32_t obs_dim;     /* raw observation size sent by the environment */
    int32_t act_dim;
    int32_t num_frames;  /* observations stacked into the features */
    int32_t hidden;
    int32_t normalize;   /* online observation normalization and reward scaling */
    float lr, gamma, lamda, kappa_policy, kappa_value, entropy_coeff;
    uint64_t seed;
} sac_config_t;

typedef struct {
    uint64_t count;
    double *mean, *var, *m2;
} sac_stats_t;

enum { SAC_OK = 0, SAC_ERR_MEMORY = 3, SAC_ERR_STATE = 4 };

/* Allocates both networks (sparse init) and the normalization state. Returns SAC_OK or SAC_ERR_MEMORY. */
int sac_init(const sac_config_t *cfg);
int sac_ready(void);
int sac_in_episode(void);
int sac_in_dim(void);
int sac_act_dim(void);
int sac_obs_dim(void);

/* Actor parameters followed by critic parameters, one flat array (see the layout above). */
float *sac_params(void);
int sac_num_actor_params(void);
int sac_num_critic_params(void);

/* Start an episode from a raw observation; writes the sampled action. */
void sac_reset(const float *raw_obs, float *action);
/* One environment transition from the raw next observation and raw reward. Runs the TD(lambda)
 * update, then (unless the episode ended) samples the next action. Returns the TD error and sets
 * *has_action to 0 when terminated or truncated (the host must call sac_reset next). */
float sac_step(const float *raw_obs, double reward, int terminated, int truncated, float *action, int *has_action);

/* Lower-level entry points on already-normalized features, for testing against PyTorch. */
float sac_update(const float *obs, const float *action, float reward, const float *next_obs, int terminated, int done);
void sac_forward(const float *obs, float *mu, float *std, float *value);

const sac_stats_t *sac_obs_stats(void);
const sac_stats_t *sac_return_stats(void);

#endif
