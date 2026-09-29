#include "stream_ac.h"

#include <math.h>
#include <string.h>

#define LN_EPS 1e-5f      /* F.layer_norm default */
#define LRELU_SLOPE 0.01f /* F.leaky_relu default */
#define NORM_EPS 1e-8     /* NormalizeObservation / ScaleReward */

/* ---------------------------------------------------------------- arena */

static uint8_t arena[SAC_ARENA_BYTES] __attribute__((aligned(8)));
static uint32_t arena_used;

static void *alloc(uint32_t bytes) {
    bytes = (bytes + 7u) & ~7u;
    if (arena_used + bytes > SAC_ARENA_BYTES) return 0;
    void *p = arena + arena_used;
    arena_used += bytes;
    memset(p, 0, bytes);
    return p;
}

/* ---------------------------------------------------------------- rng */

static uint64_t rng_state;
static int has_spare;
static float spare;

static uint32_t rng_u32(void) { /* xorshift64* */
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}

static float rng_uniform(void) { /* (0, 1) */
    return ((float)(rng_u32() >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

static float rng_normal(void) { /* Box-Muller */
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    float r = sqrtf(-2.0f * logf(rng_uniform())), t = 6.2831853071795864f * rng_uniform();
    spare = r * sinf(t);
    has_spare = 1;
    return r * cosf(t);
}

/* ---------------------------------------------------------------- mlp */

typedef struct {
    int in, hidden, out, n;
    float *p, *z;                     /* parameters and eligibility traces, same layout */
    float *n1, *y1, *n2, *y2, *h_out; /* activations of the last forward pass */
    float rstd1, rstd2;
} mlp_t;

static float *grad_a, *grad_b; /* backward scratch, `hidden` floats each */

static int mlp_size(int in, int hidden, int out) {
    return hidden * in + hidden + hidden * hidden + hidden + out * hidden + out;
}

static int mlp_alloc(mlp_t *m, int in, int hidden, int out, float *p, float *z) {
    m->in = in, m->hidden = hidden, m->out = out, m->n = mlp_size(in, hidden, out);
    m->p = p, m->z = z;
    m->n1 = alloc(4u * hidden), m->y1 = alloc(4u * hidden);
    m->n2 = alloc(4u * hidden), m->y2 = alloc(4u * hidden);
    m->h_out = alloc(4u * out);
    return m->n1 && m->y1 && m->n2 && m->y2 && m->h_out;
}

/* LeCun-uniform, then zero ceil(0.9 * fan_in) incoming weights of each unit (keeping at least one). */
static void sparse_init(float *w, int fan_out, int fan_in, int *perm) {
    float bound = sqrtf(1.0f / (float)fan_in);
    int zeros = (int)ceil(0.9 * fan_in);
    if (zeros > fan_in - 1) zeros = fan_in - 1;
    for (int r = 0; r < fan_out; r++) {
        float *row = w + r * fan_in;
        for (int j = 0; j < fan_in; j++) row[j] = (2.0f * rng_uniform() - 1.0f) * bound, perm[j] = j;
        for (int j = 0; j < zeros; j++) { /* partial Fisher-Yates */
            int k = j + (int)(rng_u32() % (uint32_t)(fan_in - j)), t = perm[j];
            perm[j] = perm[k], perm[k] = t;
            row[perm[j]] = 0.0f;
        }
    }
}

static void mlp_init(mlp_t *m, int *perm) {
    int I = m->in, H = m->hidden, O = m->out;
    float *W1 = m->p, *W2 = W1 + H * I + H, *W3 = W2 + H * H + H; /* biases stay zero */
    sparse_init(W1, H, I, perm);
    sparse_init(W2, H, H, perm);
    sparse_init(W3, O, H, perm);
}

static void linear(const float *W, const float *b, const float *x, float *y, int out, int in) {
    for (int i = 0; i < out; i++) {
        const float *w = W + i * in;
        float acc = b[i];
        for (int j = 0; j < in; j++) acc += w[j] * x[j];
        y[i] = acc;
    }
}

/* In place: h -> LayerNorm(h) stored in n, LeakyReLU(n) stored in y. Returns 1/std. */
static float ln_lrelu(const float *h, float *n, float *y, int H) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < H; i++) mean += h[i];
    mean /= (float)H;
    for (int i = 0; i < H; i++) var += (h[i] - mean) * (h[i] - mean);
    float rstd = 1.0f / sqrtf(var / (float)H + LN_EPS);
    for (int i = 0; i < H; i++) {
        n[i] = (h[i] - mean) * rstd;
        y[i] = n[i] > 0.0f ? n[i] : LRELU_SLOPE * n[i];
    }
    return rstd;
}

static const float *mlp_forward(mlp_t *m, const float *x) {
    int I = m->in, H = m->hidden, O = m->out;
    const float *W1 = m->p, *b1 = W1 + H * I, *W2 = b1 + H, *b2 = W2 + H * H, *W3 = b2 + H, *b3 = W3 + O * H;
    linear(W1, b1, x, m->n1, H, I);
    m->rstd1 = ln_lrelu(m->n1, m->n1, m->y1, H);
    linear(W2, b2, m->y1, m->n2, H, H);
    m->rstd2 = ln_lrelu(m->n2, m->n2, m->y2, H);
    linear(W3, b3, m->y2, m->h_out, O, H);
    return m->h_out;
}

/* g: gradient w.r.t. the LeakyReLU output -> gradient w.r.t. the LayerNorm input, in place. */
static void ln_lrelu_backward(float *g, const float *n, float rstd, int H) {
    float mean_g = 0.0f, mean_gn = 0.0f;
    for (int i = 0; i < H; i++) {
        g[i] *= n[i] > 0.0f ? 1.0f : LRELU_SLOPE;
        mean_g += g[i], mean_gn += g[i] * n[i];
    }
    mean_g /= (float)H, mean_gn /= (float)H;
    for (int i = 0; i < H; i++) g[i] = rstd * (g[i] - mean_g - n[i] * mean_gn);
}

/* Adds d(loss)/d(params) to the traces, given d(loss)/d(outputs) of the last forward pass on x. */
static void mlp_backward(mlp_t *m, const float *x, const float *dout) {
    int I = m->in, H = m->hidden, O = m->out;
    const float *W2 = m->p + H * I + H, *W3 = W2 + H * H + H;
    float *zW1 = m->z, *zb1 = zW1 + H * I, *zW2 = zb1 + H, *zb2 = zW2 + H * H, *zW3 = zb2 + H, *zb3 = zW3 + O * H;

    memset(grad_a, 0, 4u * H);
    for (int k = 0; k < O; k++) {
        float d = dout[k], *zr = zW3 + k * H;
        const float *wr = W3 + k * H;
        zb3[k] += d;
        for (int i = 0; i < H; i++) zr[i] += d * m->y2[i], grad_a[i] += wr[i] * d;
    }
    ln_lrelu_backward(grad_a, m->n2, m->rstd2, H);

    memset(grad_b, 0, 4u * H);
    for (int i = 0; i < H; i++) {
        float d = grad_a[i], *zr = zW2 + i * H;
        const float *wr = W2 + i * H;
        zb2[i] += d;
        for (int j = 0; j < H; j++) zr[j] += d * m->y1[j], grad_b[j] += wr[j] * d;
    }
    ln_lrelu_backward(grad_b, m->n1, m->rstd1, H);

    for (int i = 0; i < H; i++) {
        float d = grad_b[i], *zr = zW1 + i * I;
        zb1[i] += d;
        for (int j = 0; j < I; j++) zr[j] += d * x[j];
    }
}

static void traces_decay(mlp_t *m, float factor) {
    for (int i = 0; i < m->n; i++) m->z[i] *= factor;
}

/* ObGD: step size shrunk so that lr * kappa * max(|delta|, 1) * ||z||_1 <= 1. */
static void obgd_step(mlp_t *m, float lr, float kappa, float delta, int reset) {
    double z_sum = 0.0;
    for (int i = 0; i < m->n; i++) z_sum += fabsf(m->z[i]);
    float bound = lr * kappa * fmaxf(fabsf(delta), 1.0f) * (float)z_sum;
    float scale = -(bound > 1.0f ? lr / bound : lr) * delta;
    for (int i = 0; i < m->n; i++) m->p[i] += scale * m->z[i];
    if (reset) memset(m->z, 0, 4u * m->n);
}

/* ---------------------------------------------------------------- agent */

static sac_config_t cfg;
static int ready, in_episode, in_dim;
static mlp_t actor, critic;
static float *params, *traces;
static float *frames, *last_action, *features, *s, *s_next, *a, *dout;
static sac_stats_t obs_stats, ret_stats;
static double ret_acc;

static int stats_alloc(sac_stats_t *st, int n) {
    st->count = 0;
    st->mean = alloc(8u * n), st->var = alloc(8u * n), st->m2 = alloc(8u * n);
    if (!st->mean || !st->var || !st->m2) return 0;
    for (int i = 0; i < n; i++) st->var[i] = 1.0;
    return 1;
}

/* Welford, exactly as RunningMeanStd.update. */
static void stats_update(sac_stats_t *st, const double *x, int n) {
    st->count++;
    for (int i = 0; i < n; i++) {
        double delta = x[i] - st->mean[i];
        st->mean[i] += delta / (double)st->count;
        st->m2[i] += delta * (x[i] - st->mean[i]);
        if (st->count > 1) st->var[i] = st->m2[i] / (double)(st->count - 1);
    }
}

int sac_init(const sac_config_t *c) {
    cfg = *c;
    ready = in_episode = 0;
    arena_used = 0;
    rng_state = cfg.seed ? cfg.seed : 0x9E3779B97F4A7C15ULL;
    has_spare = 0;
    ret_acc = 0.0;

    int A = cfg.act_dim, H = cfg.hidden;
    in_dim = cfg.obs_dim * cfg.num_frames + A;
    int na = mlp_size(in_dim, H, 2 * A), nc = mlp_size(in_dim, H, 1);
    params = alloc(4u * (na + nc));
    traces = alloc(4u * (na + nc));
    if (!params || !traces) return SAC_ERR_MEMORY;
    if (!mlp_alloc(&actor, in_dim, H, 2 * A, params, traces)) return SAC_ERR_MEMORY;
    if (!mlp_alloc(&critic, in_dim, H, 1, params + na, traces + na)) return SAC_ERR_MEMORY;

    grad_a = alloc(4u * H), grad_b = alloc(4u * H);
    frames = alloc(4u * cfg.obs_dim * cfg.num_frames), last_action = alloc(4u * A);
    features = alloc(4u * in_dim), s = alloc(4u * in_dim), s_next = alloc(4u * in_dim);
    a = alloc(4u * A), dout = alloc(4u * 2 * A);
    int n_perm = in_dim > H ? in_dim : H;
    int *perm = alloc(4u * n_perm);
    if (!grad_a || !grad_b || !frames || !last_action || !features || !s || !s_next || !a || !dout || !perm)
        return SAC_ERR_MEMORY;
    if (!stats_alloc(&obs_stats, in_dim) || !stats_alloc(&ret_stats, 1)) return SAC_ERR_MEMORY;

    mlp_init(&actor, perm);
    mlp_init(&critic, perm);
    ready = 1;
    return SAC_OK;
}

int sac_ready(void) { return ready; }
int sac_in_dim(void) { return in_dim; }
int sac_act_dim(void) { return cfg.act_dim; }
int sac_obs_dim(void) { return cfg.obs_dim; }
float *sac_params(void) { return params; }
int sac_num_actor_params(void) { return actor.n; }
int sac_num_critic_params(void) { return critic.n; }
const sac_stats_t *sac_obs_stats(void) { return &obs_stats; }
const sac_stats_t *sac_return_stats(void) { return &ret_stats; }

static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static float softplus_grad(float x) { return x > 20.0f ? 1.0f : 1.0f / (1.0f + expf(-x)); }

void sac_forward(const float *obs, float *mu, float *std, float *value) {
    const float *out = mlp_forward(&actor, obs);
    for (int k = 0; k < cfg.act_dim; k++) mu[k] = out[k], std[k] = softplus(out[cfg.act_dim + k]);
    *value = mlp_forward(&critic, obs)[0];
}

static void sample_action(const float *obs, float *action) {
    const float *out = mlp_forward(&actor, obs);
    for (int k = 0; k < cfg.act_dim; k++) action[k] = out[k] + softplus(out[cfg.act_dim + k]) * rng_normal();
}

float sac_update(const float *obs, const float *action, float reward, const float *next_obs, int terminated, int done) {
    int A = cfg.act_dim;
    float next_value = mlp_forward(&critic, next_obs)[0];
    float value = mlp_forward(&critic, obs)[0]; /* last: the backward pass uses these activations */
    float delta = reward + cfg.gamma * (terminated ? 0.0f : 1.0f) * next_value - value;

    /* Loss = -log pi(a|s) - c * sign(delta) * entropy, gradients w.r.t. [mu, pre_std]. */
    const float *out = mlp_forward(&actor, obs);
    float ent = cfg.entropy_coeff * (float)((delta > 0.0f) - (delta < 0.0f));
    for (int k = 0; k < A; k++) {
        float mu = out[k], p = out[A + k], sd = softplus(p), diff = action[k] - mu;
        dout[k] = -diff / (sd * sd);
        dout[A + k] = (1.0f / sd - diff * diff / (sd * sd * sd) - ent / sd) * softplus_grad(p);
    }
    float minus_one = -1.0f; /* loss = -v(s) */

    float decay = cfg.gamma * cfg.lamda;
    traces_decay(&actor, decay);
    traces_decay(&critic, decay);
    mlp_backward(&actor, obs, dout);
    mlp_backward(&critic, obs, &minus_one);
    obgd_step(&actor, cfg.lr, cfg.kappa_policy, delta, done);
    obgd_step(&critic, cfg.lr, cfg.kappa_value, delta, done);
    return delta;
}

/* Features [obs_{t-k+1}, ..., obs_t, a_{t-1}] -> normalized, as HistoryObservation + NormalizeObservation. */
static void features_to_state(float *out) {
    int F = cfg.obs_dim * cfg.num_frames;
    memcpy(features, frames, 4u * F);
    memcpy(features + F, last_action, 4u * cfg.act_dim);
    if (!cfg.normalize) {
        memcpy(out, features, 4u * in_dim);
        return;
    }
    double x[in_dim];
    for (int i = 0; i < in_dim; i++) x[i] = features[i];
    stats_update(&obs_stats, x, in_dim);
    for (int i = 0; i < in_dim; i++) out[i] = (float)((x[i] - obs_stats.mean[i]) / sqrt(obs_stats.var[i] + NORM_EPS));
}

/* ScaleReward: divide by the running std of the discounted return. */
static float scale_reward(double r, int terminated, int truncated) {
    if (!cfg.normalize) return (float)r;
    ret_acc = ret_acc * (double)cfg.gamma * (terminated ? 0.0 : 1.0) + r;
    stats_update(&ret_stats, &ret_acc, 1);
    if (terminated || truncated) ret_acc = 0.0;
    return (float)(r / sqrt(ret_stats.var[0] + NORM_EPS));
}

void sac_reset(const float *raw_obs, float *action) {
    for (int f = 0; f < cfg.num_frames; f++) memcpy(frames + f * cfg.obs_dim, raw_obs, 4u * cfg.obs_dim);
    memset(last_action, 0, 4u * cfg.act_dim);
    features_to_state(s);
    sample_action(s, a);
    memcpy(action, a, 4u * cfg.act_dim);
    in_episode = 1;
}

float sac_step(const float *raw_obs, double reward, int terminated, int truncated, float *action, int *has_action) {
    int F = cfg.obs_dim * cfg.num_frames, done = terminated || truncated;
    memmove(frames, frames + cfg.obs_dim, 4u * (F - cfg.obs_dim));
    memcpy(frames + F - cfg.obs_dim, raw_obs, 4u * cfg.obs_dim);
    for (int k = 0; k < cfg.act_dim; k++) last_action[k] = fminf(fmaxf(a[k], -1.0f), 1.0f);
    features_to_state(s_next);

    float delta = sac_update(s, a, scale_reward(reward, terminated, truncated), s_next, terminated, done);
    *has_action = !done;
    if (done) {
        in_episode = 0;
        return delta;
    }
    memcpy(s, s_next, 4u * in_dim);
    sample_action(s, a);
    memcpy(action, a, 4u * cfg.act_dim);
    return delta;
}

int sac_in_episode(void) { return in_episode; }
