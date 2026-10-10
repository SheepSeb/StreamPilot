#include "protocol.h"

#include <string.h>
#include <time.h>

#include "stream_ac.h"

__attribute__((weak)) uint32_t proto_time_us(void) {
    return (uint32_t)((uint64_t)clock() * 1000000u / CLOCKS_PER_SEC);
}

static uint32_t rd_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static void wr_u32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4), *p += 4; }
static void wr_f32(uint8_t **p, float v) { memcpy(*p, &v, 4), *p += 4; }
static void wr_bytes(uint8_t **p, const void *v, uint32_t n) { memcpy(*p, v, n), *p += n; }

static void wr_stats(uint8_t **p, const sac_stats_t *st, int n) {
    wr_bytes(p, &st->count, 8);
    wr_bytes(p, st->mean, 8u * n);
    wr_bytes(p, st->var, 8u * n);
    wr_bytes(p, st->m2, 8u * n);
}

/* Floats in the payload may be unaligned: copy them into aligned scratch first. */
static float scratch_a[PROTO_MAX_PAYLOAD / 4], scratch_b[PROTO_MAX_PAYLOAD / 4], scratch_c[64];

uint8_t proto_handle(uint8_t cmd, const uint8_t *in, uint32_t len, uint8_t *out, uint32_t *out_len) {
    uint8_t *o = out;
    *out_len = 0;

    if (cmd == 'P') {
        wr_u32(&o, PROTO_VERSION);
        wr_u32(&o, SAC_ARENA_BYTES);
    } else if (cmd == 'I') {
        if (len != 5 * 4 + 6 * 4 + 8) return PROTO_ERR_LEN;
        sac_config_t c;
        memcpy(&c.obs_dim, in, 4), memcpy(&c.act_dim, in + 4, 4), memcpy(&c.num_frames, in + 8, 4);
        memcpy(&c.hidden, in + 12, 4), memcpy(&c.normalize, in + 16, 4);
        float f[6];
        memcpy(f, in + 20, sizeof f);
        c.lr = f[0], c.gamma = f[1], c.lamda = f[2], c.kappa_policy = f[3], c.kappa_value = f[4], c.entropy_coeff = f[5];
        memcpy(&c.seed, in + 44, 8);
        if (c.obs_dim < 1 || c.act_dim < 1 || c.act_dim > 16 || c.num_frames < 1 || c.hidden < 2) return PROTO_ERR_LEN;
        if (sac_init(&c) != SAC_OK) return PROTO_ERR_MEMORY;
        wr_u32(&o, (uint32_t)sac_num_actor_params());
        wr_u32(&o, (uint32_t)sac_num_critic_params());
        wr_u32(&o, (uint32_t)sac_in_dim());
    } else if (!sac_ready()) {
        return PROTO_ERR_STATE;
    } else if (cmd == 'W' || cmd == 'G') {
        uint32_t total = (uint32_t)(sac_num_actor_params() + sac_num_critic_params());
        if (len < 4) return PROTO_ERR_LEN;
        uint32_t offset = rd_u32(in);
        uint32_t count = cmd == 'W' ? (len - 4) / 4 : (len == 8 ? rd_u32(in + 4) : 0xFFFFFFFFu);
        if ((cmd == 'W' && (len - 4) % 4) || count > PROTO_MAX_PAYLOAD / 4 || offset > total || count > total - offset)
            return PROTO_ERR_LEN;
        if (cmd == 'W') memcpy(sac_params() + offset, in + 4, 4u * count);
        else wr_bytes(&o, sac_params() + offset, 4u * count);
    } else if (cmd == 'R' || cmd == 'S') {
        uint32_t n_obs = 4u * sac_obs_dim(), A = (uint32_t)sac_act_dim();
        if (len != (cmd == 'R' ? n_obs : n_obs + 10)) return PROTO_ERR_LEN;
        memcpy(scratch_a, in, n_obs);
        if (cmd == 'R') {
            sac_reset(scratch_a, scratch_c);
            wr_bytes(&o, scratch_c, 4u * A);
        } else {
            if (!sac_in_episode()) return PROTO_ERR_STATE;
            double reward;
            memcpy(&reward, in + n_obs, 8);
            int has_action;
            uint32_t t0 = proto_time_us();
            float delta = sac_step(scratch_a, reward, in[n_obs + 8], in[n_obs + 9], scratch_c, &has_action);
            uint32_t us = proto_time_us() - t0;
            wr_f32(&o, delta);
            wr_u32(&o, us);
            *o++ = (uint8_t)has_action;
            if (has_action) wr_bytes(&o, scratch_c, 4u * A);
        }
    } else if (cmd == 'N') {
        if (8u + 3u * 8u * sac_in_dim() + 32u > PROTO_MAX_PAYLOAD) return PROTO_ERR_LEN;
        wr_stats(&o, sac_obs_stats(), sac_in_dim());
        wr_stats(&o, sac_return_stats(), 1);
    } else if (cmd == 'U') {
        uint32_t n_in = 4u * sac_in_dim(), n_act = 4u * sac_act_dim();
        if (len != 2 * n_in + n_act + 4 + 2) return PROTO_ERR_LEN;
        float reward;
        memcpy(scratch_a, in, n_in);
        memcpy(scratch_c, in + n_in, n_act);
        memcpy(&reward, in + n_in + n_act, 4);
        memcpy(scratch_b, in + n_in + n_act + 4, n_in);
        const uint8_t *flags = in + 2 * n_in + n_act + 4;
        wr_f32(&o, sac_update(scratch_a, scratch_c, reward, scratch_b, flags[0], flags[1]));
    } else if (cmd == 'F') {
        uint32_t n_in = 4u * sac_in_dim(), A = (uint32_t)sac_act_dim();
        if (len != n_in) return PROTO_ERR_LEN;
        memcpy(scratch_a, in, n_in);
        float value;
        sac_forward(scratch_a, scratch_c, scratch_c + A, &value);
        wr_bytes(&o, scratch_c, 8u * A);
        wr_f32(&o, value);
    } else {
        return PROTO_ERR_CMD;
    }
    *out_len = (uint32_t)(o - out);
    return PROTO_OK;
}
