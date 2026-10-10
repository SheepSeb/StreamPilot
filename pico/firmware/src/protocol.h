/* Binary request/response protocol over a serial line. All integers and floats little-endian.
 *
 *   request:  0xA5, cmd, len (u16), payload[len]
 *   response: 0x5A, status, len (u16), payload[len]
 *
 *   'P' ping         -> u32 version, u32 arena bytes
 *   'I' init         i32 obs_dim, act_dim, num_frames, hidden, normalize,
 *                    f32 lr, gamma, lamda, kappa_policy, kappa_value, entropy_coeff, u64 seed
 *                    -> u32 actor params, u32 critic params, u32 feature dim
 *   'W' set params   u32 offset, f32[] values (flat actor || critic)
 *   'G' get params   u32 offset, u32 count -> f32[count]
 *   'R' reset        f32 raw_obs[obs_dim] -> f32 action[act_dim]
 *   'S' step         f32 raw_obs[obs_dim], f64 reward, u8 terminated, u8 truncated
 *                    -> f32 td_error, u32 update_us, u8 has_action, f32 action[act_dim] (if has_action)
 *   'N' stats        -> observation stats then return stats, each: u64 count, f64 mean[n], var[n], m2[n]
 *   'U' update       f32 obs[in], f32 action[act], f32 reward, f32 next_obs[in], u8 terminated, u8 done
 *                    -> f32 td_error   (normalized features; for testing)
 *   'F' forward      f32 obs[in] -> f32 mu[act], f32 std[act], f32 value   (for testing)
 */
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

#define PROTO_VERSION 1u
#define PROTO_REQ_MAGIC 0xA5
#define PROTO_RESP_MAGIC 0x5A
#define PROTO_MAX_PAYLOAD 4096u

enum { PROTO_OK = 0, PROTO_ERR_CMD = 1, PROTO_ERR_LEN = 2, PROTO_ERR_MEMORY = 3, PROTO_ERR_STATE = 4 };

/* Handles one request. Writes at most PROTO_MAX_PAYLOAD bytes to out. Returns the status byte. */
uint8_t proto_handle(uint8_t cmd, const uint8_t *in, uint32_t in_len, uint8_t *out, uint32_t *out_len);

/* Microsecond clock for timing the update; the firmware overrides this weak default. */
uint32_t proto_time_us(void);

#endif
