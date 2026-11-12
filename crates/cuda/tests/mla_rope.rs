//! The full model's rotary MLA path (GLM-5.3 full / `glm_moe_dsa`): the rope tables, the KV write with the rotated rope key, the
//! query preparation (absorb 192 + rotate 64) and the dense attention with the rope term, each against a host reference.
//!
//!     cargo test --release -p gb-cuda --test mla_rope -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla::{self, *};
use gb_cuda::num::f32_to_bf16;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

fn enc(dt: Dtype, x: f32) -> u16 {
    match dt {
        Dtype::BF16 => f32_to_bf16(x),
        _ => f64_to_f16(x as f64),
    }
}

fn dec(dt: Dtype, v: u16) -> f32 {
    match dt {
        Dtype::BF16 => gb_cuda::num::bf16_to_f32(v),
        _ => gb_exl3::f16::f16_to_f64(v) as f32,
    }
}

/// The rope pair rotation of the reference: pairs (2i, 2i+1), results half-split.
fn rope_rotate(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let h = ROPE_DIM / 2;
    let mut y = vec![0.0f32; ROPE_DIM];
    for i in 0..h {
        let (x0, x1) = (x[2 * i], x[2 * i + 1]);
        y[i] = x0 * cos[i] - x1 * sin[i];
        y[i + h] = x0 * sin[i] + x1 * cos[i];
    }
    y
}

fn rope_theta() -> f32 {
    8_000_000.0
}

#[test]
fn the_rope_tables_match_the_reference_formula() {
    let max_pos = 9;
    let (cos, sin) = mla::rope_tables(max_pos, rope_theta());
    let h = ROPE_DIM / 2;
    for i in 0..h {
        let inv = 1.0f32 / rope_theta().powf((2 * i) as f32 / ROPE_DIM as f32);
        for p in 0..max_pos {
            let a = p as f32 * inv;
            assert!((cos[p * h + i] - a.cos()).abs() < 1e-6, "cos[{p}][{i}]");
            assert!((sin[p * h + i] - a.sin()).abs() < 1e-6, "sin[{p}][{i}]");
        }
    }
}

fn one_case(dt: Dtype, rows: usize, pos: &[i32]) {
    let st = Stream::new().unwrap();
    let max_pos = *pos.iter().max().unwrap() as usize + 2;
    let (cos_h, sin_h) = mla::rope_tables(max_pos, rope_theta());
    let cos = DevBuf::from_slice(
        &cos_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    let sin = DevBuf::from_slice(
        &sin_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();
    let mut r = Rng(0x9E3779B9 ^ (rows as u64) ^ (dt as u64));

    // q: [rows][64 * 256], head h at h*256 = [nope 192 | rope 64]
    let hs = 64usize;
    let qk = NOPE_DIM_FULL + ROPE_DIM;
    let mut q_host = vec![0u16; rows * hs * qk];
    for v in q_host.iter_mut() {
        *v = enc(dt, r.uniform(-0.5, 0.5));
    }
    let q = DevBuf::from_slice(&q_host).unwrap();

    // w_uk: [64][192][512]
    let mut uk_host = vec![0u16; hs * NOPE_DIM_FULL * LATENT_DIM];
    for v in uk_host.iter_mut() {
        *v = enc(dt, r.uniform(-0.05, 0.05));
    }
    let uk = DevBuf::from_slice(&uk_host).unwrap();
    let w = MlaWeights { uk, uvt: DevBuf::alloc(1).unwrap(), dtype: dt };

    // the latent cache and the block table (one slot, pos rows on page 0)
    let n_blocks = 2;
    let mut cache = LatentCache::new_rope(n_blocks, dt).unwrap();
    let mut table_host = vec![0u32; 4];
    table_host[0] = 0;
    let table = BlockTable::from_host(&[table_host], 4).unwrap();
    let slots = vec![0i32; rows];
    let ri = RowIndex::from_host(&slots, pos).unwrap();

    // the cache holds one latent + rope key per position 0..max_pos, all in slot 0
    let kv_stride = LATENT_DIM + ROPE_DIM;
    let mut lat_host = vec![0u16; max_pos * LATENT_DIM];
    let mut kv_host = vec![0u16; max_pos * kv_stride];
    let mut lat_f = vec![vec![0f32; LATENT_DIM]; max_pos];
    let mut rope_f = vec![vec![0f32; ROPE_DIM]; max_pos];
    for p in 0..max_pos {
        for c in 0..LATENT_DIM {
            lat_host[p * LATENT_DIM + c] = enc(dt, r.uniform(-1.0, 1.0));
            lat_f[p][c] = dec(dt, lat_host[p * LATENT_DIM + c]);
        }
        for c in 0..ROPE_DIM {
            kv_host[p * kv_stride + LATENT_DIM + c] = enc(dt, r.uniform(-1.0, 1.0));
            rope_f[p][c] = dec(dt, kv_host[p * kv_stride + LATENT_DIM + c]);
        }
    }
    let lat = DevBuf::from_slice(&lat_host).unwrap();
    let kv = DevBuf::from_slice(&kv_host).unwrap();
    let cache_slots = vec![0i32; max_pos];
    let cache_pos: Vec<i32> = (0..max_pos as i32).collect();
    let cache_rows = RowIndex::from_host(&cache_slots, &cache_pos).unwrap();
    latent_write_rope(&lat, &kv, kv_stride, LATENT_DIM, &cos, &sin, ROPE_DIM / 2, &cache_rows, &table, &mut cache, max_pos, &st)
        .unwrap();

    // query preparation: absorb + rotate
    let mut q_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut qrot = DevBuf::alloc(rows * hs * ROPE_DIM * 2).unwrap();
    absorb_rope(&q, hs * qk, qk, NOPE_DIM_FULL, &w, &mut q_lat, &mut qrot, &cos, &sin, ROPE_DIM / 2, &ri, rows, &st)
        .unwrap();
    st.sync().unwrap();

    let q_lat_out: Vec<u16> = q_lat.to_vec().unwrap();
    let qrot_out: Vec<u16> = qrot.to_vec().unwrap();

    for row in 0..rows {
        let q_row: Vec<f32> = (0..hs * qk)
            .map(|i| dec(dt, q_host[row * hs * qk + i]))
            .collect();
        for h in 0..hs {
            // q_lat = q_nope @ W_UK, d ascending, fp32
            for c in (0..LATENT_DIM).step_by(37) {
                let mut acc = 0.0f32;
                for d in 0..NOPE_DIM_FULL {
                    acc += q_row[h * qk + d] * dec(dt, uk_host[(h * NOPE_DIM_FULL + d) * LATENT_DIM + c]);
                }
                let got = dec(dt, q_lat_out[(row * hs + h) * LATENT_DIM + c]);
                assert!((got - acc).abs() <= 2e-2 + 2e-2 * acc.abs(), "q_lat[{row}][{h}][{c}] {got} vs {acc}");
            }
            // q_rot = rotate(q_rope at pos)
            let p = if pos[row] >= 0 { pos[row] as usize } else { 0 };
            let hh = ROPE_DIM / 2;
            let want = rope_rotate(
                &q_row[h * qk + NOPE_DIM_FULL..h * qk + qk],
                &cos_h[p * hh..(p + 1) * hh],
                &sin_h[p * hh..(p + 1) * hh],
            );
            for d in (0..ROPE_DIM).step_by(7) {
                let got = dec(dt, qrot_out[(row * hs + h) * ROPE_DIM + d]);
                if pos[row] < 0 {
                    assert_eq!(got, 0.0, "padding row's q_rot");
                } else {
                    assert!((got - want[d]).abs() < 5e-3, "q_rot[{row}][{h}][{d}] {got} vs {}", want[d]);
                }
            }
        }
    }

    // attention: dense rows only; a padding row stays zero
    let mut o_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut ws = DevBuf::alloc(mla::attend_workspace_bytes(rows, max_pos)).unwrap();
    attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
    st.sync().unwrap();
    let o_out: Vec<u16> = o_lat.to_vec().unwrap();

    let scale = 1.0f32 / 16.0; // 256^-0.5
    for row in 0..rows {
        if pos[row] < 0 {
            for c in 0..LATENT_DIM {
                assert_eq!(dec(dt, o_out[(row * hs) * LATENT_DIM + c]), 0.0, "padding row's o_lat");
            }
            continue;
        }
        let n = pos[row] as usize + 1;
        for h in 0..hs {
            let ql: Vec<f32> = (0..LATENT_DIM)
                .map(|c| dec(dt, q_lat_out[(row * hs + h) * LATENT_DIM + c]))
                .collect();
            let qr: Vec<f32> = (0..ROPE_DIM).map(|d| dec(dt, qrot_out[(row * hs + h) * ROPE_DIM + d])).collect();
            // k_rot: the rotated key each position stored
            let mut krot = vec![vec![0f32; ROPE_DIM]; max_pos];
            for j in 0..n {
                let hh = ROPE_DIM / 2;
                let cosj = &cos_h[j * hh..(j + 1) * hh];
                let sinj = &sin_h[j * hh..(j + 1) * hh];
                krot[j] = rope_rotate(&rope_f[j][..], cosj, sinj);
            }
            let mut scores = vec![0f32; n];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                let mut s = 0.0f32;
                for c in 0..LATENT_DIM {
                    s += ql[c] * lat_f[j][c];
                }
                for d in 0..ROPE_DIM {
                    s += qr[d] * krot[j][d];
                }
                scores[j] = s * scale;
                mx = mx.max(scores[j]);
            }
            let mut den = 0.0f32;
            for j in 0..n {
                scores[j] = (scores[j] - mx).exp();
                den += scores[j];
            }
            for c in (0..LATENT_DIM).step_by(29) {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += scores[j] / den * lat_f[j][c];
                }
                let got = dec(dt, o_out[(row * hs + h) * LATENT_DIM + c]);
                assert!((got - acc).abs() <= 5e-2 + 5e-2 * acc.abs(), "o_lat[{row}][{h}][{c}] {got} vs {acc}");
            }
        }
    }
}

#[test]
fn the_rope_path_matches_the_host_reference_fp16_and_bf16() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    for dt in [Dtype::F16, Dtype::BF16] {
        one_case(dt, 1, &[2]);
        one_case(dt, 2, &[0, 3]);
        one_case(dt, 2, &[-1, 1]);
    }
}
