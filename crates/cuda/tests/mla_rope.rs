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
    one_case_ctx(dt, rows, pos, None)
}

/// [`one_case`] with an explicit context bound: the engine's slabs pass the *call's* last position as `max_context`, which can be
/// far past the slab's own positions (middle slabs of a long prefill).
fn one_case_ctx(dt: Dtype, rows: usize, pos: &[i32], max_ctx: Option<usize>) {
    let st = Stream::new().unwrap();
    let max_pos = max_ctx.unwrap_or(*pos.iter().max().unwrap() as usize + 2);
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

    // the latent cache and the block table: one page per 256 positions (a single page cannot hold a context past 256, and a
    // table that maps every block to page 0 makes the attention read another position's slot)
    let pages = max_pos.div_ceil(BLOCK_TOKENS) + 1;
    let n_blocks = pages;
    let mut cache = LatentCache::new_rope(n_blocks, dt).unwrap();
    let table_host: Vec<u32> = (0..pages as u32).collect();
    let table = BlockTable::from_host(&[table_host], pages).unwrap();
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

    // the workspace's prior content must not reach the output: fill it with two different patterns and compare (the engine
    // reuses one workspace across layers and ranks, so a stale read shows up as a rank divergence there)
    {
        let n = ws.bytes();
        let mut junk: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(37) ^ 0x5a) as u8).collect();
        ws.upload(&junk).unwrap();
        attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let first: Vec<u16> = o_lat.to_vec().unwrap();
        junk.reverse();
        ws.upload(&junk).unwrap();
        attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let second: Vec<u16> = o_lat.to_vec().unwrap();
        assert_eq!(first, second, "attend_rope depends on the workspace's prior content (rows {rows}, max_pos {max_pos})");
    }

    // a shared-memory poison: run the kernel once with NaN queries (which fills the CTAs' shared buffers with NaN), restore
    // the real queries, and run again. A byte of shared memory the kernel reads without writing shows up as a NaN here.
    {
        let nan_q: Vec<u16> = (0..q_lat_out.len()).map(|i| if i % 3 == 0 { 0x7e00u16 } else { q_lat_out[i] }).collect();
        q_lat.upload(&nan_q).unwrap();
        attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
        st.sync().unwrap();
        q_lat.upload(&q_lat_out).unwrap();
        attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let poisoned: Vec<u16> = o_lat.to_vec().unwrap();
        let nan_out = poisoned.iter().filter(|v| (0x7c00u16..=0x7fffu16).contains(&(**v & 0x7fffu16))).count();
        assert_eq!(nan_out, 0, "attend_rope read shared memory it never wrote (rows {rows}, max_pos {max_pos})");
        assert_eq!(poisoned, o_out, "attend_rope changed after a NaN run (rows {rows}, max_pos {max_pos})");
    }

    // the same call again on the same buffers: identical inputs must give identical bits (a rank divergence in the
    // engine starts as a difference here, with every input already proven equal by the dumps)
    for k in 0..16 {
        attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let again: Vec<u16> = o_lat.to_vec().unwrap();
        assert_eq!(again, o_out, "attend_rope differs on repeat {k} (rows {rows}, max_pos {max_pos})");
    }

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

/// The dense attention across several 256-key chunks: where the merge of chunk partials first breaks.
#[test]
fn the_rope_path_holds_across_chunk_boundaries() {
    if !device_available() {
        return;
    }
    eprintln!("case rows 128 positions 512..639, context 886 (a middle slab)");
    one_case_ctx(Dtype::F16, 128, &(512..640).collect::<Vec<i32>>(), Some(886));
    eprintln!("case rows 1 position 500 head-check done");
    for start in [60usize, 100, 130, 160, 200, 250] {
        eprintln!("case rows 1 position {start}");
        one_case(Dtype::F16, 1, &[start as i32]);
    }
    for (rows, start) in [(2usize, 250usize), (2, 500), (2, 760), (2, 878), (8, 878), (128, 760)] {
        let pos: Vec<i32> = (start..start + rows).map(|p| p as i32).collect();
        eprintln!("case rows {rows} positions {start}..{}", start + rows - 1);
        one_case(Dtype::F16, rows, &pos);
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
        let pref: Vec<i32> = (0..18).collect();
        one_case(dt, 18, &pref);
        let big: Vec<i32> = (0..128).collect();
        one_case(dt, 128, &big);
    }
}

/// The chunk partials of a two-chunk row against the host's own chunked softmax (the merge's inputs).
#[test]
fn chunk_partials_of_a_two_chunk_row_match_the_host() {
    if !device_available() {
        return;
    }
    use gb_cuda::DevBuf;
    let st = Stream::new().unwrap();
    let rows = 1usize;
    let pos = [500i32];
    let dt = Dtype::F16;
    let max_pos = 502usize;
    let (cos_h, sin_h) = mla::rope_tables(max_pos, rope_theta());
    let cos = DevBuf::from_slice(&cos_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let sin = DevBuf::from_slice(&sin_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let mut r = Rng(0x1234_5678);
    let hs = 64usize;
    let qk = NOPE_DIM_FULL + ROPE_DIM;
    let q_host: Vec<u16> = (0..rows * hs * qk).map(|_| enc(dt, r.uniform(-0.5, 0.5))).collect();
    let q = DevBuf::from_slice(&q_host).unwrap();
    let uk_host: Vec<u16> = (0..hs * NOPE_DIM_FULL * LATENT_DIM).map(|_| enc(dt, r.uniform(-0.05, 0.05))).collect();
    let uk = DevBuf::from_slice(&uk_host).unwrap();
    let w = MlaWeights { uk, uvt: DevBuf::alloc(1).unwrap(), dtype: dt };
    let n_blocks = 4usize;
    let mut cache = LatentCache::new_rope(n_blocks, dt).unwrap();
    let mut table_host = vec![0u32; 8];
    for (i, t) in table_host.iter_mut().enumerate() { *t = i as u32; }
    let table = BlockTable::from_host(&[table_host], 8).unwrap();
    let slots = vec![0i32; rows];
    let ri = RowIndex::from_host(&slots, &pos).unwrap();
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
    latent_write_rope(&lat, &kv, kv_stride, LATENT_DIM, &cos, &sin, ROPE_DIM / 2, &cache_rows, &table, &mut cache, max_pos, &st).unwrap();
    let mut q_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut qrot = DevBuf::alloc(rows * hs * ROPE_DIM * 2).unwrap();
    absorb_rope(&q, hs * qk, qk, NOPE_DIM_FULL, &w, &mut q_lat, &mut qrot, &cos, &sin, ROPE_DIM / 2, &ri, rows, &st).unwrap();
    let mut o_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut ws = DevBuf::alloc(mla::attend_workspace_bytes(rows, max_pos)).unwrap();
    attend_rope(&q_lat, &qrot, &cache, &table, &ri, rows, &mut o_lat, max_pos, max_pos, &mut ws, &st).unwrap();
    st.sync().unwrap();
    let ch = mla::CHUNK_KEYS;
    let nch = (max_pos + ch - 1) / ch;
    let pairs = rows * hs * nch;
    let wsv: Vec<f32> = ws.to_vec().unwrap();
    let (po, pm, pl) = (0usize, pairs * LATENT_DIM, pairs * LATENT_DIM + pairs);
    let ql = q_lat.to_vec::<u16>().unwrap();
    let qrv = qrot.to_vec::<u16>().unwrap();
    // host chunk partials for head 0
    let h = 0usize;
    let qlv: Vec<f32> = (0..LATENT_DIM).map(|c| dec(dt, ql[h * LATENT_DIM + c])).collect();
    let qrvv: Vec<f32> = (0..ROPE_DIM).map(|d| dec(dt, qrv[h * ROPE_DIM + d])).collect();
    let hh = ROPE_DIM / 2;
    let mut krot = vec![vec![0f32; ROPE_DIM]; max_pos];
    for j in 0..max_pos { krot[j] = rope_rotate(&rope_f[j][..], &cos_h[j * hh..(j + 1) * hh], &sin_h[j * hh..(j + 1) * hh]); }
    for c in 0..nch {
        let lo = c * ch;
        let hi = (lo + ch).min(pos[0] as usize + 1);
        if lo >= hi { continue; }
        let mut mx = f32::NEG_INFINITY;
        let mut s = vec![0f32; hi - lo];
        for j in lo..hi {
            let mut a = 0.0f32;
            for k in 0..LATENT_DIM { a += qlv[k] * lat_f[j][k]; }
            for d in 0..ROPE_DIM { a += qrvv[d] * krot[j][d]; }
            s[j - lo] = a / 16.0;
            mx = mx.max(s[j - lo]);
        }
        let e: Vec<f32> = s.iter().map(|v| (v - mx).exp()).collect();
        let l: f32 = e.iter().sum();
        let mut o = vec![0f32; LATENT_DIM];
        for (i, j) in (lo..hi).enumerate() { for k in 0..LATENT_DIM { o[k] += e[i] * lat_f[j][k]; } }
        let base = (h * 0 + h) * nch + c;
        let gm = wsv[pm + base];
        let gl = wsv[pl + base];
        let go = &wsv[po + base * LATENT_DIM..po + (base + 1) * LATENT_DIM];
        let cos_o = |a: &[f32], b: &[f32]| -> f32 {
            let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
            for i in 0..a.len() { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
            d / (na.sqrt() * nb.sqrt() + 1e-12)
        };
        eprintln!("chunk {c}: host m {mx:.4} l {l:.4} | gpu m {gm:.4} l {gl:.4} | o cos {:.5} (gpu |o| {:.4}, host |o| {:.4})",
            cos_o(go, &o), go.iter().map(|v| v * v).sum::<f32>().sqrt(), o.iter().map(|v| v * v).sum::<f32>().sqrt());
        // a host merge of the GPU's own partials, then the kernel's merged row against it
        let mut gm_ = -f32::INFINITY;
        for cc in 0..nch {
            let b = (h * 0 + h) * nch + cc;
            gm_ = gm_.max(wsv[pm + b]);
        }
        let mut ml = 0f32;
        let mut mo = vec![0f32; LATENT_DIM];
        for cc in 0..nch {
            let b = (h * 0 + h) * nch + cc;
            let (cm, cl) = (wsv[pm + b], wsv[pl + b]);
            if !(cl > 0.0) { continue; }
            let w = (cm - gm_).exp2();
            ml += cl * w;
            for k in 0..LATENT_DIM { mo[k] += wsv[po + b * LATENT_DIM + k] * w; }
        }
        for k in 0..LATENT_DIM { mo[k] /= ml; }
        let outv = o_lat.to_vec::<u16>().unwrap();
        let ko: Vec<f32> = (0..LATENT_DIM).map(|k| dec(dt, outv[h * LATENT_DIM + k])).collect();
        eprintln!("  merged: host-of-gpu-partials |o| {:.4} vs kernel |o| {:.4} cos {:.5}",
            mo.iter().map(|v| v * v).sum::<f32>().sqrt(), ko.iter().map(|v| v * v).sum::<f32>().sqrt(), cos_o(&mo, &ko));
    }
}

/// The sparse (DSA) attention over a synthetic selection, against the host, and its hash printed: two GPUs running this must
/// agree bit for bit.
#[test]
fn the_sparse_path_matches_the_host_and_is_gpu_independent() {
    if !device_available() {
        return;
    }
    use gb_cuda::DevBuf;
    let st = Stream::new().unwrap();
    let dt = Dtype::F16;
    let rows = 4usize;
    let pos = [5000i32, 5001, 5002, 5003];
    let max_pos = 5006usize;
    let (cos_h, sin_h) = mla::rope_tables(max_pos, rope_theta());
    let cos = DevBuf::from_slice(&cos_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let sin = DevBuf::from_slice(&sin_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let mut r = Rng(0xDEAD_BEEF);
    let hs = 64usize;
    let qk = NOPE_DIM_FULL + ROPE_DIM;
    let q_host: Vec<u16> = (0..rows * hs * qk).map(|_| enc(dt, r.uniform(-0.5, 0.5))).collect();
    let q = DevBuf::from_slice(&q_host).unwrap();
    let uk_host: Vec<u16> = (0..hs * NOPE_DIM_FULL * LATENT_DIM).map(|_| enc(dt, r.uniform(-0.05, 0.05))).collect();
    let uk = DevBuf::from_slice(&uk_host).unwrap();
    let w = MlaWeights { uk, uvt: DevBuf::alloc(1).unwrap(), dtype: dt };
    let pages = max_pos.div_ceil(BLOCK_TOKENS) + 1;
    let mut cache = LatentCache::new_rope(pages, dt).unwrap();
    let table_host: Vec<u32> = (0..pages as u32).collect();
    let table = BlockTable::from_host(&[table_host], pages).unwrap();
    let slots = vec![0i32; rows];
    let ri = RowIndex::from_host(&slots, &pos).unwrap();
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
    latent_write_rope(&lat, &kv, kv_stride, LATENT_DIM, &cos, &sin, ROPE_DIM / 2, &cache_rows, &table, &mut cache, max_pos, &st).unwrap();
    let mut q_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut qrot = DevBuf::alloc(rows * hs * ROPE_DIM * 2).unwrap();
    absorb_rope(&q, hs * qk, qk, NOPE_DIM_FULL, &w, &mut q_lat, &mut qrot, &cos, &sin, ROPE_DIM / 2, &ri, rows, &st).unwrap();
    // a synthetic selection per row: 2048 keys, a deterministic subset of 0..=pos (spread out), ascending
    let k = 2048usize;
    let stride = gb_cuda::glm_idx::K_STRIDE;
    let mut sel_host = vec![-1i32; rows * stride];
    let mut meta_host = vec![0i32; rows * 2];
    let mut chosen: Vec<Vec<usize>> = Vec::new();
    for rr in 0..rows {
        let n = pos[rr] as usize + 1;
        let take = n.min(k);
        let stepf = n as f64 / take as f64;
        let mut list: Vec<usize> = (0..take).map(|i| ((i as f64) * stepf) as usize).collect();
        list.dedup();
        for (j, &p) in list.iter().enumerate() {
            sel_host[rr * stride + j] = p as i32;
        }
        meta_host[2 * rr] = 1;
        meta_host[2 * rr + 1] = list.len() as i32;
        chosen.push(list);
    }
    let sel = DevBuf::from_slice(&sel_host).unwrap();
    let meta = DevBuf::from_slice(&meta_host).unwrap();
    let mut o_lat = DevBuf::alloc(rows * hs * LATENT_DIM * 2).unwrap();
    let mut ws = DevBuf::alloc(mla::attend_workspace_bytes(rows, gb_cuda::glm_idx::K_STRIDE)).unwrap();
    let s = SparseSel { indices: &sel, meta: &meta, stride };
    attend_rope_sparse(&q_lat, &qrot, &cache, &table, &ri, &s, rows, &mut o_lat, stride, &mut ws, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = o_lat.to_vec().unwrap();
    let mut hsh = 0xcbf2_9ce4_8422_2325u64;
    for b in got.iter().flat_map(|v| v.to_le_bytes()) {
        hsh = (hsh ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    eprintln!("sparse o_lat hash {hsh:016x}");
    // host reference for row 0, head 0 over its selection
    let ql: Vec<f32> = (0..LATENT_DIM).map(|c| dec(dt, got[0 * LATENT_DIM + c])).collect();
    let qlv: Vec<f32> = {
        let ql16 = q_lat.to_vec::<u16>().unwrap();
        (0..LATENT_DIM).map(|c| dec(dt, ql16[c])).collect()
    };
    let qrv: Vec<f32> = {
        let qr16 = qrot.to_vec::<u16>().unwrap();
        (0..ROPE_DIM).map(|d| dec(dt, qr16[d])).collect()
    };
    let hh = ROPE_DIM / 2;
    let mut krot = vec![vec![0f32; ROPE_DIM]; max_pos];
    for j in 0..max_pos { krot[j] = rope_rotate(&rope_f[j][..], &cos_h[j * hh..(j + 1) * hh], &sin_h[j * hh..(j + 1) * hh]); }
    let list = &chosen[0];
    let mut mx = f32::NEG_INFINITY;
    let mut sc = vec![0f32; list.len()];
    for (i, &j) in list.iter().enumerate() {
        let mut a = 0.0f32;
        for c in 0..LATENT_DIM { a += qlv[c] * lat_f[j][c]; }
        for d in 0..ROPE_DIM { a += qrv[d] * krot[j][d]; }
        sc[i] = a / 16.0;
        mx = mx.max(sc[i]);
    }
    let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
    let z: f32 = e.iter().sum();
    let mut worst = 0f32;
    for c in (0..LATENT_DIM).step_by(29) {
        let mut acc = 0.0f32;
        for (i, &j) in list.iter().enumerate() { acc += e[i] / z * lat_f[j][c]; }
        let g = dec(dt, got[c]);
        worst = worst.max((g - acc).abs() / (acc.abs() + 1.0));
        assert!((g - acc).abs() <= 5e-2 + 5e-2 * acc.abs(), "sparse o_lat[{c}] {g} vs {acc}");
    }
    eprintln!("sparse worst rel {worst:.4}; ql[0] {:.4}", ql[0]);
}
