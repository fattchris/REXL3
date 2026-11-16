//! The full model's DSA indexer kernels against a host reference: key LayerNorm+rope, query rope, score, top-k.
//!     cargo test --release -p gb-cuda --test glm_idx -- --test-threads=1
#![cfg(not(gb_no_cuda))]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_idx::{self, D, H, ROPE, TOPK};
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

#[test]
fn the_indexer_kernels_match_the_host_reference() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let dt = Dtype::F16;
    let rows_n = 2usize;
    let pos = [0i32, 3];
    let max_pos = 4usize;
    let (cos_h, sin_h) = gb_cuda::mla::rope_tables(max_pos, 8_000_000.0);
    let cos = DevBuf::from_slice(&cos_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let sin = DevBuf::from_slice(&sin_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let mut r = Rng(0x5EED_1D);

    // kraw, the LayerNorm weights, the q and w inputs
    let kraw_h: Vec<u16> = (0..rows_n * D).map(|_| enc(dt, r.uniform(-2.0, 2.0))).collect();
    let lnw_h: Vec<u16> = (0..D).map(|_| enc(dt, r.uniform(0.5, 1.5))).collect();
    let lnb_h: Vec<u16> = (0..D).map(|_| enc(dt, r.uniform(-0.2, 0.2))).collect();
    let q_h: Vec<u16> = (0..rows_n * H * D).map(|_| enc(dt, r.uniform(-1.0, 1.0))).collect();
    let w_h: Vec<u16> = (0..rows_n * H).map(|_| enc(dt, r.uniform(0.1, 1.0))).collect();
    let kraw = DevBuf::from_slice(&kraw_h).unwrap();
    let lnw = DevBuf::from_slice(&lnw_h).unwrap();
    let lnb = DevBuf::from_slice(&lnb_h).unwrap();
    let q = DevBuf::from_slice(&q_h).unwrap();
    let w = DevBuf::from_slice(&w_h).unwrap();

    // one slot, page 0; both rows at positions 0 and 3
    let table = BlockTable::from_host(&[vec![0u32, 1, 0, 0]], 4).unwrap();
    let ri = RowIndex::from_host(&[0i32, 0], &pos).unwrap();
    let mut keys = DevBuf::alloc(2 * glm_idx::PAGE * D * 2).unwrap();
    glm_idx::key(&kraw, D, &lnw, &lnb, &cos, &sin, ROPE / 2, &ri, &table, &mut keys, dt, rows_n, &st).unwrap();
    let mut qout = DevBuf::alloc(rows_n * H * D * 2).unwrap();
    glm_idx::query_rope(&q, H * D, &cos, &sin, ROPE / 2, &ri, &mut qout, rows_n, &st).unwrap();
    let mut scores = DevBuf::alloc(rows_n * max_pos * 4).unwrap();
    glm_idx::score(&qout, H * D, &w, &keys, &table, &ri, max_pos, max_pos, &mut scores, rows_n, &st).unwrap();
    let mut sel = DevBuf::alloc(rows_n * glm_idx::K_STRIDE * 4).unwrap();
    let mut meta = DevBuf::alloc(rows_n * 2 * 4).unwrap();
    glm_idx::select(&scores, max_pos, &ri, 2, &mut sel, &mut meta, rows_n, &st).unwrap();
    st.sync().unwrap();

    let keys_out: Vec<u16> = keys.to_vec().unwrap();
    let qout_v: Vec<u16> = qout.to_vec().unwrap();
    let scores_v: Vec<f32> = scores.to_vec().unwrap();
    let sel_v: Vec<i32> = sel.to_vec().unwrap();
    let meta_v: Vec<i32> = meta.to_vec().unwrap();

    // host reference
    let mut keys_ref = vec![0f32; 2 * glm_idx::PAGE * D];
    let mut q_ref = vec![vec![0f32; H * D]; rows_n];
    for row in 0..rows_n {
        let x: Vec<f32> = (0..D).map(|d| dec(dt, kraw_h[row * D + d])).collect();
        let mean: f32 = x.iter().sum::<f32>() / D as f32;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D as f32;
        let mut y: Vec<f32> = (0..D)
            .map(|d| (x[d] - mean) / (var + glm_idx::LN_EPS).sqrt() * dec(dt, lnw_h[d]) + dec(dt, lnb_h[d]))
            .collect();
        let p = pos[row] as usize;
        let hh = ROPE / 2;
        for i in 0..hh {
            let (x0, x1) = (y[2 * i], y[2 * i + 1]);
            let c = cos_h[p * hh + i];
            let s = sin_h[p * hh + i];
            y[i] = x0 * c - x1 * s;
            y[i + hh] = x0 * s + x1 * c;
        }
        let base = (0 * glm_idx::PAGE + p) * D;
        keys_ref[base..base + D].copy_from_slice(&y);
        for h in 0..H {
            let src: Vec<f32> = (0..D).map(|d| dec(dt, q_h[row * H * D + h * D + d])).collect();
            let mut out = src.clone();
            for i in 0..hh {
                let (x0, x1) = (src[2 * i], src[2 * i + 1]);
                let c = cos_h[p * hh + i];
                let s = sin_h[p * hh + i];
                out[i] = x0 * c - x1 * s;
                out[i + hh] = x0 * s + x1 * c;
            }
            q_ref[row][h * D..(h + 1) * D].copy_from_slice(&out);
        }
        // score
        for t in 0..=p {
            let mut total = 0.0f32;
            for h in 0..H {
                let mut dot = 0.0f32;
                for d in 0..D {
                    dot += q_ref[row][h * D + d] * keys_ref[base + d];
                }
                total += (dot * (D as f32).sqrt().recip()).max(0.0) * dec(dt, w_h[row * H + h]);
            }
            total *= (H as f32).sqrt().recip();
            let got = scores_v[row * max_pos + t];
            assert!((got - total).abs() < 5e-3 + 5e-3 * total.abs(), "score[{row}][{t}] {got} vs {total}");
        }
        // key / q outputs
        for d in (0..D).step_by(11) {
            let got = dec(dt, keys_out[base + d]);
            assert!((got - keys_ref[base + d]).abs() < 2e-2 + 2e-2 * keys_ref[base + d].abs(), "key[{row}][{d}]");
            let gotq = dec(dt, qout_v[row * H * D + d]);
            assert!((gotq - q_ref[row][d]).abs() < 5e-3 + 5e-3 * q_ref[row][d].abs(), "q[{row}][{d}]");
        }
    }
    // selection: row 1 (pos 3) selects 2 of 4 keys by (score desc, lower position), ascending
    assert_eq!(meta_v[2], 1);
    assert_eq!(meta_v[3], 2);
    let mut idx: Vec<(f32, usize)> = (0..=pos[1] as usize).map(|t| (scores_v[max_pos + t], t)).collect();
    idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let mut want: Vec<i32> = idx.iter().take(2).map(|x| x.1 as i32).collect();
    want.sort();
    let got: Vec<i32> = sel_v[glm_idx::K_STRIDE..glm_idx::K_STRIDE + 2].to_vec();
    assert_eq!(got, want, "selection");
    assert_eq!(sel_v[glm_idx::K_STRIDE + 2], -1);
    // row 0 (pos 0) selects 1 key
    assert_eq!(meta_v[1], 1);
    assert_eq!(sel_v[0], 0);
}
