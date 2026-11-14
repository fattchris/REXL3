//! A host (CPU, f64) reference of GLM-5.3 full's *layer 0* for the first few tokens of a prompt: the embedding, the MLA query
//! path (q_a -> q_a_norm -> q_b, the W_UK absorb, the rope tail), the latent write, and the absorbed attention over the
//! latents. The pack's EXL3 groups are decoded with `gb_exl3::decode_weight`, so nothing here touches the GPU path it checks.
//!
//!     REXL3_GLM_DIR=<pack> REXL3_LAYER0_IDS=154822,154824,... cargo test --release -p gb-model --test glm_layer0_host -- --nocapture
//!
//! Writes `<REXL3_LAYER0_OUT or /tmp/host_layer0.bin>`: `take u32, heads u32`, then `take * heads * 512` f32 of the attention
//! output (the latent per head, before W_UV), then `take * 512` f32 of the cached latents, so a run of the engine with
//! `REXL3_GLM_INV_DUMP` can be diffed against it.
#![allow(clippy::needless_range_loop)]

use gb_cuda::mla;
use gb_exl3::checkpoint::scan_indexed;
use gb_exl3::{codebook, decode_weight};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

fn f16v(h: u16) -> f64 {
    gb_exl3::f16::f16_to_f64(h)
}
fn bf16v(b: u16) -> f64 {
    gb_cuda::num::bf16_to_f32(b) as f64
}
fn round_f16(x: f64) -> f64 {
    f16v(gb_exl3::f16::f64_to_f16(x))
}

/// One `[k, n]` EXL3 group as a dense f64 matrix, row-major `w[k * n + out]` (`decode_weight`'s own layout).
fn group(ck: &gb_exl3::checkpoint::Checkpoint, prefix: &str) -> (Vec<f64>, usize, usize) {
    let g = ck.load_group(prefix).expect("group");
    let (k, n) = (g.group.k, g.group.n);
    let table = codebook(g.group.codebook);
    let w = decode_weight(&g.trellis, g.kt, g.nt, g.group.bits, &table, &g.suh, &g.svh);
    assert_eq!(w.len(), k * n, "{prefix}: {} values for {k}x{n}", w.len());
    (w, k, n)
}

/// A plain 16-bit tensor as f64 values.
fn plain16(ck: &gb_exl3::checkpoint::Checkpoint, name: &str, bf: bool) -> Vec<f64> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("no plain tensor {name}"));
    let f = if bf { bf16v as fn(u16) -> f64 } else { f16v };
    ck.read_le16(e, f).expect("read")
}

/// Rows `ids` of the embedding, each `hidden` wide, as f64 (read from the shard at the row's offset).
fn embed_rows(ck: &gb_exl3::checkpoint::Checkpoint, name: &str, hidden: usize, ids: &[u32]) -> Vec<Vec<f64>> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("no plain tensor {name}"));
    let (path, start) = ck.tensor_location(e).expect("location");
    let bf = e.dtype == "BF16";
    let mut f = std::fs::File::open(&path).expect("open shard");
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        f.seek(SeekFrom::Start(start + id as u64 * hidden as u64 * 2)).expect("seek");
        let mut buf = vec![0u8; hidden * 2];
        f.read_exact(&mut buf).expect("read row");
        out.push(buf.chunks_exact(2).map(|c| {
            let u = u16::from_le_bytes([c[0], c[1]]);
            if bf { bf16v(u) } else { f16v(u) }
        }).collect());
    }
    out
}

fn rmsnorm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len();
    let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let r = 1.0 / (ms + eps).sqrt();
    (0..n).map(|i| round_f16(w[i] * (x[i] * r))).collect()
}

/// The rope pair rotation of the reference: pairs (2i, 2i+1), written half-split ([i], [h + i]).
fn rope_rotate(x: &[f64], cos: &[f32], sin: &[f32]) -> Vec<f64> {
    let h = mla::ROPE_DIM / 2;
    let mut y = vec![0.0f64; mla::ROPE_DIM];
    for i in 0..h {
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        let (c, s) = (cos[i] as f64, sin[i] as f64);
        y[i] = round_f16(a * c - b * s);
        y[i + h] = round_f16(a * s + b * c);
    }
    y
}

#[test]
fn layer0_host_reference() {
    let Some(dir) = std::env::var("REXL3_GLM_DIR").ok().map(PathBuf::from) else {
        eprintln!("REXL3_GLM_DIR is not set: not run");
        return;
    };
    let ids: Vec<u32> = std::env::var("REXL3_LAYER0_IDS")
        .expect("REXL3_LAYER0_IDS (comma separated)")
        .split(',')
        .map(|s| s.trim().parse().expect("id"))
        .collect();
    let take = ids.len().min(4);
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let theta = cfg["rope_parameters"]["rope_theta"].as_f64().expect("rope_parameters.rope_theta") as f32;
    let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let hidden = cfg["hidden_size"].as_u64().unwrap() as usize; // 6144
    let qk = 256usize; // qk_head_dim: nope 192 + rope 64
    let nope = 192usize;
    let lat = 512usize;
    let heads = 64usize;
    let ck = scan_indexed(&dir, false).expect("scan");
    eprintln!("config: hidden {hidden}, heads {heads}, theta {theta}, eps {eps}");

    let in_norm = plain16(&ck, "model.layers.0.input_layernorm.weight", true);
    let qa_norm = plain16(&ck, "model.layers.0.self_attn.q_a_layernorm.weight", true);
    let kva_norm = plain16(&ck, "model.layers.0.self_attn.kv_a_layernorm.weight", true);
    let emb = embed_rows(&ck, "model.embed_tokens.weight", hidden, &ids[..take]);
    let (qa, ka, na) = group(&ck, "model.layers.0.self_attn.q_a_proj");
    let (qb, kb, nb) = group(&ck, "model.layers.0.self_attn.q_b_proj");
    let (kva, kk, nkva) = group(&ck, "model.layers.0.self_attn.kv_a_proj_with_mqa");
    assert_eq!((ka, kb, kk), (hidden, 2048, hidden), "layer-0 group input widths");
    assert_eq!(na, 2048);
    assert_eq!(nb, heads * qk);
    assert!(nkva >= lat + mla::ROPE_DIM, "kv_a output {nkva} holds latent + rope");
    let kvb = plain16(&ck, "model.layers.0.self_attn.kv_b_proj.weight", false); // fp16 in the pack: [heads * 448, 512]
    assert_eq!(kvb.len(), heads * (nope + 256) * lat);

    let (cos_t, sin_t) = mla::rope_tables(take + 1, theta);
    let hh = mla::ROPE_DIM / 2;
    let mut latents: Vec<Vec<f64>> = Vec::new();
    let mut q_lat: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut q_rot: Vec<Vec<Vec<f64>>> = Vec::new();
    for (r, row) in emb.iter().enumerate() {
        let hdn = rmsnorm(row, &in_norm, eps);
        eprintln!("row {r}: embed norm {:.4} hidden norm {:.4}", (row.iter().map(|v| v*v).sum::<f64>()).sqrt(), (hdn.iter().map(|v| v*v).sum::<f64>()).sqrt());
        let qa_out: Vec<f64> = (0..na).map(|o| (0..ka).map(|i| hdn[i] * qa[i * na + o]).sum()).collect();
        eprintln!("row {r}: qa norm {:.4}", (qa_out.iter().map(|v| v*v).sum::<f64>()).sqrt());
        let q_res = rmsnorm(&qa_out, &qa_norm, eps);
        let q: Vec<f64> = (0..nb).map(|o| (0..kb).map(|i| q_res[i] * qb[i * nb + o]).sum()).collect();
        eprintln!("row {r}: q_res norm {:.4} q norm {:.4}", (q_res.iter().map(|v| v*v).sum::<f64>()).sqrt(), (q.iter().map(|v| v*v).sum::<f64>()).sqrt());
        let kva_out: Vec<f64> = (0..nkva).map(|o| (0..kk).map(|i| hdn[i] * kva[i * nkva + o]).sum()).collect();
        // the latent and its rope key at position r
        let ckv = rmsnorm(&kva_out[..lat], &kva_norm, eps);
        latents.push(ckv.clone());
        let cos = &cos_t[r * hh..(r + 1) * hh];
        let sin = &sin_t[r * hh..(r + 1) * hh];
        let mut ql = vec![vec![0.0f64; lat]; heads];
        let mut qr = vec![vec![0.0f64; mla::ROPE_DIM]; heads];
        for h in 0..heads {
            // q_lat = q_nope . W_UK[h] (W_UK = kv_b rows h*448 + d, d < 192)
            for c in 0..lat {
                let mut acc = 0.0f64;
                for d in 0..nope {
                    acc += q[h * qk + d] * kvb[(h * (nope + 256) + d) * lat + c];
                }
                ql[h][c] = round_f16(acc);
            }
            let tail: Vec<f64> = (0..mla::ROPE_DIM).map(|d| q[h * qk + nope + d]).collect();
            qr[h] = rope_rotate(&tail, cos, sin);
        }
        q_lat.push(ql);
        q_rot.push(qr);
    }
    // the rotated rope keys of the cache (the engine writes them at layer 0's latent_write)
    let mut k_rot: Vec<Vec<f64>> = Vec::new();
    for r in 0..take {
        k_rot.push(vec![0.0; mla::ROPE_DIM]); // filled below from a fresh kv_a pass (kept simple: recomputed)
        let hdn = rmsnorm(&emb[r], &in_norm, eps);
        let kva_out: Vec<f64> = (0..nkva).map(|o| (0..kk).map(|i| hdn[i] * kva[i * nkva + o]).sum()).collect();
        let cos = &cos_t[r * hh..(r + 1) * hh];
        let sin = &sin_t[r * hh..(r + 1) * hh];
        let tail: Vec<f64> = (0..mla::ROPE_DIM).map(|d| kva_out[lat + d]).collect();
        k_rot[r] = rope_rotate(&tail, cos, sin);
    }
    // absorbed attention at each row over keys 0..=row
    let mut o: Vec<Vec<Vec<f64>>> = Vec::new();
    for r in 0..take {
        let mut oh = vec![vec![0.0f64; lat]; heads];
        for h in 0..heads {
            let mut s = Vec::with_capacity(r + 1);
            for j in 0..=r {
                let mut a = 0.0f64;
                for c in 0..lat {
                    a += q_lat[r][h][c] * latents[j][c];
                }
                let mut b = 0.0f64;
                for d in 0..mla::ROPE_DIM {
                    b += q_rot[r][h][d] * k_rot[j][d];
                }
                s.push((a + b) / 16.0);
            }
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..lat {
                let mut acc = 0.0f64;
                for j in 0..=r {
                    acc += e[j] / z * latents[j][c];
                }
                oh[h][c] = acc;
            }
        }
        o.push(oh);
    }
    // write: take, heads, o (f32), latents (f32)
    let out = std::env::var("REXL3_LAYER0_OUT").unwrap_or_else(|_| "/tmp/host_layer0.bin".into());
    let mut f = std::fs::File::create(&out).unwrap();
    f.write_all(&(take as u32).to_le_bytes()).unwrap();
    f.write_all(&(heads as u32).to_le_bytes()).unwrap();
    for r in 0..take {
        for h in 0..heads {
            for c in 0..lat {
                f.write_all(&(o[r][h][c] as f32).to_le_bytes()).unwrap();
            }
        }
    }
    for r in 0..take {
        for c in 0..lat {
            f.write_all(&(latents[r][c] as f32).to_le_bytes()).unwrap();
        }
    }
    // q_lat and q_rot, contiguous over rows like the engine's dump
    for r in 0..take {
        for h in 0..heads {
            for c in 0..lat {
                f.write_all(&(q_lat[r][h][c] as f32).to_le_bytes()).unwrap();
            }
        }
    }
    for r in 0..take {
        for h in 0..heads {
            for d in 0..mla::ROPE_DIM {
                f.write_all(&(q_rot[r][h][d] as f32).to_le_bytes()).unwrap();
            }
        }
    }
    // the chain's intermediates, in the engine dump's order: every row's q_res, then q, then hidden, then q_a
    let mut chain: Vec<Vec<f64>> = Vec::new();
    for r in 0..take {
        let hdn = rmsnorm(&emb[r], &in_norm, eps);
        let qa_out: Vec<f64> = (0..na).map(|o| (0..ka).map(|i| hdn[i] * qa[i * na + o]).sum()).collect();
        let q_res = rmsnorm(&qa_out, &qa_norm, eps);
        let q: Vec<f64> = (0..nb).map(|o| (0..kb).map(|i| q_res[i] * qb[i * nb + o]).sum()).collect();
        chain.push(q_res);
        chain.push(q);
        chain.push(hdn);
        chain.push(qa_out);
    }
    let (nres, nq, nhid) = (take * na, take * nb, take * hidden);
    let mut at = 0usize;
    for block in chain.iter().take(take) { let _ = (block, at); }
    for r in 0..take { for v in &chain[r * 4] { f.write_all(&(*v as f32).to_le_bytes()).unwrap(); } }
    for r in 0..take { for v in &chain[r * 4 + 1] { f.write_all(&(*v as f32).to_le_bytes()).unwrap(); } }
    for r in 0..take { for v in &chain[r * 4 + 2] { f.write_all(&(*v as f32).to_le_bytes()).unwrap(); } }
    for r in 0..take { for v in &chain[r * 4 + 3] { f.write_all(&(*v as f32).to_le_bytes()).unwrap(); } }
    let _ = (nres, nq, nhid);
    eprintln!("wrote {out}: take {take}, heads {heads}");
    eprintln!("o[0][0][..4] {:?}", &o[0][0][..4]);
    eprintln!("latent[0][..4] {:?}", &latents[0][..4]);
}
