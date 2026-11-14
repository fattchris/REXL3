//! The full GLM-5.3 trunk (`glm_moe_dsa`): plain residuals, MLA with the 64-wide rotary tail, no KDA and no mHC. Every layer is the NextN
//! layer's sequence (`model/mtp.rs`): norm -> mixer -> add, norm -> feed-forward -> add; the head is the final norm then `lm_head`.
//!
//! Dense attention only for now: rows at position >= [`gb_cuda::mla::DENSE_MAX_KEYS`] need the full model's DSA selection (per-token key
//! cache, shared selections), which is not ported yet, so such a call is refused by name.

use super::*;
use crate::families::glm::device::{FfnDevice, FullIndexerDevice, MixerDevice};
use crate::families::glm::model::layers;
use gb_cuda::mla::{self, LatentCache};

/// Row stride of the full model's selection indices (`gb_cuda::glm_idx::K_STRIDE`).
pub(super) const K_STRIDE: usize = gb_cuda::glm_idx::K_STRIDE;

impl GlmModel {
    /// A config whose trunk joins through plain residuals (the full model): the mHC op schedule, KDA and the window commits do not apply.
    pub(super) fn plain_trunk(&self) -> bool {
        self.cfg.layers.iter().any(|l| l.residual == crate::family::Residual::Plain)
    }

    pub(super) fn forward_plain(&mut self, b: &RowBatch, head: Head, max_pos: usize) -> Result<(), ModelError> {
        let GlmModel { cfg, dev, stream, s, latent, mla_ord, table, cos, sin, sig, idx_keys, tp, own_s, .. } = self;
        let st = &*stream;
        let h = cfg.hidden;
        let rows = b.rows;
        let cos = cos.as_ref().ok_or_else(|| ModelError::Config("the rotary tables are missing".into()))?;
        let sin = sin.as_ref().ok_or_else(|| ModelError::Config("the rotary tables are missing".into()))?;
        let mut x = DevBuf::alloc(alloc::MAX_ROWS * h * 4)?;
        let mut a = 0usize;
        while a < rows {
            let n = (rows - a).min(alloc::MAX_ROWS);
            let ri = layers::slab_index(&s.rowidx, a, n);
            let ids = s.ids.subview(a * 4, n * 4);
            gb_cuda::glm_mtp::embed_f32(&dev.embed, &ids, &mut x, n, h, cfg.vocab, st)?;
            if let Ok(path) = std::env::var("REXL3_GLM_TAP_DUMP") {
                // debug: the embedding itself (layer tag 0xFFFFFFFF), so a tap comparison starts below layer 0
                st.sync()?;
                let mut v = vec![0f32; n * h];
                x.download(&mut v)?;
                use std::io::Write;
                let pos0 = b.positions.get(a).copied().unwrap_or(a as i32) as u32;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = f.write_all(&u32::MAX.to_le_bytes());
                    let _ = f.write_all(&pos0.to_le_bytes());
                    let _ = f.write_all(&((n as u32).to_le_bytes()));
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = f.write_all(&bytes);
                }
            }
            for (l, layer) in dev.layers.iter().enumerate() {
                gb_cuda::glm_mtp::rmsnorm_f32_f16(&x, &layer.input_norm, &mut s.hidden, n, h, RMS_EPS, st)?;
                let MixerDevice::Mla(w) = &layer.mixer else {
                    return Err(ModelError::Config("the full model's trunk has MLA layers only".into()));
                };
                let ord = mla_ord[l];
                let idx = match (&w.full_indexer, &mut idx_keys[ord]) {
                    (Some(ix), Some(plane)) => Some((ix, &mut *plane)),
                    _ => None,
                };
                mixer_rope(cfg, w, idx, &mut latent[ord], s, table, &ri, n, max_pos, cos, sin, l, st)?;
                if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() {
                    // debug: the mixer's own output, so a layer divergence splits into the mixer and the FFN
                    st.sync()?;
                    let mut v = vec![0f32; n * h];
                    s.mixer_out.download(&mut v)?;
                    let mut hsh = 0xcbf2_9ce4_8422_2325u64;
                    for f in &v[(n - 1) * h..n * h] {
                        for b in f.to_le_bytes() {
                            hsh = (hsh ^ b as u64).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                    eprintln!("rexl3 event=mix_dump layer={l} a={a} n={n} rows={rows} hash={hsh:016x}");
                }
                gb_cuda::glm_mtp::add_rmsnorm_f32(&mut x, &s.mixer_out, &layer.post_norm, &mut s.hidden, n, h, RMS_EPS, st)?;
                match &layer.ffn {
                    FfnDevice::Dense(m) => {
                        let sc = s.dense.as_mut().ok_or_else(|| ModelError::Config("no dense scratch".into()))?;
                        m.forward(&s.hidden, Dtype::F16, n, &mut s.mixer_out, sc, st)?;
                    }
                    FfnDevice::Moe { layer: ml, shared } => {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        shared.forward(&s.hidden, Dtype::F16, n, &mut s.moe_shared_out, sc, st)?;
                        ml.route(&s.hidden, Dtype::F16, n, sig, &mut s.moe, st)?;
                        match tp.as_mut() {
                            // a rank of the group: its owned experts, the exchange, the combine (the shared expert is replicated and added once)
                            Some(t) => t.experts(ml, &s.hidden, n, &mut s.moe, Some(&s.moe_shared_out), &mut s.mixer_out, st, &mut || Ok(()))?,
                            None => ml.experts_picked(
                                &s.hidden,
                                Dtype::F16,
                                n,
                                Some(&s.moe_shared_out),
                                &mut s.mixer_out,
                                &mut s.moe,
                                st,
                            )?,
                        }
                    }
                    FfnDevice::MoeSubbanks { layer: ml, shared } => {
                        let sc = s.shared.as_mut().ok_or_else(|| ModelError::Config("no shared-expert scratch".into()))?;
                        shared.forward(&s.hidden, Dtype::F16, n, &mut s.moe_shared_out, sc, st)?;
                        ml.route(&s.hidden, Dtype::F16, n, sig, &mut s.moe, st)?;
                        match tp.as_mut() {
                            Some(t) => t.subbanks(ml, &s.hidden, n, &mut s.moe, Some(&s.moe_shared_out), &mut s.mixer_out, st, &mut || Ok(()))?,
                            None => {
                                let os = own_s.as_mut().ok_or_else(|| ModelError::Config("no subbank scratch".into()))?;
                                ml.experts.run_pairs(&s.hidden, Dtype::F16, n, &mut s.moe, os, st)?;
                                let d = ml.dims;
                                moe::combine_pairs(
                                    &s.moe.y_pairs,
                                    &s.moe.pick,
                                    &s.moe.wts,
                                    Some(&s.moe_shared_out),
                                    &mut s.mixer_out,
                                    n,
                                    &d,
                                    st,
                                )?;
                            }
                        }
                    }
                }
                gb_cuda::glm_mtp::add_f32(&mut x, &s.mixer_out, n * h, st)?;
                if let Ok(path) = std::env::var("REXL3_GLM_TAP_DUMP") {
                    // debug: the residual stream after the taps TensorFold's capture records ([5,19,33,47,61,75]), so an oracle
                    // comparison can say which layer first diverges
                    if l <= 5 || matches!(l, 19 | 33 | 47 | 61 | 75) {
                        st.sync()?;
                        let mut v = vec![0f32; n * h];
                        x.download(&mut v)?;
                        use std::io::Write;
                        let pos0 = b.positions.get(a).copied().unwrap_or(a as i32) as u32;
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                            let _ = f.write_all(&((l as u32).to_le_bytes()));
                            let _ = f.write_all(&pos0.to_le_bytes());
                            let _ = f.write_all(&((n as u32).to_le_bytes()));
                            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            let _ = f.write_all(&bytes);
                        }
                    }
                }
                if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() {
                    // debug: a digest of the last row's residual after each layer, so a rank divergence names its layer
                    let mut v = vec![0f32; n * h];
                    st.sync()?;
                    x.download(&mut v)?;
                    let mut hsh = 0xcbf2_9ce4_8422_2325u64;
                    for f in &v[(n - 1) * h..n * h] {
                        for b in f.to_le_bytes() {
                            hsh = (hsh ^ b as u64).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                    eprintln!("rexl3 event=layer_dump layer={l} a={a} n={n} rows={rows} hash={hsh:016x}");
                }
            }
            gb_cuda::glm_mtp::rmsnorm_f32_f16(&x, &dev.final_norm, &mut s.hidden, n, h, RMS_EPS, st)?;
            match head {
                Head::None => {}
                Head::LastRow => {
                    let hv = s.hidden.subview((n - 1) * h * 2, h * 2);
                    if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() {
                        // debug: the head's input (the final norm's last row) and its output, so a candidate difference is placed
                        st.sync()?;
                        let mut v = vec![0u8; h * 2];
                        hv.download(&mut v)?;
                        let hh = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ *b as u64).wrapping_mul(0x100_0000_01b3));
                        eprintln!("rexl3 event=hnorm layer=head rows=1 hash={hh:016x}");
                    }
                    dev.lm_head.forward(&hv, &mut s.logits, 1, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                    if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() {
                        st.sync()?;
                        let mut v = vec![0u8; cfg.vocab * 2];
                        s.logits.download(&mut v)?;
                        let hh = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ *b as u64).wrapping_mul(0x100_0000_01b3));
                        eprintln!("rexl3 event=hlogits layer=head rows=1 hash={hh:016x}");
                    }
                }
                Head::AllRows => {
                    dev.lm_head.forward(&s.hidden, &mut s.logits, n, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
                }
            }
            a += n;
        }
        Ok(())
    }
}

/// One MLA layer of the full model: q_a -> norm -> q_b; kv_a -> the latent norm and the rope-key rotation into both cache planes;
/// on a layer that owns the indexer, the per-token keys (LayerNorm + rope) and this row's top-k selection; the absorbed and rotated
/// query; attention (dense rows below 2,048, the selection above); o_proj. Writes `s.mixer_out`.
#[allow(clippy::too_many_arguments)]
fn mixer_rope(
    cfg: &GlmConfig,
    w: &crate::families::glm::device::MlaDevice,
    idx: Option<(&FullIndexerDevice, &mut DevBuf)>,
    latent: &mut LatentCache,
    s: &mut Scratch,
    table: &BlockTable,
    ri: &RowIndex,
    rows: usize,
    max_pos: usize,
    cos: &DevBuf,
    sin: &DevBuf,
    layer: usize,
    st: &Stream,
) -> Result<(), ModelError> {
    let h = cfg.hidden;
    let m = &cfg.mla;
    let qk = m.qk_head_dim();
    let kv_n = (m.kv_lora_rank + m.qk_rope_head_dim).next_multiple_of(128);
    let sparse = max_pos + 1 > mla::DENSE_MAX_KEYS;
    let stride = max_pos + 1;
    w.q_a.forward(&s.hidden, &mut s.q_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    gb_cuda::glm_glue::rmsnorm_f16(&s.q_a, &w.q_a_norm, &mut s.q_resid, rows, m.q_lora_rank, RMS_EPS, st)?;
    w.q_b.forward(&s.q_resid, &mut s.q, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    w.kv_a.forward(&s.hidden, &mut s.kv_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    gb_cuda::glm_glue::rmsnorm_f16_strided(&s.kv_a, kv_n, &w.kv_a_norm, &mut s.ckv, rows, m.kv_lora_rank, RMS_EPS, st)?;
    mla::latent_write_rope(&s.ckv, &s.kv_a, kv_n, m.kv_lora_rank, cos, sin, mla::ROPE_DIM / 2, ri, table, latent, rows, st)?;
    if std::env::var_os("REXL3_GLM_CACHE_DUMP").is_some() {
        // debug: what this layer's cache holds at the rows' own positions (the attention's other input, per rank)
        let mut slot = vec![0i32; rows];
        let mut pos = vec![0i32; rows];
        let mut tbl = vec![0u32; table.slots * table.max_blocks];
        ri.slot.download(&mut slot)?;
        ri.pos.download(&mut pos)?;
        st.sync()?;
        table.buf.download(&mut tbl)?;
        st.sync()?;
        let mut hs: Vec<String> = Vec::with_capacity(rows);
        for r in 0..rows {
            let p = pos[r] as usize;
            let page = tbl[slot[r] as usize * table.max_blocks + p / mla::BLOCK_TOKENS] as usize;
            let at = (page * mla::BLOCK_TOKENS + p % mla::BLOCK_TOKENS) as usize;
            let mut v = vec![0u8; mla::LATENT_DIM * 2];
            latent.lat.download_range(at * mla::LATENT_DIM * 2, &mut v)?;
            let mut hsh = 0xcbf2_9ce4_8422_2325u64;
            for b in &v {
                hsh = (hsh ^ *b as u64).wrapping_mul(0x100_0000_01b3);
            }
            if let Some(rot) = latent.rot.as_ref() {
                let mut rv = vec![0u8; mla::ROPE_DIM * 2];
                rot.download_range(at * mla::ROPE_DIM * 2, &mut rv)?;
                for b in &rv {
                    hsh = (hsh ^ *b as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
            hs.push(format!("{p}:{hsh:016x}"));
        }
        let mut hh = 0xcbf2_9ce4_8422_2325u64;
        for v in slot.iter().chain(pos.iter()) {
            for b in v.to_le_bytes() {
                hh = (hh ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        for v in tbl.iter() {
            for b in v.to_le_bytes() {
                hh = (hh ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
        eprintln!("rexl3 event=cache_dump layer={layer} rows={rows} ri_table={hh:016x} c={}", hs.join(","));
    }
    // the indexer: this layer's keys for every row, and the selection when the layer computes it (shared layers reuse the buffer)
    if sparse {
        if let Some((ix, plane)) = idx {
            use gb_cuda::glm_idx;
            moe::linear_f16(&s.hidden, Dtype::F16, &ix.wk, &mut s.k_raw, rows, h, glm_idx::D, st)?;
            glm_idx::key(&s.k_raw, glm_idx::D, &ix.ln_w, &ix.ln_b, cos, sin, mla::ROPE_DIM / 2, ri, table, plane, Dtype::F16, rows, st)?;
            moe::linear_f16(&s.hidden, Dtype::F16, &ix.weights_proj, &mut s.wproj, rows, h, glm_idx::H, st)?;
            ix.wq_b.forward(&s.q_resid, &mut s.iq, rows, &mut s.ws, st, false)?;
            glm_idx::query_rope(&s.iq, glm_idx::H * glm_idx::D, cos, sin, mla::ROPE_DIM / 2, ri, &mut s.idx_q, rows, st)?;
            glm_idx::score(&s.idx_q, glm_idx::H * glm_idx::D, &s.wproj, plane, table, ri, stride, stride, &mut s.idx_scores, rows, st)?;
            glm_idx::select(&s.idx_scores, stride, ri, glm_idx::TOPK, &mut s.idx_sel, &mut s.idx_meta, rows, st)?;
        }
    }
    mla::absorb_rope(
        &s.q,
        m.heads * qk,
        qk,
        m.qk_nope_head_dim,
        &w.kv_b,
        &mut s.mla.q_lat,
        &mut s.qrot,
        cos,
        sin,
        mla::ROPE_DIM / 2,
        ri,
        rows,
        st,
    )?;
    if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() && layer == 0 {
        // debug: what the attention workspace holds BEFORE its first write (a stale read would depend on it, and it should be zeros)
        st.sync()?;
        let n = s.mla.ws.bytes().min(1 << 20);
        let mut v = vec![0u8; n];
        s.mla.ws.download_range(0, &mut v)?;
        let hsh = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ *b as u64).wrapping_mul(0x100_0000_01b3));
        let nz = v.iter().filter(|b| **b != 0).count();
        eprintln!("rexl3 event=ws_before layer={layer} bytes={} head_hash={hsh:016x} nonzero={nz}", s.mla.ws.bytes());
    }
    if sparse {
        // the dense rows read at most the dense limit; the sparse rows read their selection list (<= K_STRIDE entries)
        let dense = stride.min(mla::DENSE_MAX_KEYS);
        mla::attend_rope(&s.mla.q_lat, &s.qrot, latent, table, ri, rows, &mut s.mla.o_lat, dense, dense, &mut s.mla.ws, st)?;
        let sel = mla::SparseSel { indices: &s.idx_sel, meta: &s.idx_meta, stride: K_STRIDE };
        mla::attend_rope_sparse(&s.mla.q_lat, &s.qrot, latent, table, ri, &sel, rows, &mut s.mla.o_lat, K_STRIDE, &mut s.mla.ws, st)?;
    } else {
        let dense = stride.min(mla::DENSE_MAX_KEYS);
        mla::attend_rope(&s.mla.q_lat, &s.qrot, latent, table, ri, rows, &mut s.mla.o_lat, dense, dense, &mut s.mla.ws, st)?;
    }
    if std::env::var_os("REXL3_GLM_LAYER_DUMP").is_some() {
        // debug: the mixer's two middle tensors, so a layer difference splits into the query/absorb and the attention
        let mut d = |name: &str, b: &DevBuf| -> Result<(), ModelError> {
            st.sync()?;
            let mut v = vec![0u8; rows * m.kv_lora_rank * 2];
            b.download(&mut v)?;
            let mut hsh = 0xcbf2_9ce4_8422_2325u64;
            for x in &v {
                hsh = (hsh ^ *x as u64).wrapping_mul(0x100_0000_01b3);
            }
            eprintln!("rexl3 event={name} layer={layer} rows={rows} hash={hsh:016x}");
            Ok(())
        };
        d("qlat_dump", &s.mla.q_lat)?;
        st.sync()?;
        {
            // the rotated query tail, per head: the attention's other input
            let heads = cfg.mla.heads;
            let mut v = vec![0u8; rows * heads * mla::ROPE_DIM * 2];
            s.qrot.download(&mut v)?;
            let mut hs: Vec<String> = Vec::with_capacity(heads);
            for hd in 0..heads {
                let mut hsh = 0xcbf2_9ce4_8422_2325u64;
                for r in 0..rows {
                    let off = (r * heads + hd) * mla::ROPE_DIM * 2;
                    for b in &v[off..off + mla::ROPE_DIM * 2] {
                        hsh = (hsh ^ *b as u64).wrapping_mul(0x100_0000_01b3);
                    }
                }
                hs.push(format!("{hsh:016x}"));
            }
            eprintln!("rexl3 event=qrot layer={layer} rows={rows} h={}", hs.join(","));
        }
        st.sync()?;
        {
            // every head's slice of the attention output, so a difference names the head (o_lat is rows x heads x kv_lora)
            let heads = cfg.mla.heads;
            let head_count = heads;
            let dc = m.kv_lora_rank;
            let mut v = vec![0u8; rows * heads * dc * 2];
            s.mla.o_lat.download(&mut v)?;
            let mut hs: Vec<String> = Vec::with_capacity(heads);
            for hd in 0..heads {
                let mut hsh = 0xcbf2_9ce4_8422_2325u64;
                for r in 0..rows {
                    let off = (r * heads + hd) * dc * 2;
                    for b in &v[off..off + dc * 2] {
                        hsh = (hsh ^ *b as u64).wrapping_mul(0x100_0000_01b3);
                    }
                }
                hs.push(format!("{hsh:016x}"));
            }
            eprintln!("rexl3 event=oheads layer={layer} rows={rows} h={}", hs.join(","));
            if layer == 0 {
                // debug: the same attention again, into fresh buffers (a fresh workspace), so the two results can be compared
                // within one launch history: equal means the kernel is stable in-engine and the difference is an input
                let dense2 = (max_pos + 1).min(mla::DENSE_MAX_KEYS);
                let mut o2 = DevBuf::alloc(rows * heads * dc * 2)?;
                let mut ws2 = DevBuf::alloc(mla::attend_workspace_bytes(rows, dense2))?;
                mla::attend_rope(&s.mla.q_lat, &s.qrot, latent, table, ri, rows, &mut o2, dense2, dense2, &mut ws2, st)?;
                st.sync()?;
                let mut v2 = vec![0u8; rows * heads * dc * 2];
                o2.download(&mut v2)?;
                let h1 = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ *b as u64).wrapping_mul(0x100_0000_01b3));
                let h2 = v2.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ *b as u64).wrapping_mul(0x100_0000_01b3));
                eprintln!("rexl3 event=attend_again layer={layer} first={h1:016x} second={h2:016x} same={}", h1 == h2);
            }
            if layer == 0 {
                if let Ok(path) = std::env::var("REXL3_GLM_DUMP_FILE") {
                    // debug: the raw attention output of layer 0, appended, so two ranks' values can be diffed on the host
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                        let mut hdr = Vec::with_capacity(24);
                        hdr.extend_from_slice(&(rows as u64).to_le_bytes());
                        hdr.extend_from_slice(&(head_count as u64).to_le_bytes());
                        hdr.extend_from_slice(&(dc as u64).to_le_bytes());
                        let _ = std::io::Write::write_all(&mut f, &hdr);
                        let _ = std::io::Write::write_all(&mut f, &v);
                    }
                }
            }
        }
    }
    if std::env::var_os("REXL3_GLM_CACHE_ALL").is_some() && layer == 0 {
        // debug: the whole latent cache and rope plane after this call's writes (layer 0 only), so two ranks' caches can be
        // compared without assuming which positions the attention reads this call
        st.sync()?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let total = latent.lat.bytes();
        let step = 1 << 20;
        let mut off = 0usize;
        while off < total {
            let n = step.min(total - off);
            let mut v = vec![0u8; n];
            latent.lat.download_range(off, &mut v)?;
            for b in &v {
                h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
            }
            off += n;
        }
        if let Some(rot) = latent.rot.as_ref() {
            let total = rot.bytes();
            let mut off = 0usize;
            while off < total {
                let n = step.min(total - off);
                let mut v = vec![0u8; n];
                rot.download_range(off, &mut v)?;
                for b in &v {
                    h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
                }
                off += n;
            }
        }
        eprintln!("rexl3 event=cache_all layer={layer} bytes={total} hash={h:016x}");
    }
    if let Ok(path) = std::env::var("REXL3_GLM_INV_DUMP") {
        // debug: layer 0's attention output for the first rows, every head, with the rows' positions and the cached latents
        // those rows read: a host reference (from the pack's own weights) is compared against it
        if layer == 0 {
            st.sync()?;
            let mut slot = vec![0i32; rows];
            let mut pos = vec![0i32; rows];
            ri.slot.download(&mut slot)?;
            ri.pos.download(&mut pos)?;
            let mut tbl = vec![0u32; table.slots * table.max_blocks];
            table.buf.download(&mut tbl)?;
            st.sync()?;
            let heads = cfg.mla.heads;
            let dc = m.kv_lora_rank;
            let take = rows.min(4);
            let mut o = vec![0u8; take * heads * dc * 2];
            s.mla.o_lat.download_range(0, &mut o)?;
            let mut ql = vec![0u8; take * heads * dc * 2];
            s.mla.q_lat.download_range(0, &mut ql)?;
            let mut qr = vec![0u8; take * heads * mla::ROPE_DIM * 2];
            s.qrot.download_range(0, &mut qr)?;
            let qk = cfg.mla.qk_head_dim();
            let qres_w = cfg.mla.q_lora_rank;
            let mut qres = vec![0u8; take * qres_w * 2];
            s.q_resid.download_range(0, &mut qres)?;
            let mut qv = vec![0u8; take * heads * qk * 2];
            s.q.download_range(0, &mut qv)?;
            let mut hid = vec![0u8; take * h * 2];
            s.hidden.download_range(0, &mut hid)?;
            let mut qa_o = vec![0u8; take * qres_w * 2];
            s.q_a.download_range(0, &mut qa_o)?;
            let mut rec = Vec::new();
            rec.extend_from_slice(&(take as u32).to_le_bytes());
            rec.extend_from_slice(&(heads as u32).to_le_bytes());
            let mut lats = Vec::new();
            for r in 0..take {
                let p = pos[r] as usize;
                let page = tbl[slot[r] as usize * table.max_blocks + p / mla::BLOCK_TOKENS] as usize;
                let at = (page * mla::BLOCK_TOKENS + p % mla::BLOCK_TOKENS) as usize;
                let mut lat = vec![0u8; mla::LATENT_DIM * 2];
                latent.lat.download_range(at * mla::LATENT_DIM * 2, &mut lat)?;
                lats.extend_from_slice(&lat);
                eprintln!("rexl3 event=inv row={r} pos={p}");
            }
            rec.extend_from_slice(&o);
            rec.extend_from_slice(&lats);
            rec.extend_from_slice(&ql);
            rec.extend_from_slice(&qr);
            rec.extend_from_slice(&qres);
            rec.extend_from_slice(&qv);
            rec.extend_from_slice(&hid);
            rec.extend_from_slice(&qa_o);
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                use std::io::Write;
                let _ = f.write_all(&rec);
            }
        }
    }
    // the absorbed latent (kv_lora per head) expands through W_UV to v_head_dim per head: the o_proj's input
    mla::expand(&s.mla.o_lat, &w.kv_b, &mut s.mla_out, rows, st)?;
    w.o.forward(&s.mla_out, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st)?;
    Ok(())
}
