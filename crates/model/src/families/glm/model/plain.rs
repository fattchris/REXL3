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
                mixer_rope(cfg, w, idx, &mut latent[ord], s, table, &ri, n, max_pos, cos, sin, st)?;
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
            }
            gb_cuda::glm_mtp::rmsnorm_f32_f16(&x, &dev.final_norm, &mut s.hidden, n, h, RMS_EPS, st)?;
            match head {
                Head::None => {}
                Head::LastRow => {
                    let hv = s.hidden.subview((n - 1) * h * 2, h * 2);
                    dev.lm_head.forward(&hv, &mut s.logits, 1, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
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
    w.o.forward(&s.mla.o_lat, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st)?;
    Ok(())
}
