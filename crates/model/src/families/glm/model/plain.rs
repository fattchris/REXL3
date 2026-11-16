//! The full GLM-5.3 trunk (`glm_moe_dsa`): plain residuals, MLA with the 64-wide rotary tail, no KDA and no mHC. Every layer is the NextN
//! layer's sequence (`model/mtp.rs`): norm -> mixer -> add, norm -> feed-forward -> add; the head is the final norm then `lm_head`.
//!
//! Dense attention only for now: rows at position >= [`gb_cuda::mla::DENSE_MAX_KEYS`] need the full model's DSA selection (per-token key
//! cache, shared selections), which is not ported yet, so such a call is refused by name.

use super::*;
use crate::families::glm::device::{FfnDevice, MixerDevice};
use crate::families::glm::model::layers;
use gb_cuda::mla::{self, LatentCache};

impl GlmModel {
    /// A config whose trunk joins through plain residuals (the full model): the mHC op schedule, KDA and the window commits do not apply.
    pub(super) fn plain_trunk(&self) -> bool {
        self.cfg.layers.iter().any(|l| l.residual == crate::family::Residual::Plain)
    }

    pub(super) fn forward_plain(&mut self, b: &RowBatch, head: Head, max_pos: usize) -> Result<(), ModelError> {
        let GlmModel { cfg, dev, stream, s, latent, mla_ord, table, cos, sin, sig, .. } = self;
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
                mixer_rope(cfg, w, &mut latent[ord], s, table, &ri, n, max_pos, cos, sin, st)?;
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
                        ml.experts_picked(&s.hidden, Dtype::F16, n, Some(&s.moe_shared_out), &mut s.mixer_out, &mut s.moe, st)?;
                    }
                    FfnDevice::MoeSubbanks { .. } => {
                        return Err(ModelError::Config(
                            "the full model's MoE layers must bind as one expert bank on one node".into(),
                        ))
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

/// One MLA layer of the full model on dense rows: q_a -> norm -> q_b; kv_a -> the latent norm and the rope-key rotation into both cache
/// planes; the absorbed query and rotated rope query; attention with the rope term; o_proj. Writes `s.mixer_out`.
#[allow(clippy::too_many_arguments)]
fn mixer_rope(
    cfg: &GlmConfig,
    w: &crate::families::glm::device::MlaDevice,
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
    if max_pos + 1 > mla::DENSE_MAX_KEYS {
        return Err(ModelError::Request(format!(
            "position {max_pos} needs the full model's DSA indexer (dense attention serves below {} tokens)",
            mla::DENSE_MAX_KEYS
        )));
    }
    let qk = m.qk_head_dim();
    let kv_n = (m.kv_lora_rank + m.qk_rope_head_dim).next_multiple_of(128);
    w.q_a.forward(&s.hidden, &mut s.q_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    gb_cuda::glm_glue::rmsnorm_f16(&s.q_a, &w.q_a_norm, &mut s.q_resid, rows, m.q_lora_rank, RMS_EPS, st)?;
    w.q_b.forward(&s.q_resid, &mut s.q, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    w.kv_a.forward(&s.hidden, &mut s.kv_a, rows, Dtype::F16, Dtype::F16, &mut s.ws, st)?;
    gb_cuda::glm_glue::rmsnorm_f16_strided(&s.kv_a, kv_n, &w.kv_a_norm, &mut s.ckv, rows, m.kv_lora_rank, RMS_EPS, st)?;
    mla::latent_write_rope(
        &s.ckv,
        &s.kv_a,
        kv_n,
        m.kv_lora_rank,
        cos,
        sin,
        mla::ROPE_DIM / 2,
        ri,
        table,
        latent,
        rows,
        st,
    )?;
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
    let (q_lat, qrot, o_lat) = (&s.mla.q_lat, &s.qrot, &mut s.mla.o_lat);
    mla::attend_rope(q_lat, qrot, latent, table, ri, rows, o_lat, max_pos + 1, max_pos + 1, &mut s.mla.ws, st)?;
    let o_lat = &s.mla.o_lat;
    w.o.forward(o_lat, &mut s.mixer_out, rows, Dtype::F16, Dtype::F32, &mut s.ws, st)?;
    Ok(())
}
