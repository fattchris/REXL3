//! The full GLM-5.3 (`glm_moe_dsa`) DSA indexer kernels (`csrc/glm_idx.cu`): per-token keys (LayerNorm + rope), the query rope, the
//! row-invariant score and the deterministic top-k selection. The indexer is per token (no pools); a full layer writes its keys once
//! per row and scores its own row against every visible key.

use crate::attention::{BlockTable, RowIndex};
use crate::rt::{CudaError, DevBuf, Stream};

/// `index_head_dim`.
pub const D: usize = 128;
/// `index_n_heads`.
pub const H: usize = 32;
/// The first dims the indexer ropes.
pub const ROPE: usize = 64;
/// Tokens per cache page.
pub const PAGE: usize = 256;
/// Row stride of the selection indices (`dsa::K_STRIDE`), -1 after the count.
pub const K_STRIDE: usize = 2056;
/// `index_topk`.
pub const TOPK: usize = 2048;
/// The key LayerNorm epsilon of the full model.
pub const LN_EPS: f32 = 1e-6;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        #[allow(clippy::too_many_arguments)]
        pub fn gb_glm_idx_key(
            kraw: *const c_void,
            kraw_stride: i64,
            lnw: *const c_void,
            lnb: *const c_void,
            ct: *const c_void,
            st: *const c_void,
            table_stride: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            block_table: *const i32,
            max_blocks: i32,
            keys: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_glm_idx_q(
            q: *const c_void,
            q_stride: i64,
            ct: *const c_void,
            st: *const c_void,
            table_stride: i32,
            row_pos: *const i32,
            qout: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_glm_idx_score(
            q: *const c_void,
            q_stride: i64,
            w: *const c_void,
            keys: *const c_void,
            block_table: *const i32,
            max_blocks: i32,
            row_slot: *const i32,
            row_pos: *const i32,
            n_keys_max: i32,
            stride: i32,
            scores: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn gb_glm_idx_select(
            scores: *const c_void,
            stride: i32,
            row_pos: *const i32,
            k: i32,
            out: *mut c_void,
            meta: *mut c_void,
            rows: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn check_dtype(d: crate::exl3_linear::Dtype) -> i32 {
    assert!(matches!(d, crate::exl3_linear::Dtype::F16 | crate::exl3_linear::Dtype::BF16), "glm_idx takes 16-bit tensors");
    d as i32
}

/// Writes the keys of `n_rows` rows: `kraw` `T [rows][stride]` (the `wk` output), the LayerNorm weights `lnw`/`lnb` `T [128]`, the
/// rope tables fp32 `[max_pos][32]`, into `keys` `T [blocks][256][128]` at each row's position.
#[allow(clippy::too_many_arguments)]
pub fn key(
    kraw: &DevBuf,
    kraw_stride: usize,
    lnw: &DevBuf,
    lnb: &DevBuf,
    cos: &DevBuf,
    sin: &DevBuf,
    table_stride: usize,
    rows: &RowIndex,
    table: &BlockTable,
    keys: &mut DevBuf,
    dtype: crate::exl3_linear::Dtype,
    n_rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let _ = check_dtype(dtype);
    assert!(n_rows <= rows.rows && kraw_stride >= D);
    let rc = unsafe {
        ffi::gb_glm_idx_key(
            kraw.ptr(),
            kraw_stride as i64,
            lnw.ptr(),
            lnb.ptr(),
            cos.ptr(),
            sin.ptr(),
            table_stride as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            keys.ptr(),
            n_rows as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Ropes the first 64 dims of every head of `qi` `T [rows][32][128]` into `qout` (same layout).
#[allow(clippy::too_many_arguments)]
pub fn query_rope(
    qi: &DevBuf,
    q_stride: usize,
    cos: &DevBuf,
    sin: &DevBuf,
    table_stride: usize,
    rows: &RowIndex,
    qout: &mut DevBuf,
    n_rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= rows.rows && q_stride >= H * D);
    let rc = unsafe {
        ffi::gb_glm_idx_q(
            qi.ptr(),
            q_stride as i64,
            cos.ptr(),
            sin.ptr(),
            table_stride as i32,
            rows.pos.ptr() as *const i32,
            qout.ptr(),
            n_rows as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Scores `n_rows` rows over their visible keys (`0..=pos`): `scores` f32 `[rows][stride]` (`-inf` past the row's position).
#[allow(clippy::too_many_arguments)]
pub fn score(
    qi: &DevBuf,
    q_stride: usize,
    w: &DevBuf,
    keys: &DevBuf,
    table: &BlockTable,
    rows: &RowIndex,
    n_keys_max: usize,
    stride: usize,
    scores: &mut DevBuf,
    n_rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= rows.rows && stride >= n_keys_max && n_keys_max >= 1);
    let rc = unsafe {
        ffi::gb_glm_idx_score(
            qi.ptr(),
            q_stride as i64,
            w.ptr(),
            keys.ptr(),
            table.buf.ptr() as *const i32,
            table.max_blocks as i32,
            rows.slot.ptr() as *const i32,
            rows.pos.ptr() as *const i32,
            n_keys_max as i32,
            stride as i32,
            scores.ptr(),
            n_rows as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The top-`k` selection of each row's scores: `out` i32 `[rows][K_STRIDE]` (ascending positions, -1 after the count), `meta`
/// i32 `[rows][2]` (`meta[2r] = 1` for a sparse row, `meta[2r+1]` the count).
#[allow(clippy::too_many_arguments)]
pub fn select(
    scores: &DevBuf,
    stride: usize,
    rows: &RowIndex,
    k: usize,
    out: &mut DevBuf,
    meta: &mut DevBuf,
    n_rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(n_rows <= rows.rows && (1..=TOPK).contains(&k));
    let rc = unsafe {
        ffi::gb_glm_idx_select(
            scores.ptr(),
            stride as i32,
            rows.pos.ptr() as *const i32,
            k as i32,
            out.ptr(),
            meta.ptr(),
            n_rows as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}
