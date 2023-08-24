/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin
   Copyright (C) 2023, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include STRINGIFY_INT(../PASTEMAC(plugin,BLIS_PNAME_INFIX).h)

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             struc_t strucc, \
             diag_t  diagc, \
             uplo_t  uploc, \
             conj_t  conjc, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   panel_dim, \
             dim_t   panel_len, \
             dim_t   panel_dim_max, \
             dim_t   panel_len_max, \
             dim_t   panel_dim_off, \
             dim_t   panel_len_off, \
             dim_t   panel_bcast, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params_, \
       const cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	fmm_params_t*    params    = ( fmm_params_t* )params_; \
	packm_cxk_ker_ft packm_def = bli_cntx_get_ukr_dt( dt, BLIS_PACKM_KER, cntx ); \
\
	dim_t nsplit = params->nsplit; \
	ctype* restrict coef = ( ctype* )params->coef; \
	inc_t* restrict off_m = params->off_m; \
	inc_t* restrict off_k = params->off_k; \
	dim_t m_max = params->m_max, k_max = params->k_max; \
\
	/* The first sub-matrix also needs a coefficient. */ \
	ctype kappa_cast, lambda; \
	kappa_cast = *( ctype* )kappa; \
	PASTEMAC(ch,scal2s)( kappa_cast, coef[ 0 ], lambda ); \
\
	/* First, call the usual packing kernel to pack the first sub-matrix and take
	   care zeroing out the edges. */ \
	packm_def \
	( \
	  conjc, \
	  schema, \
	  panel_dim, \
	  panel_dim_max, \
	  panel_bcast, \
	  panel_len, \
	  panel_len_max, \
	  &lambda, \
	  c, incc, ldc, \
	  p,       ldp, \
	  params, \
	  cntx \
	); \
\
	for ( dim_t k = 1; k < nsplit; k++ ) \
	{ \
		const ctype* restrict c_use = ( ctype* )c + off_m[ k-1 ] * incc + off_k[ k-1 ] * ldc; \
		      ctype* restrict p_use = ( ctype* )p; \
\
		PASTEMAC(ch,scal2s)( kappa_cast, coef[ k ], lambda ); \
\
		/* Check if we need to shrink the micro-panel due to unequal partitioning. */ \
		dim_t panel_dim_use = bli_min( panel_dim, m_max - ( panel_dim_off + off_m[ k-1 ] ) ); \
		dim_t panel_len_use = bli_min( panel_len, k_max - ( panel_len_off + off_k[ k-1 ] ) ); \
\
		/* For subsequence sub-matrices, we don't need to re-zero any edges, just accumulate. */ \
		for ( dim_t j = 0; j < panel_len_use; j++ ) \
		{ \
			for ( dim_t i = 0; i < panel_dim_use; i++ ) \
			for ( dim_t d = 0; d < panel_bcast; d++ ) \
			{ \
				PASTEMAC(ch,axpys)( lambda, c_use[ i*incc ], p_use[ i*panel_bcast + d ] ); \
			} \
			c_use += ldc; \
			p_use += ldp; \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( packm_fmm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
