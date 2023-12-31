/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_SCAL2S_MXN_H
#define BLIS_SCAL2S_MXN_H

// scal2s_mxn

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
BLIS_INLINE void PASTEMAC(ch,opname) \
     ( \
       const conj_t       conjx, \
       const dim_t        m, \
       const dim_t        n, \
       ctype*    restrict alpha, \
       ctype*    restrict x, const inc_t rs_x, const inc_t cs_x, \
       ctype*    restrict y, const inc_t rs_y, const inc_t cs_y  \
     ) \
{ \
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict xj = x + j*cs_x; \
			ctype* restrict yj = y + j*cs_y; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict xij = xj + i*rs_x; \
				ctype* restrict yij = yj + i*rs_y; \
\
				PASTEMAC(ch,scal2js)( *alpha, *xij, *yij ); \
			} \
		} \
	} \
	else /* if ( bli_is_noconj( conjx ) ) */ \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict xj = x + j*cs_x; \
			ctype* restrict yj = y + j*cs_y; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype* restrict xij = xj + i*rs_x; \
				ctype* restrict yij = yj + i*rs_y; \
\
				PASTEMAC(ch,scal2s)( *alpha, *xij, *yij ); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( scal2s_mxn )

#endif
