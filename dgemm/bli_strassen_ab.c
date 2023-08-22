#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"
#include <time.h>

#if 0
Create a packm kernel 

struct_t strucc = BLIS_GENERAL;
diag_t diagc    = BLIS_NONUNIT_DIAG; 
uplo_t uploc    = BLIS_DENSE; 
pack_t schema   = BLIS_PACKED_ROW_PANELS;
bool invdiag    = FALSE;

dim_t   panel_dim;
dim_t   panel_len; 
dim_t   panel_dim_max; 
dim_t   panel_len_max; 
dim_t   panel_dim_off; 
dim_t   panel_len_off; 
dim_t   panel_bcast; 

#endif	


typedef struct fmm_params_t
{
	dim_t R_L; // = 7; //number of multiplies
    dim_t num_splits; // number of partitions of each matrix

    int* coeff; // ( m_s * k_s ) x R_L matrix

    // offsets of the partitions from A0
    inc_t *row_off; 
    inc_t *col_off; 
} fmm_params_t;


void bli_dpackm_lcombination_ker 
     ( 
             struc_t strucc, 
             diag_t  diagc, 
             uplo_t  uploc, 
             conj_t  conjc, 
             pack_t  schema, 
             bool    invdiag, 
             dim_t   panel_dim, 
             dim_t   panel_len, 
             dim_t   panel_dim_max, 
             dim_t   panel_len_max, 
             dim_t   panel_dim_off, 
             dim_t   panel_len_off, 
             dim_t   panel_bcast, 
       const void*   kappa, 
       const void*   c, inc_t incc, inc_t ldc, 
             void*   p,             inc_t ldp, 
       const void*   params, 
       const cntx_t* cntx  
     )
{
    num_t dt = BLIS_DOUBLE;

    double *pptr = p;
    double *cptr = c; 
    double kappa_cast = *(double *) kappa;

    fmm_params_t fmm_params = *(fmm_params_t *)params;
    dim_t *col_off;
    dim_t *row_off;
   
#if 0
    printf("A: off_k[0] %lld, off_k[1] %lld, off_m[0] %lld, off_m[1] %lld\n", off_k[0], off_k[1], off_m[0], off_m[1]);

    printf("panel len %lld, panel_dim %lld\n", panel_len, panel_dim);
    printf("kappa cast %f\n", kappa_cast);

    printf("panel len %lld, panel_dim %lld\n", panel_len, panel_dim);
    printf("Max panel len %lld, panel_dim %lld\n", panel_len_max, panel_dim_max);

    printf("panel len off %lld, panel_dim_off %lld\n", panel_len_off, panel_dim_off);

    printf("incc %lld, ldc %lld, ldp %lld\n", incc, ldc, ldp);

#endif 

    if ( schema != BLIS_PACKED_ROW_PANELS && 
		 schema != BLIS_PACKED_COL_PANELS ) 
		bli_abort(); 

    if ( schema == BLIS_PACKED_ROW_PANELS )
    {
        row_off = fmm_params.row_off;
        col_off = fmm_params.col_off;

    }  
    else if ( schema == BLIS_PACKED_COL_PANELS )  
    {
        row_off = fmm_params.col_off;
        col_off = fmm_params.row_off;     
    } 


    /* These loops assume that there are only two matrices that need to be added
    together during packing. This needs to be generalized for the code that 
    I've started writing in bli_strassen_ab. 
    Based on the params.coeff we need to determine which matrices need to be 
    added together and with what coefficients. 
    */
    for ( dim_t j = 0; j < panel_len; j++ ) 
    {
        for (dim_t i = 0;i < panel_dim;i++) 
        {
            //bli_dscal2s( kappa_cast, cptr[ i*incc + j*ldc ], pptr[ i + j*ldp ] ); 

            pptr[ i + j*ldp ] = kappa_cast * 
                ( cptr[ (i + row_off[0] ) * incc + (j + col_off[0] ) * ldc ] 
                + cptr[ (i + col_off[1] ) * incc + (j + col_off[1] ) * ldc ] );
        }
        for (dim_t i = panel_dim;i < panel_dim_max;i++) 
            bli_dset0s( pptr[ i + j*ldp ] ); 
    }   
    for (dim_t j = panel_len;j < panel_len_max;j++)
    {
        for (dim_t i = 0;i < panel_dim_max;i++) 
        {
            bli_dset0s( pptr[ i + j*ldp ] ); 
        }
    }
        
}
 

void bl_acquire_spart 
     (
             dim_t     row_splits,
             dim_t     col_splits,
             dim_t     split_rowidx,
             dim_t     split_colidx,
       const obj_t*    obj, // source
             obj_t*    sub_obj // destination
     )

{
    dim_t m, n;
    dim_t row_part, col_part; //size of partition
    dim_t row_left, col_left; // edge case
    inc_t  offm_inc = 0;
	inc_t  offn_inc = 0;

    //I do not like that this partition size calcuation has been done both here and in bli_strassen_ab. Maybe this routine should set up the params struct. 

    m = bli_obj_length( &obj ); 
    n = bli_obj_width( &obj ); 

    row_part = m / row_splits;
    col_part = n / col_splits;

    row_left = m % row_splits;
    col_left = n % col_splits;

    /* AT the moment not dealing with edge cases. bli_strassen_ab checks for 
    edge cases. But does not do anything with it. 
    */
    if ( row_left != 0 || col_left != 0 )
        bli_abort(); 

    row_part = m / row_splits;
    col_part = n / col_splits;

    bli_obj_init_subpart_from( obj, sub_obj );

    bli_obj_set_dims( row_part, col_part, sub_obj );

    offm_inc = split_rowidx * row_part;
	offn_inc = split_colidx * col_part;

    //Taken directly from BLIS. Need to verify if this is still true. 
    // Compute the diagonal offset based on the m and n offsets.
	doff_t diagoff_inc = ( doff_t )offm_inc - ( doff_t )offn_inc;

    bli_obj_inc_offs( offm_inc, offn_inc, sub_obj );
	bli_obj_inc_diag_offset( diagoff_inc, sub_obj );

}

void bli_strassen_ab( obj_t* alpha, obj_t* A, obj_t* B, obj_t* beta, obj_t* C )
{
#if 0
    bli_gemm( alpha, A, B, beta, C);
#else
    bli_init_once();

    cntx_t* cntx = NULL;
    rntm_t* rntm = NULL;
    
	// Check the operands.
	// if ( bli_error_checking_is_enabled() )
	// 	bli_gemm_check( alpha, A, B, beta, C, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, A, B, beta, C ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( C );
	ind_t im = BLIS_NAT;

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C in case we need to apply transformations.
	obj_t A_local;
	obj_t B_local;
	obj_t C_local;
    // Aliasing the first partition of A, B, and C into the local parameters. 
	// bli_obj_alias_submatrix( A, &A_local );
	// bli_obj_alias_submatrix( B, &B_local );
	// bli_obj_alias_submatrix( C, &C_local );
    
	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_GEMM,
	  alpha,
	  &A_local,
	  &B_local,
	  beta,
	  &C_local,
	  cntx,
	  &cntl
	);


    func_t kers;
    bli_func_set_dt( &bli_dpackm_lcombination_ker, BLIS_DOUBLE, &kers);
    bli_gemm_cntl_set_packa_ukr_simple( &kers, &cntl );
    bli_gemm_cntl_set_packb_ukr_simple( &kers, &cntl );

    dim_t m, k, n;
    dim_t m_edge, m_whole, k_edge, k_whole, n_edge, n_whole;
    dim_t m_splits, k_splits, n_splits;

    // For <2, 2, 2> Strassen -> This needs to be made generic. 

    m_splits = 2, k_splits = 2, n_splits = 2;

    m = bli_obj_length( C );
    n = bli_obj_width ( C );
    k = bli_obj_width ( A );

    // Calculation of edge cases taken from Jianyu's code. 
    m_edge = m % ( m_splits * DGEMM_MR );
    k_edge = k % ( k_splits );
    n_edge = n % ( n_splits * DGEMM_NR );
    m_whole = (m - m_edge);
    k_whole = (k - k_edge); 
    n_whole = (n - n_edge);

    // total splits, which split, source, destination
    obj_t A0, B0, C0;

    bl_acquire_spart (m_splits, k_splits, 0, 0, A, &A0 );
    bl_acquire_spart (k_splits, n_splits, 0, 0, B, &B0 );
    bl_acquire_spart (m_splits, n_splits, 0, 0, C, &C0 );

    bli_obj_alias_submatrix( &A0, &A_local );
	bli_obj_alias_submatrix( &B0, &B_local );
	bli_obj_alias_submatrix( &C0, &C_local );
    

    fmm_params_t paramsA, paramsB, paramsC;

    // For <2, 2, 2> Strassen -> This needs to be made generic. 
    // Having the number of multiplies (R_L) in the params is probably //redundant...
    paramsA.R_L = 7; paramsA.num_splits = 4; 
    paramsB.R_L = 7; paramsB.num_splits = 4; 
    paramsC.R_L = 7; paramsC.num_splits = 4; 



    // There is probably a better way to define these...?? I don't like this. 
    /*
    1 0 0 0 1 0 0
    1 0 -1 -1 0 -1 0
    0 -1 0 0 1 1 -1
    0 0 -1 0 0 0 -1
    #
    1 0 0 -1 1 -1 0
    0 -1 0 0 1 0 0
    0 0 -1 1 0 0 0
    0 1 -1 0 0 -1 1
    #
    1 0 0 -1 0 0 0
    -1 -1 0 0 1 1 0
    0 0 1 1 0 -1 1
    0 1 0 0 0 0 -1
    */

    int U[4][7] = {{ 1,  0,  0,  0,  1,  0,  0}, { 1,  0, -1, -1,  0, -1,  0}, { 0, -1,  0,  0,  1,  1, -1}, { 0,  0, -1,  0,  0,  0, -1} };

    paramsA.coeff = &U[0][0];

    int V[4][7] = { { 1,  0,  0, -1,  1, -1,  0}, { 0, -1,  0,  0,  1,  0,  0}, { 0,  0, -1,  1,  0,  0,  0}, { 0,  1, -1,  0,  0, -1,  1} } ;
    paramsB.coeff = &V[0][0];


    int W[4][7] = {{ 1,  0,  0, -1,  0,  0,  0}, {-1, -1,  0,  0,  1,  1,  0},
    { 0,  0,  1,  1,  0, -1,  1}, { 0,  1,  0,  0,  0, 0,  1}};

    paramsC.coeff = &W[0][0];

    
    // For matrix A
    // (0,0)
    // (0, k/2)
    // (m/2, 0)
    // (m/2, k/2)

    dim_t row_off_A[4], col_off_A[4];

    row_off_A[0] = 0,         col_off_A[0] = 0;
    row_off_A[1] = 0,         col_off_A[1] = k_whole/2;
    row_off_A[2] = m_whole/2, col_off_A[2] = 0;
    row_off_A[3] = m_whole/2, col_off_A[3] = k_whole/2;

    paramsA.row_off = row_off_A;
    paramsA.col_off = col_off_A;

    // For Matrix B
    // (0, 0)
    // (0, n/2)
    // (k/2, 0)
    // (k/2, n/2)

    dim_t row_off_B[4], col_off_B[4];

    row_off_B[0] = 0,         col_off_B[0] = 0;
    row_off_B[1] = 0,         col_off_B[1] = n_whole/2;
    row_off_B[2] = k_whole/2, col_off_B[2] = 0;
    row_off_B[3] = k_whole/2, col_off_B[3] = n_whole/2;

    paramsB.row_off = row_off_B;
    paramsB.col_off = col_off_B;


    // For Matrix C
    // (0, 0)
    // (0, n/2)
    // (m/2, 0)
    // (m/2, n/2)

    dim_t row_off_C[4], col_off_C[4];

    row_off_C[0] = 0,         col_off_C[0] = 0;
    row_off_C[1] = 0,         col_off_C[1] = n_whole/2;
    row_off_C[2] = m_whole/2, col_off_C[2] = 0;
    row_off_C[3] = m_whole/2, col_off_C[3] = n_whole/2;

    paramsC.row_off = row_off_C;
    paramsC.col_off = col_off_C;

    bli_gemm_cntl_set_packa_params((const void *) &paramsA, &cntl);
    bli_gemm_cntl_set_packb_params((const void *) &paramsB, &cntl);


	// Invoke the internal back-end via the thread handler.
	bli_l3_thread_decorator
	(
	  &A_local,
	  &B_local,
	  &C_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
#endif 


}
