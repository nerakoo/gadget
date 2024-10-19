#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <arm_sve.h>

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
		   int    m,
                   int    n,
                   double *a,
                   double *b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i, len = m / 4 * 4;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < len; i+=4 )
            { 
	      // ldc is used here because a[] and b[] are not packed by the
	      // starter code
	      // cse260 - you can modify the leading indice to DGEMM_NR and DGEMM_MR as appropriate
	      //
	        c( i, j, ldc ) += a( i, l, ldc) * b( l, j, ldc );   
            c(i + 1, j, ldc) += a(l, i + 1, DGEMM_MR) * b(l, j, DGEMM_NR);
            c(i + 2, j, ldc) += a(l, i + 2, DGEMM_MR) * b(l, j, DGEMM_NR);
            c(i + 3, j, ldc) += a(l, i + 3, DGEMM_MR) * b(l, j, DGEMM_NR);
            }
            for (; i < m; ++i)
                c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
        }
    }

}


// cse260
// you can put your optimized kernels here
// - put the function prototypes in bl_dgemm_kernel.h
// - define BL_MICRO_KERNEL appropriately in bl_config.h
//
inline void bl_dgemm_sve_4x8(int k,
                             int m,
                             int n,
                             const double *restrict a,
                             const double *restrict b,
                             double *c,
                             unsigned long long ldc,
                             aux_t *data)
{
    int l;
    
    svbool_t pg = svptrue_b64(); // Predicate to enable SVE instructions
    svfloat64_t c00_c01_c02_c03 = svld1(pg, c + 0 * ldc);
    svfloat64_t c04_c05_c06_c07 = svld1(pg, c + 0 * ldc + 4);
    svfloat64_t c10_c11_c12_c13 = svld1(pg, c + 1 * ldc);
    svfloat64_t c14_c15_c16_c17 = svld1(pg, c + 1 * ldc + 4);
    svfloat64_t c20_c21_c22_c23 = svld1(pg, c + 2 * ldc);
    svfloat64_t c24_c25_c26_c27 = svld1(pg, c + 2 * ldc + 4);
    svfloat64_t c30_c31_c32_c33 = svld1(pg, c + 3 * ldc);
    svfloat64_t c34_c35_c36_c37 = svld1(pg, c + 3 * ldc + 4);

    for (l = 0; l < k; ++l)
    {
        // Load broadcasted values from matrix A
        svfloat64_t a0l_a0l_a0l_a0l = svdup_n_f64(a[l * 4 + 0]);
        svfloat64_t a1l_a1l_a1l_a1l = svdup_n_f64(a[l * 4 + 1]);
        svfloat64_t a2l_a2l_a2l_a2l = svdup_n_f64(a[l * 4 + 2]);
        svfloat64_t a3l_a3l_a3l_a3l = svdup_n_f64(a[l * 4 + 3]);

        // Load B matrix values
        svfloat64_t bl0_bl1_bl2_bl3 = svld1(pg, b + l * 8);
        svfloat64_t bl4_bl5_bl6_bl7 = svld1(pg, b + l * 8 + 4);

        // Perform fused multiply-add (FMA) operations
        c00_c01_c02_c03 = svmla_f64_m(pg, c00_c01_c02_c03, a0l_a0l_a0l_a0l, bl0_bl1_bl2_bl3);
        c04_c05_c06_c07 = svmla_f64_m(pg, c04_c05_c06_c07, a0l_a0l_a0l_a0l, bl4_bl5_bl6_bl7);
        c10_c11_c12_c13 = svmla_f64_m(pg, c10_c11_c12_c13, a1l_a1l_a1l_a1l, bl0_bl1_bl2_bl3);
        c14_c15_c16_c17 = svmla_f64_m(pg, c14_c15_c16_c17, a1l_a1l_a1l_a1l, bl4_bl5_bl6_bl7);
        c20_c21_c22_c23 = svmla_f64_m(pg, c20_c21_c22_c23, a2l_a2l_a2l_a2l, bl0_bl1_bl2_bl3);
        c24_c25_c26_c27 = svmla_f64_m(pg, c24_c25_c26_c27, a2l_a2l_a2l_a2l, bl4_bl5_bl6_bl7);
        c30_c31_c32_c33 = svmla_f64_m(pg, c30_c31_c32_c33, a3l_a3l_a3l_a3l, bl0_bl1_bl2_bl3);
        c34_c35_c36_c37 = svmla_f64_m(pg, c34_c35_c36_c37, a3l_a3l_a3l_a3l, bl4_bl5_bl6_bl7);
    }

    // Store the results back to C matrix
    svst1(pg, c + 0 * ldc, c00_c01_c02_c03);
    svst1(pg, c + 0 * ldc + 4, c04_c05_c06_c07);
    svst1(pg, c + 1 * ldc, c10_c11_c12_c13);
    svst1(pg, c + 1 * ldc + 4, c14_c15_c16_c17);
    svst1(pg, c + 2 * ldc, c20_c21_c22_c23);
    svst1(pg, c + 2 * ldc + 4, c24_c25_c26_c27);
    svst1(pg, c + 3 * ldc, c30_c31_c32_c33);
    svst1(pg, c + 3 * ldc + 4, c34_c35_c36_c37);
}
