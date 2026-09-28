#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include "fftw_executor.h"
#include "soacomplex.h"

#ifndef DDM_FFT
#include "wrap_fft.h"
/*
* necessary work size is (size_x * size_y * size_z * 4).
*/
inline
void SetPotentialByPoissonFFT(double* Vhart, const double* rho, int size_x, int size_y, int size_z,
	double dx, double dy, double dz, double* work) {

	OneComplex* rhok=(OneComplex*)work;
	const int64_t size_3d = (int64_t)size_x * (int64_t)size_y * (int64_t)size_z;

	//#undef USE_R2C_C2R
#ifndef USE_R2C_C2R

    OneComplex* rho_comp = (OneComplex*)(work + size_3d * 2);
	for (size_t i = 0; i < size_3d; ++i) {
		rho_comp[i] = { rho[i],0.0 };
	}

	FFT_3D(rhok, rho_comp, size_x, size_y, size_z);

	const int kz_end = size_z;
	const int kx_end = size_x;

#else
	/*
	problem: this path cannot obtain correct Vhart
	*/
	auto& rho_d = m_work_rd;


	{
		vecmath::Copy<double>(rho_d, rho, size_3d);
		//rho_d[m_size_x / 2 + m_size_x * (m_size_y / 2 + m_size_y * (m_size_z / 2))] -= 1.0 / (m_dx * m_dy * m_dz);
	}

	double sumrho = 0.0;
	for (size_t i = 0; i < size_3d; ++i) {
		sumrho += rho_d[i];
	}
	sumrho *= dx * dy * dz;
	printf("rho in poisson = %f\n", sumrho);

	/*fftw_plan_dft_r2c_3d*/
	FFT_3D(rhok, rho_d, m_size_x, m_size_y, m_size_z);

	const int kz_end = m_size_z;
	const int kx_end = m_size_x / 2 + 1;
#endif

	//NOTE: 
	//  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
	//    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

	const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
	const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
	const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

	auto SQ = [](double x) { return x * x; };

	//#define NO_FOLD

	for (int kz = 0; kz < kz_end; ++kz) {
#ifdef NO_FOLD
		const double kz2 = coef_z * SQ((double)(kz));
#else

		const double kz2 = SQ(coef1_z * (double)(kz *2 > size_z ? kz - size_z: kz));
#endif
		for (int ky = 0; ky < size_y; ++ky) {
#ifdef NO_FOLD
			const double ky2 = coef_z * SQ((double)(ky));
#else
			const double ky2 = SQ(coef1_y * (double)(ky * 2 > size_y ?  ky - size_y : ky));
#endif


			for (int kx = 0; kx < kx_end; ++kx) {
#ifdef NO_FOLD
				const double kx2 = coef_z * SQ((double)(kx));
#else
                const double kx2 = SQ(coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
#endif
				const size_t i = kx + kx_end * (ky + (size_y * kz));

				if (kx == 0 && ky == 0 && kz == 0) {
					rhok[i] = 0.0;
				} else {
					rhok[i].r *= 4.0 * M_PI / (kx2 + ky2 + kz2);
                    rhok[i].i *= 4.0 * M_PI / (kx2 + ky2 + kz2);
				}
			}

		}
	}

#ifndef USE_R2C_C2R 

    OneComplex* Vh =(OneComplex*)(work + size_3d * 2);
	/*
	for (size_t i = 0; i < m_size_3d; ++i) {
		Vh[i] = { 0.0,0.0 };
	}
	*/
	IFFT_3D(Vh, rhok, size_x, size_y, size_z);

	for (size_t i = 0; i < size_3d; ++i) {
		Vhart[i] = Vh[i].r;
	}
#else

	IFFT_3D(Vhart, rhok, size_x, size_y, size_z);

#endif


}


/*
* necessary work size is (size_x * size_y * size_z * 6).
*/
inline
void SetPotentialByPoissonFFT_keepRhok(double* Vhart, const double* rho, int size_x, int size_y, int size_z,
    double dx, double dy, double dz, double* work) {

    const int64_t size_3d = (int64_t)size_x * (int64_t)size_y * (int64_t)size_z;
    OneComplex* rhok = (OneComplex*)work;
    OneComplex* rhok_kk = (OneComplex*)(work + size_3d * 2);

    //#undef USE_R2C_C2R
#ifndef USE_R2C_C2R

    OneComplex* rho_comp = (OneComplex*)(work + size_3d * 4);
    for (size_t i = 0; i < size_3d; ++i) {
        rho_comp[i] = { rho[i],0.0 };
    }

    FFT_3D(rhok, rho_comp, size_x, size_y, size_z);

    const int kz_end = size_z;
    const int kx_end = size_x;

#else
    /*
    problem: this path cannot obtain correct Vhart
    */
    auto& rho_d = m_work_rd;


    {
        vecmath::Copy<double>(rho_d, rho, size_3d);
        //rho_d[m_size_x / 2 + m_size_x * (m_size_y / 2 + m_size_y * (m_size_z / 2))] -= 1.0 / (m_dx * m_dy * m_dz);
    }

    double sumrho = 0.0;
    for (size_t i = 0; i < size_3d; ++i) {
        sumrho += rho_d[i];
    }
    sumrho *= dx * dy * dz;
    printf("rho in poisson = %f\n", sumrho);

    /*fftw_plan_dft_r2c_3d*/
    FFT_3D(rhok, rho_d, m_size_x, m_size_y, m_size_z);

    const int kz_end = m_size_z;
    const int kx_end = m_size_x / 2 + 1;
#endif

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

    auto SQ = [](double x) { return x * x; };

    //#define NO_FOLD

    for (int kz = 0; kz < kz_end; ++kz) {
#ifdef NO_FOLD
        const double kz2 = coef_z * SQ((double)(kz));
#else

        const double kz2 = SQ(coef1_z * (double)(kz * 2 > size_z ?  kz - size_z : kz));
#endif
        for (int ky = 0; ky < size_y; ++ky) {
#ifdef NO_FOLD
            const double ky2 = coef_z * SQ((double)(ky));
#else
            const double ky2 = SQ(coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));
#endif


            for (int kx = 0; kx < kx_end; ++kx) {
#ifdef NO_FOLD
                const double kx2 = coef_z * SQ((double)(kx));
#else
                const double kx2 = SQ(coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
#endif
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                if (kx == 0 && ky == 0 && kz == 0) {
                    rhok_kk[i] = 0.0;
                } else {
                    rhok_kk[i].r = rhok[i].r * 4.0 * M_PI / (kx2 + ky2 + kz2);
                    rhok_kk[i].i = rhok[i].i * 4.0 * M_PI / (kx2 + ky2 + kz2);
                }
            }

        }
    }

#ifndef USE_R2C_C2R 

    OneComplex* Vh = (OneComplex*)(work + size_3d * 4);
    /*
    for (size_t i = 0; i < m_size_3d; ++i) {
        Vh[i] = { 0.0,0.0 };
    }
    */
    IFFT_3D(Vh, rhok_kk, size_x, size_y, size_z);

    for (size_t i = 0; i < size_3d; ++i) {
        Vhart[i] = Vh[i].r;
    }
#else

    IFFT_3D(Vhart, rhok_kk, size_x, size_y, size_z);

#endif


}
#endif //DDM_FFT


/*
* If fftw of R2C is used, the last argument kx_end is set to (size_x / 2 + 1).
*/
inline
void SolvePoissonInKspace(OneComplex* rhok, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

    auto SQ = [](double x) { return x * x; };

    //#define NO_FOLD

    for (int kz = 0; kz < kz_end; ++kz) {
        const double kz2 = SQ(coef1_z * (double)(kz*2 > size_z ? kz - size_z:kz ));
        for (int ky = 0; ky < size_y; ++ky) {
            const double ky2 = SQ(coef1_y * (double)(ky*2 > size_y ?  ky - size_y : ky));
            for (int kx = 0; kx < kx_end; ++kx) {
                const double kx2 = SQ(coef1_x * (double)(kx*2 > size_x ? kx - size_x : kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                if (kx == 0 && ky == 0 && kz == 0) {
                    rhok[i].r = 0.0;
                    rhok[i].i = 0.0;
                } else {
                    rhok[i].r *= 4.0 * M_PI / (kx2 + ky2 + kz2);
                    rhok[i].i *= 4.0 * M_PI / (kx2 + ky2 + kz2);
                }
            }

        }
    }
}



/*
* If fftw of R2C is used, the last argument kx_end is set to (size_x / 2 + 1).
*/
inline 
void GradientXInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {   
        kx_end = size_x;
    }
    
    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

    

    for (int kz = 0; kz < size_z; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 >  size_z ? kz - size_z : kz));
        for (int ky = 0; ky < size_y; ++ky) {
            //const double ky1 = (coef1_y * (double)(ky *2 > size_y  ? ky - size_y : ky ));

#if 1
            const int k_half = size_x / 2 + 1;
            for (int kx = 0; kx < k_half; ++kx) {
                const double kx1 = (coef1_x * (double)(kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * kx1;
                destk[i].i = srck[i].r * kx1;

            }
            for (int kx = k_half; kx < kx_end; ++kx) {
                const double kx1 = (coef1_x * (double)( kx - size_x ));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * kx1;
                destk[i].i = srck[i].r * kx1;

            }
#else
            for (int kx = 0; kx < kx_end; ++kx) {
                const double kx1 = (coef1_x * (double)(kx*2 > size_x ? kx - size_x : kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                
                destk[i].r = -srck[i].i * kx1;
                destk[i].i = srck[i].r * kx1;
              
            }
#endif
        }
    }
}


/*
* If fftw of R2C is used, the last argument kx_end is set to (size_x / 2 + 1).
*/
inline
void GradientXInKspace_inplace(OneComplex* destk, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));



    for (int kz = 0; kz < size_z; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 >  size_z ? kz - size_z : kz));
        for (int ky = 0; ky < size_y; ++ky) {
            //const double ky1 = (coef1_y * (double)(ky *2 > size_y  ? ky - size_y : ky ));

#if 1
            const int k_half = size_x / 2 + 1;
            for (int kx = 0; kx < k_half; ++kx) {
                const double kx1 = (coef1_x * (double)(kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                double src_re = destk[i].r;
                double src_im = destk[i].i;
                destk[i].r = -src_im * kx1;
                destk[i].i = src_re * kx1;

            }
            for (int kx = k_half; kx < kx_end; ++kx) {
                const double kx1 = (coef1_x * (double)(kx - size_x));
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                double src_re = destk[i].r;
                double src_im = destk[i].i;
                destk[i].r = -src_im * kx1;
                destk[i].i = src_re * kx1;
            }
#else
            for (int kx = 0; kx < kx_end; ++kx) {
                const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x : kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * kx1;
                destk[i].i = srck[i].r * kx1;

            }
#endif
        }
    }
}


inline
void GradientYInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

    
    for (int kz = 0; kz < kz_end; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 > size_z ?  kz - size_z : kz));
#if 1

        const int k_half = size_y / 2 + 1;
        for (int ky = 0; ky < k_half; ++ky) {
            const double ky1 = (coef1_y * (double)( ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * ky1;
                destk[i].i = srck[i].r * ky1;

            }

        }
        for (int ky = k_half; ky < size_y; ++ky) {
            const double ky1 = (coef1_y * (double)( ky - size_y ));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * ky1;
                destk[i].i = srck[i].r * ky1;

            }

        }
#else
        for (int ky = 0; ky < size_y; ++ky) {
            const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * ky1;
                destk[i].i = srck[i].r * ky1;

            }

        }
#endif
    }
}


inline
void GradientYInKspace_inplace(OneComplex* destk, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));


    for (int kz = 0; kz < kz_end; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 > size_z ?  kz - size_z : kz));
#if 1

        const int k_half = size_y / 2 + 1;
        for (int ky = 0; ky < k_half; ++ky) {
            const double ky1 = (coef1_y * (double)(ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                const double src_re = destk[i].r;
                const double src_im = destk[i].i;
                destk[i].r = -src_im * ky1;
                destk[i].i = src_re * ky1;

            }

        }
        for (int ky = k_half; ky < size_y; ++ky) {
            const double ky1 = (coef1_y * (double)(ky - size_y));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                const double src_re = destk[i].r;
                const double src_im = destk[i].i;
                destk[i].r = -src_im * ky1;
                destk[i].i = src_re * ky1;
            }

        }
#else
        for (int ky = 0; ky < size_y; ++ky) {
            const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * ky1;
                destk[i].i = srck[i].r * ky1;

            }

        }
#endif
    }
}


inline
void GradientZInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

#if 1

    const int k_half = size_z / 2 + 1;
    for (int kz = 0; kz < k_half; ++kz) {
        const double kz1 = (coef1_z * (double)( kz));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;

            destk[i].r = -srck[i].i * kz1;
            destk[i].i = srck[i].r * kz1;
        }
    }
    for (int kz = k_half; kz < kz_end; ++kz) {
        const double kz1 = (coef1_z * (double)( kz - size_z ));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;

            destk[i].r = -srck[i].i * kz1;
            destk[i].i = srck[i].r * kz1;
        }
    }
#else
    for (int kz = 0; kz < kz_end; ++kz) {
        const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z: kz ));
        for (int ky = 0; ky < size_y; ++ky) {
            //const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx *2 > size_x ? kx - size_x : kx ));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * kz1;
                destk[i].i = srck[i].r * kz1;

            }

        }
    }
#endif
}


inline
void GradientZInKspace_inplace(OneComplex* destk, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

#if 1

    const int k_half = size_z / 2 + 1;
    for (int kz = 0; kz < k_half; ++kz) {
        const double kz1 = (coef1_z * (double)(kz));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;
            const double src_re = destk[i].r;
            const double src_im = destk[i].i;
            destk[i].r = -src_im * kz1;
            destk[i].i = src_re * kz1;
        }
    }
    for (int kz = k_half; kz < kz_end; ++kz) {
        const double kz1 = (coef1_z * (double)(kz - size_z));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;
            const double src_re = destk[i].r;
            const double src_im = destk[i].i;
            destk[i].r = -src_im * kz1;
            destk[i].i = src_re * kz1;
        }
    }
#else
    for (int kz = 0; kz < kz_end; ++kz) {
        const double kz1 = (coef1_z * (double)(kz * 2 > size_z ? kz - size_z : kz));
        for (int ky = 0; ky < size_y; ++ky) {
            //const double ky1 = (coef1_y * (double)(ky * 2 > size_y ? ky - size_y : ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx *2 > size_x ? kx - size_x : kx ));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = -srck[i].i * kz1;
                destk[i].i = srck[i].r * kz1;

            }

        }
    }
#endif
}

/*
* necessary work size is (size_x * size_y * size_z * 4).
*/
inline
void SetPotentialByPoissonFFT_v2(FFTW_Executor* fftw_e, double* Vhart, const double* rho, double* Vhart_in_k, int size_x, int size_y, int size_z,
    double dx, double dy, double dz) {


    const int64_t size_3d = (int64_t)size_x * (int64_t)size_y * (int64_t)size_z;

    auto* rho_comp = fftw_e->GetBuffer();
    for (size_t i = 0; i < size_3d; ++i) {
        rho_comp[i].r =  rho[i];
        rho_comp[i].i =  0.0;
    }

    auto rhok = fftw_e->ForwardDirect(rho_comp);

    SolvePoissonInKspace(rhok, size_x, size_y, size_z, dx, dy, dz);

    if (Vhart_in_k != nullptr) {
        const int64_t size_3d_r2c = 2 * (int64_t)size_x * (int64_t)size_y * (int64_t)size_z;
        memcpy(Vhart_in_k, rhok, sizeof(double) * size_3d_r2c);
    }

    //IFFT_3D(Vh, rhok, size_x, size_y, size_z);
    auto* Vh = fftw_e->BackwardDirect(rhok);
    const double invN = 1.0 / (double)(size_3d);

    for (size_t i = 0; i < size_3d; ++i) {
        Vhart[i] = Vh[i].r * invN;
    }


}



inline
void IntegralXInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));



    for (int kz = 0; kz < size_z; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 >  size_z ? kz - size_z : kz));
        for (int ky = 0; ky < size_y; ++ky) {
            //const double ky1 = (coef1_y * (double)(ky *2 > size_y  ? ky - size_y : ky ));


            const int k_half = size_x / 2 + 1;
            {
                const int kx = 0;
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                destk[i].r = 0.0;
                destk[i].i = 0.0;
            }
            for (int kx = 1; kx < k_half; ++kx) {
                const double kx1 = (coef1_x * (double)(kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = srck[i].i / kx1;
                destk[i].i = -srck[i].r / kx1;

            }
            for (int kx = k_half; kx < kx_end; ++kx) {
                const double kx1 = (coef1_x * (double)(kx - size_x));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = srck[i].i / kx1;
                destk[i].i = -srck[i].r / kx1;

            }
        }
    }
}


inline
void IntegralYInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));


    for (int kz = 0; kz < kz_end; ++kz) {
        //const double kz1 = (coef1_z * (double)(kz * 2 > size_z ?  kz - size_z : kz));


        const int k_half = size_y / 2 + 1;
        {
            const int ky = 0;            
            for (int kx = 0; kx < kx_end; ++kx) {
                const size_t i = kx + kx_end * (ky + (size_y * kz));
                destk[i].r = 0.0;
                destk[i].i = 0.0;
            }
        }
        for (int ky = 1; ky < k_half; ++ky) {
            const double ky1 = (coef1_y * (double)(ky));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = srck[i].i / ky1;
                destk[i].i = -srck[i].r / ky1;

            }

        }
        for (int ky = k_half; ky < size_y; ++ky) {
            const double ky1 = (coef1_y * (double)(ky - size_y));

            for (int kx = 0; kx < kx_end; ++kx) {
                //const double kx1 = (coef1_x * (double)(kx * 2 > size_x ? kx - size_x: kx));
                const size_t i = kx + kx_end * (ky + (size_y * kz));

                destk[i].r = srck[i].i / ky1;
                destk[i].i = -srck[i].r / ky1;

            }

        }
    }
}

inline
void IntegralZInKspace(OneComplex* destk, const OneComplex* srck, int size_x, int size_y, int size_z, double dx, double dy, double dz,
    int kx_end = 0) {

    //kx_end is used for FFTW_R2C mode//
    if (kx_end == 0) {
        kx_end = size_x;
    }
    const int kz_end = size_z;

    //NOTE: 
    //  V = c rho, where c = 4pi / (kx^2 + ky^2 + kz^2) //
    //    = 1.0/ ((pi*kx^2/box_x^2) + (pi*ky^2/box_y^2) + (pi*kz^2/box_z^2)) 

    const double coef1_x = (2.0 * M_PI / (dx * (double)size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)size_z));

    const int k_half = size_z / 2 + 1;
    {
        const int kz = 0;
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;
            destk[i].r = 0.0;
            destk[i].i = 0.0;
        }
    }
    for (int kz = 1; kz < k_half; ++kz) {
        const double kz1 = (coef1_z * (double)(kz));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;

            destk[i].r = srck[i].i / kz1;
            destk[i].i = -srck[i].r / kz1;
        }
    }
    for (int kz = k_half; kz < kz_end; ++kz) {
        const double kz1 = (coef1_z * (double)(kz - size_z));
        size_t offset_z = kx_end * size_y * kz;
        for (int kxy = 0; kxy < size_y * kx_end; ++kxy) {
            const size_t i = kxy + offset_z;

            destk[i].r = srck[i].i / kz1;
            destk[i].i = -srck[i].r / kz1;
        }
    }
}
