#pragma once
#include <cmath>
#include "soacomplex.h"




//#define SvF_NO_FOLDING

//note1: kx,ky,kz==0 is important for accuracy because they relate Hermitian conjugate between terms of kx and -kx//
//Nが偶数の時、
// 2k == N のポイントでは虚数成分は0である必要がある。
//このとき、上の方法ではexp(-ikR)での回転を、虚数成分だけ起こらないようにしたが、
//全空間積分したときの積分値も変わってしまう。
//よって、積分値を変えないためにはcosも作用しない下の方法のように、
//
//(間違い) 2k==Nのときには1を乗じた方が精度が出る
// --> 2k==Nのときに1=exp(0)を乗じると実際には精度が出ない。他と同様にexp(-k*rRx)を乗じる方が良い
//もっと理想的にはNに奇数とすればこの問題は回避できる//

inline 
void ShiftInKspace_add(OneComplex* k_dest, double Rx, double Ry, double Rz, const OneComplex* k_src,
    int grid_x, int grid_y, int grid_z,
    double dx, double dy, double dz) {

    double box_x = dx * (double)grid_x;
    double box_y = dy * (double)grid_y;
    double box_z = dz * (double)grid_z;
    double rRx = Rx / box_x;
    //rRx -= floor(rRx);
    double rRy = Ry / box_y;
    //rRy -= floor(rRy);
    double rRz = Rz / box_z;
    //rRz -= floor(rRz);


#ifdef SvF_NO_FOLDING
    for (int iz = 0; iz < grid_z; ++iz) {
        const double kz = (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            const double ky = (double)iy;
            for (int ix = 0; ix < grid_x; ++ix) {
                const double kx = (double)ix;
                
                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + grid_x * (iy + grid_y * iz);
                k_dest[i].r += cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i += sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#else
    for (int iz = 0; iz < grid_z; ++iz) {
        //const double kz = (2 * iz == grid_z) ? 0.0 : (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        //const double kz = (double)iz;
        const double kz = (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            //const double ky = (2 * iy == grid_y) ? 0.0 : (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            //const double ky = (double)iy;
            const double ky = (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            for (int ix = 0; ix < grid_x; ++ix) {
                //const double kx = (2 * ix == grid_x) ? 0.0 : (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;
                //const double kx = (double)ix;
                const double kx = (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + grid_x * (iy + grid_y * iz);
                k_dest[i].r += cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i += sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#endif
}


inline
void ShiftInKspace_set(OneComplex* k_dest, double Rx, double Ry, double Rz, const OneComplex* k_src,
    int grid_x, int grid_y, int grid_z,
    double dx, double dy, double dz) {

    double box_x = dx * (double)grid_x;
    double box_y = dy * (double)grid_y;
    double box_z = dz * (double)grid_z;
    double rRx = Rx / box_x;
    //rRx -= floor(rRx);
    double rRy = Ry / box_y;
    //rRy -= floor(rRy);
    double rRz = Rz / box_z;
    //rRz -= floor(rRz);


#ifdef SvF_NO_FOLDING
    for (int iz = 0; iz < grid_z; ++iz) {
        const double kz = (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            const double ky = (double)iy;
            for (int ix = 0; ix < grid_x; ++ix) {
                const double kx = (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + grid_x * (iy + grid_y * iz);
                k_dest[i].r += cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i += sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#else
    for (int iz = 0; iz < grid_z; ++iz) {
        //const double kz = (2 * iz == grid_z) ? 0.0 : (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        //const double kz = (double)iz;
        const double kz = (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            //const double ky = (2 * iy == grid_y) ? 0.0 : (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            //const double ky = (double)iy;
            const double ky = (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            for (int ix = 0; ix < grid_x; ++ix) {
                //const double kx = (2 * ix == grid_x) ? 0.0 : (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;
                //const double kx = (double)ix;
                const double kx = (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + grid_x * (iy + grid_y * iz);
                k_dest[i].r = cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i = sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#endif
}


inline
void ShiftInKspace_r2c_set(OneComplex* k_dest, double Rx, double Ry, double Rz, const OneComplex* k_src,
    int grid_x, int grid_y, int grid_z,
    double dx, double dy, double dz) {

    double box_x = dx * (double)grid_x;
    double box_y = dy * (double)grid_y;
    double box_z = dz * (double)grid_z;
    double rRx = Rx / box_x;
    //rRx -= floor(rRx);
    double rRy = Ry / box_y;
    //rRy -= floor(rRy);
    double rRz = Rz / box_z;
    //rRz -= floor(rRz);

    const int stride_x = grid_x / 2 + 1;


#ifdef SvF_NO_FOLDING
    for (int iz = 0; iz < grid_z; ++iz) {
        const double kz = (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            const double ky = (double)iy;
            for (int ix = 0; ix < stride_x; ++ix) {
                const double kx = (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + stride_x * (iy + grid_y * iz);
                k_dest[i].r = cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i = sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#else
    for (int iz = 0; iz < grid_z; ++iz) {
        //const double kz = (2 * iz == grid_z) ? 0.0 : (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        const double kz = (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            //const double ky = (2 * iy == grid_y) ? 0.0 : (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            const double ky = (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            for (int ix = 0; ix < stride_x; ++ix) {
                //const double kx = (2*ix == grid_x) ? 0.0 : (double)ix;
                //const double kx = (double)ix;
                const double kx = (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + stride_x * (iy + grid_y * iz);
                k_dest[i].r = cos_shift * k_src[i].r - sin_shift * k_src[i].i;
                k_dest[i].i = sin_shift * k_src[i].r + cos_shift * k_src[i].i;

            }
        }
    }
#endif
}


inline
void ShiftInKspace_update(OneComplex* k_src_dest, double Rx, double Ry, double Rz, 
    int grid_x, int grid_y, int grid_z,
    double dx, double dy, double dz,
    int ix_end = 0) {

    if (ix_end == 0) {
        ix_end = grid_x;
    }


    double box_x = dx * (double)grid_x;
    double box_y = dy * (double)grid_y;
    double box_z = dz * (double)grid_z;
    double rRx = Rx / box_x;
    double rRy = Ry / box_y;
    double rRz = Rz / box_z;
    /*
    rRx -= floor(rRx);
    rRy -= floor(rRy);
    rRz -= floor(rRz);
    */


#ifdef SvF_NO_FOLDING
    for (int iz = 0; iz < grid_z; ++iz) {
        const double kz = (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            const double ky = (double)iy;
            for (int ix = 0; ix < ix_end; ++ix) {
                const double kx = (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + ix_end * (iy + grid_y * iz);
                const double src_re = k_src_dest[i].r;
                const double src_im = k_src_dest[i].i;
                k_src_dest[i].r = cos_shift * src_re - sin_shift * src_im;
                k_src_dest[i].i = sin_shift * src_re + cos_shift * src_im;

            }
        }
    }
#else
    for (int iz = 0; iz < grid_z; ++iz) {
        //const double kz = (2 * iz == grid_z) ? 0.0 : (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        const double kz = (2 * iz > grid_z) ? (double)(iz - grid_z) : (double)iz;
        for (int iy = 0; iy < grid_y; ++iy) {
            //const double ky = (2 * iy == grid_y) ? 0.0 : (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            const double ky = (2 * iy > grid_y) ? (double)(iy - grid_y) : (double)iy;
            for (int ix = 0; ix < ix_end; ++ix) {
                //const double kx = (2 * ix == grid_x) ? 0.0 : (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;
                const double kx = (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;

                const double phase = -2.0 * M_PI * (kx * rRx + ky * rRy + kz * rRz);
                const double cos_shift = cos(phase);
                const double sin_shift = sin(phase);

                const int i = ix + ix_end * (iy + grid_y * iz);
                const double src_re = k_src_dest[i].r;
                const double src_im = k_src_dest[i].i;
                k_src_dest[i].r = cos_shift * src_re - sin_shift * src_im;
                k_src_dest[i].i = sin_shift * src_re + cos_shift * src_im;

            }
        }
    }
#endif
}



inline
void ShiftInKspace_1d(OneComplex* k_src_dest, double Rx, int grid_x, double dx) {
    const int ix_end = grid_x;

    double box_x = dx * (double)grid_x;
    double rRx = Rx / box_x;
    
    for (int ix = 0; ix < ix_end; ++ix) {
        const double kx = (2 * ix > grid_x) ? (double)(ix - grid_x) : (double)ix;

        const double phase = -2.0 * M_PI * (kx * rRx);
        const double cos_shift = cos(phase);
        const double sin_shift = sin(phase);

        const int i = ix;
        const double src_re = k_src_dest[i].r;
        const double src_im = k_src_dest[i].i;
        k_src_dest[i].r = cos_shift * src_re - sin_shift * src_im;
        k_src_dest[i].i = sin_shift * src_re + cos_shift * src_im;

    }
}


/*
* 偶数グリッドの場合に、最も高波数(k*2==grid_size)の波を消す
* これをしておくと、偶数グリッドでもSvFしたときにノルム性が良くなる
*/
inline
double ClearHighestWaveInKspace(OneComplex* k_src_dest, int grid_x, int grid_y, int grid_z, int ix_end = 0) {

    if (ix_end == 0) {
        ix_end = grid_x;
    }

    double reduce_norm = 0.0;
    if ((grid_x & 0x1) == 0) {//偶数の場合のみ//
        const int kx_half = grid_x / 2;
        for (int kz = 0; kz < grid_z; ++kz) {
            for (int ky = 0; ky < grid_y; ++ky) {
                int i = kx_half + ix_end * (ky + grid_y * kz);
                reduce_norm += k_src_dest[i].r * k_src_dest[i].r;
                k_src_dest[i].r = 0.0;
                k_src_dest[i].i = 0.0;
            }
        }
    }

    if ((grid_y & 0x1) == 0) {//偶数の場合のみ//
        const int ky_half = grid_y / 2;
        for (int kz = 0; kz < grid_z; ++kz) {
            for (int kx = 0; kx < ix_end; ++kx) {
                int i = kx+ ix_end * (ky_half + grid_y * kz);
                reduce_norm += k_src_dest[i].r * k_src_dest[i].r;
                k_src_dest[i].r = 0.0;
                k_src_dest[i].i = 0.0;
            }
        }
    }

    if ((grid_z & 0x1) == 0) {//偶数の場合のみ//
        const int kz_half = grid_z / 2;
        for (int ky = 0; ky < grid_y; ++ky) {
            for (int kx = 0; kx < ix_end; ++kx) {
                int i = kx + ix_end * (ky+ grid_y * kz_half);
                reduce_norm += k_src_dest[i].r * k_src_dest[i].r;
                k_src_dest[i].r = 0.0;
                k_src_dest[i].i = 0.0;
            }
        }
    }
    return reduce_norm;
}
