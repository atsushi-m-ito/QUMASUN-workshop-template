#pragma once
#include "shift_via_fft.h"
#include "fftw_executor.h"
#include "poisson_fft.h"

/*
* サブグリッドに定義された場tmp_RYの中心点を
* 現在地からグリッドの中点(halppoint)へシフトする
* シフト後に、境界点での値がゼロになるように補正を加える
* シフト前の現在地はcx,cy,czで指定される
* ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
*/
inline
void ShiftToHalfPoint(OneComplex* tmp_RY, int cx, int cy, int cz, int csize, 
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z) {
    constexpr double half = 0.5;
    auto fold_pos = [&half, &csize](int cx) {
        return half - (double)(cx * 2 > csize ? cx - csize : cx) / (double)csize;
        };

    const double center_x = fold_pos(cx);
    const double center_y = fold_pos(cy);
    const double center_z = fold_pos(cz);

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;

    if (cx != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            (half - center_x) * dx, 0.0, 0.0,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

        //境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 0; iy < sub_size_y; ++iy) {
                const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_x / 2);
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    double phase = 2.0 * M_PI * k * (double)ix / (double)sub_size_x;
                    tmp_RY[offset + ix].r -= ksum_r * cos(phase);
                    tmp_RY[offset + ix].i -= ksum_i * cos(phase);
                }
            }
        }

        const double invN = 1.0 / (double)sub_size;
        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

    if (cy != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            0.0, (half - center_y) * dy, 0.0,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

        //境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_y / 2);
                for (int iy = 0; iy < sub_size_y; ++iy) {
                    double phase = 2.0 * M_PI * k * (double)iy / (double)sub_size_y;
                    tmp_RY[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
                    tmp_RY[offset + sub_size_x * iy].i -= ksum_i * cos(phase);;
                }

            }
        }

        const double invN = 1.0 / (double)sub_size;
        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

    if (cz != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            0.0, 0.0, (half - center_z) * dz,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

        //境界点での値がゼロになるように補正を加える
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_z / 2);
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    double phase = 2.0 * M_PI * k * (double)iz / (double)sub_size_z;
                    tmp_RY[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
                    tmp_RY[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
                }

            }
        }

        const double invN = 1.0 / (double)sub_size;
        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

}



inline
void ShiftToHalfPoint_2(OneComplex* tmp_RY, int cx, int cy, int cz, int csize,
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z, const char* print_path) {
    constexpr double half = 0.5;
    auto fold_pos = [&half, &csize](int cx) {
        return half - (double)(cx * 2 > csize ? cx - csize : cx) / (double)csize;
        };

    const double center_x = fold_pos(cx);
    const double center_y = fold_pos(cy);
    const double center_z = fold_pos(cz);

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;
    const double invN = 1.0 / (double)sub_size;

    FILE* fp = nullptr;
#ifdef _DEBUG
    if (print_path) {
        fp = fopen(print_path, "w");
    }
#endif

    if (cx != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            (half - center_x) * dx, 0.0, 0.0,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "x-shift: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ",tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }            
        }
#endif

        //境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 0; iy < sub_size_y; ++iy) {
                const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_x / 2);
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    double phase = 2.0 * M_PI * k * (double)ix / (double)sub_size_x;
                    tmp_RY[offset + ix].r -= ksum_r * cos(phase);
                    tmp_RY[offset + ix].i -= ksum_i * cos(phase);
                }
            }
        }

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)x-shift: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

    if (cy != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            0.0, (half - center_y) * dy, 0.0,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "y-shift: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_y / 2);
                for (int iy = 0; iy < sub_size_y; ++iy) {
                    double phase = 2.0 * M_PI * k * (double)iy / (double)sub_size_y;
                    tmp_RY[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
                    tmp_RY[offset + sub_size_x * iy].i -= ksum_i * cos(phase);;
                }

            }
        }

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)y-shift: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

    if (cz != 0) {
        ShiftInKspace_update(&tmp_RY[0],
            0.0, 0.0, (half - center_z) * dz,
            sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "z-shift: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //境界点での値がゼロになるように補正を加える
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
                double ksum_r = tmp_RY[offset].r;
                double ksum_i = tmp_RY[offset].i;
                double k = (double)(sub_size_z / 2);
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    double phase = 2.0 * M_PI * k * (double)iz / (double)sub_size_z;
                    tmp_RY[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
                    tmp_RY[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
                }

            }
        }

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)z-shift: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            tmp_RY[i].r *= invN;
            tmp_RY[i].i *= invN;
        }
        fftw->ForwardDirect(tmp_RY);
    }

    if (fp) {
        fclose(fp);
    }
}


/*
* 境界点での値がゼロになるように補正を加える
* シフト前の現在地はcx, cy, czで指定される
* ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
*/
inline
void CorrectEdgeZeroByWave(OneComplex * tmp_RY, 
    FFTW_Executor * fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z) {
    
    const int sub_size = sub_size_x * sub_size_y * sub_size_z;

    
    fftw->BackwardDirect(tmp_RY);

    //x境界点での値がゼロになるように補正を加える
    for (int iz = 0; iz < sub_size_z; ++iz) {
        for (int iy = 0; iy < sub_size_y; ++iy) {
            const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;
            const int num_edge = 1 + ((iy==0)?1:0) + ((iz == 0) ? 1 : 0);
            if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                ksum_r /= (double)num_edge;  
                ksum_i /= (double)num_edge;
            }
                
            double k = (double)(sub_size_x / 2);
            for (int ix = 0; ix < sub_size_x; ++ix) {
                double phase = 2.0 * M_PI * k * (double)ix / (double)sub_size_x;
                tmp_RY[offset + ix].r -= ksum_r * cos(phase);
                tmp_RY[offset + ix].i -= ksum_i * cos(phase);
            }
        }
    }


    //y境界点での値がゼロになるように補正を加える
    for (int iz = 0; iz < sub_size_z; ++iz) {
        for (int ix = 0; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;

            const int num_edge = 1 + ((iz == 0) ? 1 : 0);
            if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                ksum_r /= (double)num_edge;
                ksum_i /= (double)num_edge;
            }

            double k = (double)(sub_size_y / 2);
            for (int iy = 0; iy < sub_size_y; ++iy) {
                double phase = 2.0 * M_PI * k * (double)iy / (double)sub_size_y;
                tmp_RY[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
                tmp_RY[offset + sub_size_x * iy].i -= ksum_i * cos(phase);;
            }

        }
    }


    //z境界点での値がゼロになるように補正を加える
    for (int iy = 0; iy < sub_size_y; ++iy) {
        for (int ix = 0; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;
            double k = (double)(sub_size_z / 2);
            for (int iz = 0; iz < sub_size_z; ++iz) {
                double phase = 2.0 * M_PI * k * (double)iz / (double)sub_size_z;
                tmp_RY[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
                tmp_RY[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
            }
        }
    }

    const double invN = 1.0 / (double)sub_size;
    for (int i = 0; i < sub_size; ++i) {
        tmp_RY[i].r *= invN;
        tmp_RY[i].i *= invN;
    }
    fftw->ForwardDirect(tmp_RY);
    


}


/*
* 境界点での値がゼロになるように補正を加える
* ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
*/
inline
void CorrectEdgeZeroByWave_2(OneComplex* tmp_RY,
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z, const char* print_path) {

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;
    const double invN = 1.0 / (double)sub_size;


    FILE* fp = nullptr;
#ifdef _DEBUG
    if (print_path) {
        fp = fopen(print_path, "w");
    }
#endif


    fftw->BackwardDirect(tmp_RY);

#if _DEBUG
    if (fp) {
        fprintf(fp, "total: x-y (z =half)\n");
        const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
            }
            fprintf(fp, "\n");
        }
    }
#endif

    //x境界点での値がゼロになるように補正を加える
    for (int iz = 0; iz < sub_size_z; ++iz) {
        for (int iy = 0; iy < sub_size_y; ++iy) {
            const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
            double ksum_r = tmp_RY[offset ].r;
            double ksum_i = tmp_RY[offset ].i;
            const int num_edge = 1 + ((iy == 0) ? 1 : 0) + ((iz == 0) ? 1 : 0);
            if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                ksum_r /= (double)num_edge;
                ksum_i /= (double)num_edge;
            }

            double k = (double)(sub_size_x / 2);
            for (int ix = 0; ix < sub_size_x; ++ix) {
                double phase = 2.0 * M_PI * k * (double)(ix ) / (double)sub_size_x;
                tmp_RY[offset + ix].r -= ksum_r * cos(phase);
                tmp_RY[offset + ix].i -= ksum_i * cos(phase);
            }
        }
    }


    //y境界点での値がゼロになるように補正を加える
    for (int iz = 0; iz < sub_size_z; ++iz) {
        for (int ix = 0; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
            double ksum_r = tmp_RY[offset ].r;
            double ksum_i = tmp_RY[offset ].i;

            const int num_edge = 1 + ((iz == 0) ? 1 : 0);
            if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                ksum_r /= (double)num_edge;
                ksum_i /= (double)num_edge;
            }

            double k = (double)(sub_size_y / 2);
            for (int iy = 0; iy < sub_size_y; ++iy) {
                double phase = 2.0 * M_PI * k * (double)(iy) / (double)sub_size_y;
                tmp_RY[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
                tmp_RY[offset + sub_size_x * iy].i -= ksum_i * cos(phase);
            }

        }
    }


    //z境界点での値がゼロになるように補正を加える
    for (int iy = 0; iy < sub_size_y; ++iy) {
        for (int ix = 0; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
            double ksum_r = tmp_RY[offset ].r;
            double ksum_i = tmp_RY[offset ].i;
            double k = (double)(sub_size_z / 2);
            for (int iz = 0; iz < sub_size_z; ++iz) {
                double phase = 2.0 * M_PI * k * (double)(iz) / (double)sub_size_z;
                tmp_RY[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
                tmp_RY[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
            }
        }
    }

#ifdef _DEBUG
    if (fp) {
        fprintf(fp, "(corrected)total: x-y (z =half)\n");
        const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
#endif

    for (int i = 0; i < sub_size; ++i) {
        tmp_RY[i].r *= invN;
        tmp_RY[i].i *= invN;
    }
    fftw->ForwardDirect(tmp_RY);



}


/*
* 境界点での値がゼロになるように補正を加える
* ただし、tmp_RYはkspaceへFFTされた後の場を受け取る
*/
#define CANCEL_WITH_IMAGINARY
//#define CANCEL_TYPE_ZERO
//#define CANCEL_TYPE_ONE
//#define CANCEL_TYPE_HALF
//#define CANCEL_TYPE_N_MINUS_1
inline
void CorrectEdgeZeroByWave_3(OneComplex* tmp_RY,
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z, const char* print_path) {

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;
    const double invN = 1.0 / (double)sub_size;



    FILE* fp = nullptr;
#ifdef _DEBUG
    if (print_path) {
        fp = fopen(print_path, "w");
    }
#endif


    fftw->BackwardDirect(tmp_RY);

#if _DEBUG
    double sum_org = 0.0;
    if (fp) {
        fprintf(fp, "total: x-y (z =half)\n");
        const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "Im: x-y (z =half)\n");
        //const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].i * invN);
            }
            fprintf(fp, "\n");
        }

        for (int i = 0; i < sub_size; ++i) {
            sum_org += tmp_RY[i].r;
        }
        sum_org *= dx * dy * dz * invN;
    }
#endif

    //x境界点での値がゼロになるように補正を加える
    for (int iz = 1; iz < sub_size_z; ++iz) {
        for (int iy = 1; iy < sub_size_y; ++iy) {
            const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;
            
#ifdef CANCEL_TYPE_ONE
            double k = 1.0;
#elif defined(CANCEL_TYPE_ZERO)
            double k = 0.0;
#elif defined(CANCEL_TYPE_HALF)
            double k = (double)(sub_size_x / 4);
#elif defined(CANCEL_TYPE_N_MINUS_1)
            double k = (double)(sub_size_x / 2 - 1);
#else
            double k = (double)(sub_size_x / 2);
#endif
            for (int ix = 0; ix < sub_size_x; ++ix) {
                double phase = 2.0 * M_PI * k * (double)(ix) / (double)sub_size_x;
                tmp_RY[offset + ix].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                tmp_RY[offset + ix].i -= ksum_i * cos(phase);
#endif
            }
        }
    }


    //y境界点での値がゼロになるように補正を加える
    for (int iz = 1; iz < sub_size_z; ++iz) {
        for (int ix = 1; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;

#ifdef CANCEL_TYPE_ONE
            double k = 1.0;
#elif defined(CANCEL_TYPE_ZERO)
            double k = 0.0;
#elif defined(CANCEL_TYPE_HALF)
            double k = (double)(sub_size_y / 4);
#elif defined(CANCEL_TYPE_N_MINUS_1)
            double k = (double)(sub_size_y / 2 - 1);
#else
            double k = (double)(sub_size_y / 2);
#endif
            for (int iy = 0; iy < sub_size_y; ++iy) {
                double phase = 2.0 * M_PI * k * (double)(iy) / (double)sub_size_y;
                tmp_RY[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                tmp_RY[offset + sub_size_x * iy].i -= ksum_i * cos(phase);
#endif
            }

        }
    }


    //z境界点での値がゼロになるように補正を加える
    for (int iy = 1; iy < sub_size_y; ++iy) {
        for (int ix = 1; ix < sub_size_x; ++ix) {
            const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
            double ksum_r = tmp_RY[offset].r;
            double ksum_i = tmp_RY[offset].i;

#ifdef CANCEL_TYPE_ONE
            double k = 1.0;
#elif defined(CANCEL_TYPE_ZERO)
            double k = 0.0;
#elif defined(CANCEL_TYPE_HALF)
            double k = (double)(sub_size_z / 4);
#elif defined(CANCEL_TYPE_N_MINUS_1)
            double k = (double)(sub_size_z / 2 - 1);
#else
            double k = (double)(sub_size_z / 2);
#endif
            for (int iz = 0; iz < sub_size_z; ++iz) {
                double phase = 2.0 * M_PI * k * (double)(iz) / (double)sub_size_z;
                tmp_RY[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                tmp_RY[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
#endif
            }
        }
    }


    //立方体の辺を全て0で埋める
    for (int ix = 0; ix < sub_size_x; ++ix) {
        const int iy = 0;
        const int iz = 0;
        const int i = ix + sub_size_x * (iy + sub_size_y * iz);
        tmp_RY[i].r = 0.0;
        tmp_RY[i].i = 0.0;
    }
    for (int iy = 0; iy < sub_size_y; ++iy) {
        const int ix = 0;
        const int iz = 0;
        const int i = ix + sub_size_x * (iy + sub_size_y * iz);
        tmp_RY[i].r = 0.0;
        tmp_RY[i].i = 0.0;
    }
    for (int iz = 0; iz < sub_size_z; ++iz) {
        const int iy = 0;
        const int ix = 0;
        const int i = ix + sub_size_x * (iy + sub_size_y * iz);
        tmp_RY[i].r = 0.0;
        tmp_RY[i].i = 0.0;
    }


#ifdef _DEBUG
    if (fp) {
        fprintf(fp, "(corrected)total: x-y (z =half)\n");
        const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].r * invN);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "(corrected)Im: x-y (z =half)\n");
        //const int iz = sub_size_z / 2;
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                fprintf(fp, "%.7f  ", tmp_RY[i].i * invN);
            }
            fprintf(fp, "\n");
        }
        double sum_cor = 0.0;
        for (int i = 0; i < sub_size; ++i) {
            sum_cor += tmp_RY[i].r;
        }
        sum_cor *= dx * dy * dz * invN;

        fprintf(fp, "sum: %f -> %f\n", sum_org, sum_cor);

        fclose(fp);
    }
#endif

    for (int i = 0; i < sub_size; ++i) {
        tmp_RY[i].r *= invN;
        tmp_RY[i].i *= invN;
    }
    fftw->ForwardDirect(tmp_RY);



}


#if 0
//微分値がsubgridの端で0になるように補正する//
inline
void CorrectEdgeDifferenceZeroByWave(int proj_odd_flag, OneComplex* tmp_RY,
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z, const char* print_path) {

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;
    const double invN = 1.0 / (double)sub_size;
    const double coef1_x = (2.0 * M_PI / (dx * (double)sub_size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)sub_size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)sub_size_z));

    //auto grad_p = gy::make_unique_aligned<OneComplex[]>(sub_size);
    auto grad_p = gy::AlignedAlloc< OneComplex>(sub_size);

    FILE* fp = nullptr;
#ifdef _DEBUG
    if (print_path) {
        fp = fopen(print_path, "w");
    }
#endif



    //以降は微分値が端点で0になるように//
    if (proj_odd_flag & ODD_YLM_X)
        //if(false)
    {

//#define TEST_OUT_OF_PLACE
#ifdef TEST_OUT_OF_PLACE
        GradientXInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        

#else
        GradientXInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-x: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //x境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 0; iy < sub_size_y; ++iy) {
                const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;
                const int num_edge = 1 + ((iy == 0) ? 1 : 0) + ((iz == 0) ? 1 : 0);
                if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                    ksum_r /= (double)num_edge;
                    ksum_i /= (double)num_edge;
                }

                double k = (double)(sub_size_x / 2);
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    double phase = 2.0 * M_PI * k * (double)(ix) / (double)sub_size_x;
                    grad_p[offset + ix].r -= ksum_r * cos(phase);
                    grad_p[offset + ix].i -= ksum_i * cos(phase);
                }
            }
        }


#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-x: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }

        fftw->ForwardExecute(&grad_p[0], tmp_RY);
        

        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 1; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    const double kx1 = (coef1_x * (double)(ix * 2 > sub_size_x ? ix - sub_size_x : ix));
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / kx1;
                    tmp_RY[i].i = -ore / kx1;
                }
            }
        }
    }

    if (proj_odd_flag & ODD_YLM_Y)
        //if (false)
    {
#ifdef TEST_OUT_OF_PLACE
        GradientYInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        
#else
        GradientYInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-y: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //y境界点での値がゼロになるように補正を加える
        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;

                const int num_edge = 1 + ((iz == 0) ? 1 : 0);
                if (num_edge > 1) {  //2方向以上での端点(cubicの辺や頂点)の場合は各方向での補正にも一部任せる//
                    ksum_r /= (double)num_edge;
                    ksum_i /= (double)num_edge;
                }

                double k = (double)(sub_size_y / 2);
                for (int iy = 0; iy < sub_size_y; ++iy) {
                    double phase = 2.0 * M_PI * k * (double)(iy) / (double)sub_size_y;
                    grad_p[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
                    grad_p[offset + sub_size_x * iy].i -= ksum_i * cos(phase);
                }

            }
        }



#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-y: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }

        fftw->ForwardExecute(&grad_p[0], tmp_RY);
        


        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 1; iy < sub_size_y; ++iy) {
                const double ky1 = (coef1_y * (double)(iy * 2 > sub_size_y ? iy - sub_size_y : iy));

                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / ky1;
                    tmp_RY[i].i = -ore / ky1;
                }
            }
        }
    }

    if (proj_odd_flag & ODD_YLM_Z)
        //if (false)
    {
#ifdef TEST_OUT_OF_PLACE
        GradientZInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        
#else
        GradientZInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-z: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //z境界点での値がゼロになるように補正を加える
        for (int iy = 0; iy < sub_size_y; ++iy) {
            for (int ix = 0; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;
                double k = (double)(sub_size_z / 2);
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    double phase = 2.0 * M_PI * k * (double)(iz) / (double)sub_size_z;
                    grad_p[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
                    grad_p[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
                }
            }
        }

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-z: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }
        fftw->ForwardExecute(&grad_p[0], tmp_RY);
        

        for (int iz = 1; iz < sub_size_z; ++iz) {
            const double kz1 = (coef1_z * (double)(iz * 2 > sub_size_z ? iz - sub_size_z : iz));

            for (int iy = 0; iy < sub_size_y; ++iy) {

                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / kz1;
                    tmp_RY[i].i = -ore / kz1;
                }
            }
        }

    }

    gy::AlignedFree(grad_p);

    if (fp) {
        fclose(fp);
    }



}
#endif

//#define CANCEL_DIFF_TYPE_ONE   3
//#define CANCEL_DIFF_TYPE_HALF
//#define CANCEL_DIFF_TYPE_N_MINUS_1
//微分値がsubgridの端で0になるように補正する//
inline
void CorrectEdgeDifferenceZeroByWave_2(int proj_odd_flag, OneComplex* tmp_RY,
    FFTW_Executor* fftw,
    double dx, double dy, double dz,
    int sub_size_x, int sub_size_y, int sub_size_z, const char* print_path) {

    const int sub_size = sub_size_x * sub_size_y * sub_size_z;
    const double invN = 1.0 / (double)sub_size;
    const double coef1_x = (2.0 * M_PI / (dx * (double)sub_size_x));
    const double coef1_y = (2.0 * M_PI / (dy * (double)sub_size_y));
    const double coef1_z = (2.0 * M_PI / (dz * (double)sub_size_z));
    FILE* fp = nullptr;
#ifdef _DEBUG
    if (print_path) {
        fp = fopen(print_path, "w");
    }
#endif



//#define TEST_OUT_OF_PLACE
#define TEST_IN_PLACE




#ifdef TEST_IN_PLACE
    //memcpy(grad_p, tmp_RY, sizeof(OneComplex)* sub_size);
    //OneComplex* keep = tmp_RY;
    auto grad_p = tmp_RY;
#else
    //auto grad_p = std::make_unique<OneComplex[]>(sub_size);
    //auto grad_p = gy::make_unique_aligned<OneComplex[]>(sub_size);
    auto grad_p = gy::AlignedAlloc< OneComplex>(sub_size);
#endif



    //以降は微分値が端点で0になるように//
    if (proj_odd_flag & ODD_YLM_X)
        //if(false)
    {


#ifdef TEST_OUT_OF_PLACE
        GradientXInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        
#elif defined(TEST_IN_PLACE)
        GradientXInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);
        
#else
        GradientXInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-x: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }

            fprintf(fp, "Im(diff-x): x-y (z =half)\n");
            //const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //x境界点での値がゼロになるように補正を加える
        for (int iz = 1; iz < sub_size_z; ++iz) {
            for (int iy = 1; iy < sub_size_y; ++iy) {
                const int64_t offset = sub_size_x * (iy + sub_size_y * iz);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;
#ifdef CANCEL_DIFF_TYPE_ONE
                double k = CANCEL_DIFF_TYPE_ONE;
#elif defined(CANCEL_DIFF_TYPE_HALF)
                double k = (double)(sub_size_x / 4);                
#elif defined(CANCEL_DIFF_TYPE_N_MINUS_1)
                double k = (double)(sub_size_x / 2 - 1);
#else
                double k = (double)(sub_size_x / 2);
#endif
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    double phase = 2.0 * M_PI * k * (double)(ix) / (double)sub_size_x;
                    grad_p[offset + ix].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                    grad_p[offset + ix].i -= ksum_i * cos(phase);
#endif
                }
            }
        }


#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-x: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "(corrected)Im(diff-x): x-y (z =half)\n");
            //const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }
#ifdef TEST_IN_PLACE
        fftw->ForwardDirect(tmp_RY);
        
#else
        fftw->ForwardExecute(&grad_p[0], tmp_RY);
        
#endif

        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 1; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    const double kx1 = (coef1_x * (double)(ix * 2 > sub_size_x ? ix - sub_size_x : ix));
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / kx1;
                    tmp_RY[i].i = -ore / kx1;
                }
            }
        }
    }

    if (proj_odd_flag & ODD_YLM_Y)
        //if (false)
    {
#ifdef TEST_OUT_OF_PLACE
        GradientYInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        
#elif defined(TEST_IN_PLACE)
        GradientYInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);
        
#else
        GradientYInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-y: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "Im(diff-y): x-y (z =half)\n");
            //const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //y境界点での値がゼロになるように補正を加える
        for (int iz = 1; iz < sub_size_z; ++iz) {
            for (int ix = 1; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (0 + sub_size_y * iz);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;
#ifdef CANCEL_DIFF_TYPE_ONE
                double k = CANCEL_DIFF_TYPE_ONE;
#elif defined(CANCEL_DIFF_TYPE_HALF)
                double k = (double)(sub_size_y / 4);
#elif defined(CANCEL_DIFF_TYPE_N_MINUS_1)
                double k = (double)(sub_size_y / 2 - 1);
#else
                double k = (double)(sub_size_y / 2);
#endif
                for (int iy = 0; iy < sub_size_y; ++iy) {
                    double phase = 2.0 * M_PI * k * (double)(iy) / (double)sub_size_y;
                    grad_p[offset + sub_size_x * iy].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                    grad_p[offset + sub_size_x * iy].i -= ksum_i * cos(phase);
#endif
                }

            }
        }



#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-y: x-y (z =half)\n");
            const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "(corrected)Im(diff-y): x-y (z =half)\n");
            //const int iz = sub_size_z / 2;
            for (int iy = 0; iy < sub_size_y; ++iy) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }

#ifdef TEST_IN_PLACE
        fftw->ForwardDirect(tmp_RY);
        
#else
        fftw->ForwardExecute(&grad_p[0], tmp_RY);
        
#endif

        for (int iz = 0; iz < sub_size_z; ++iz) {
            for (int iy = 1; iy < sub_size_y; ++iy) {
                const double ky1 = (coef1_y * (double)(iy * 2 > sub_size_y ? iy - sub_size_y : iy));

                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / ky1;
                    tmp_RY[i].i = -ore / ky1;
                }
            }
        }
    }

    if (proj_odd_flag & ODD_YLM_Z)
        //if (false)
    {
#ifdef TEST_OUT_OF_PLACE
        GradientZInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardExecute(tmp_RY, &grad_p[0]);
        
#elif defined(TEST_IN_PLACE)
        GradientZInKspace_inplace(tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(tmp_RY);
        
#else
        GradientZInKspace(&grad_p[0], tmp_RY, sub_size_x, sub_size_y, sub_size_z, dx, dy, dz);
        fftw->BackwardDirect(&grad_p[0]);
        
#endif

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "diff-z: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "Im(diff-z): x-z (y =half)\n");
            //const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        //z境界点での値がゼロになるように補正を加える
        for (int iy = 1; iy < sub_size_y; ++iy) {
            for (int ix = 1; ix < sub_size_x; ++ix) {
                const int64_t offset = ix + sub_size_x * (iy + sub_size_y * 0);
                double ksum_r = grad_p[offset].r;
                double ksum_i = grad_p[offset].i;

#ifdef CANCEL_DIFF_TYPE_ONE
                double k = CANCEL_DIFF_TYPE_ONE;
#elif defined(CANCEL_DIFF_TYPE_HALF)
                double k = (double)(sub_size_z / 4);
#elif defined(CANCEL_DIFF_TYPE_N_MINUS_1)
                double k = (double)(sub_size_z / 2 - 1);
#else
                double k = (double)(sub_size_z / 2);
#endif
                for (int iz = 0; iz < sub_size_z; ++iz) {
                    double phase = 2.0 * M_PI * k * (double)(iz) / (double)sub_size_z;
                    grad_p[offset + sub_size_x * sub_size_y * iz].r -= ksum_r * cos(phase);
#ifdef CANCEL_WITH_IMAGINARY
                    grad_p[offset + sub_size_x * sub_size_y * iz].i -= ksum_i * cos(phase);
#endif
                }
            }
        }

#ifdef _DEBUG
        if (fp) {
            fprintf(fp, "(corrected)diff-z: x-z (y =half)\n");
            const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].r * invN);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "(corrected)Im(diff-z): x-z (y =half)\n");
            //const int iy = sub_size_y / 2;
            for (int iz = 0; iz < sub_size_z; ++iz) {
                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    fprintf(fp, "%.7f  ", grad_p[i].i * invN);
                }
                fprintf(fp, "\n");
            }
        }
#endif

        for (int i = 0; i < sub_size; ++i) {
            grad_p[i].r *= invN;
            grad_p[i].i *= invN;
        }

#ifdef TEST_IN_PLACE
        fftw->ForwardDirect(tmp_RY);
#else
        fftw->ForwardExecute(&grad_p[0], tmp_RY);
#endif
        

        for (int iz = 1; iz < sub_size_z; ++iz) {
            const double kz1 = (coef1_z * (double)(iz * 2 > sub_size_z ? iz - sub_size_z : iz));

            for (int iy = 0; iy < sub_size_y; ++iy) {

                for (int ix = 0; ix < sub_size_x; ++ix) {
                    const int64_t i = ix + sub_size_x * (iy + sub_size_y * iz);
                    double ore = tmp_RY[i].r;
                    double oim = tmp_RY[i].i;
                    tmp_RY[i].r = oim / kz1;
                    tmp_RY[i].i = -ore / kz1;
                }
            }
        }

    }

#ifdef TEST_IN_PLACE
    //tmp_RY = keep;
    //memcpy(tmp_RY, grad_p, sizeof(OneComplex) * sub_size);
#else

    gy::AlignedFree(grad_p);
#endif


    if (fp) {
        fclose(fp);
    }



}
