#pragma once
#include <cstdint>
#include <cstring>


template<class T>
void UpConvert_Kspace_1d(const T* src, int size_x, T* hr_dest, int hr_ratio_x) {

    //正の領域半分と負の領域半分で2度に分けて焼き直し//
    const int ix_half = (size_x + 1) / 2;
    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//

    for (int ix = 0; ix < ix_half; ++ix) {
        const int oi = ix;
        hr_dest[oi] = src[ix];
    }
    //memset(&hr_dest[ix_half], 0, sizeof(T) * (size_x * (hr_ratio_x - 1)));
    for (int oi = ix_half; oi < ix_half + size_x * (hr_ratio_x - 1); ++oi){
        hr_dest[oi] = 0.0;
    }

    if ((size_x & 0x1)==0) {//even number
        /*
        偶数の場合はN/2の成分を二カ所に分けて入力しないと
        実数場にもかかわらずUpConvert後に虚数成分が乗ってしまう

        また、FFTした場合、元(低解像度側)のN/2の成分は要素が1つしかないので2倍の値が入っている仕様になっている

        前提としてN/2の要素の虚数成分は0のはず(実数場でなくてもNが偶数なら).
        そうでなければhr_dest[oi]には複素共役を入れる必要がある
        */
        const int ix = ix_half;
        const int oi = size_x * (hr_ratio_x - 1) + ix;
        hr_dest[oi] = hr_dest[ix] = src[ix] * 0.5;
    }
    
    
    for (int ix = ix_half2; ix < size_x; ++ix) {
        const int oi = size_x * (hr_ratio_x - 1) + ix;
        hr_dest[oi] = src[ix];
    }
    

}



template<class T>
void UpConvert_Kspace(const T* src, int size_x, int size_y, int size_z, T* hr_dest, int hr_ratio_x, int hr_ratio_y, int hr_ratio_z) {

    memset(hr_dest, 0, sizeof(T) * (int64_t)size_x * (int64_t)size_y * (int64_t)size_z * (int64_t)hr_ratio_x * (int64_t)hr_ratio_y * (int64_t)hr_ratio_z);

    const int ix_half = (size_x + 1) / 2;
    const int iy_half = (size_y + 1) / 2;
    const int iz_half = (size_z + 1) / 2;
    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//
    const int iy_half2 = size_y / 2 + 1;
    const int iz_half2 = size_z / 2 + 1;

    for (int iz = 0; iz < iz_half; ++iz) {
        const int64_t oiz = iz * hr_ratio_x * size_x * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_ratio_x * size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
        if ((size_y & 0x1) == 0) {//even number
            const int iy = iy_half;
            for (int64_t ky = 0; ky < 2; ++ky) {
                const int64_t oiy = (size_y * (hr_ratio_y - 1) * ky + iy) * hr_ratio_x * size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi] = src[i]*0.5;
                }
            }
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
    }
    {
        const int64_t iz = iz_half;
        for (int64_t kz = 0; kz < 2; ++kz) {

            const int64_t oiz = (size_z * (hr_ratio_z - 1) * kz + iz) * hr_ratio_x * size_x * hr_ratio_y * size_y;
            for (int iy = 0; iy < iy_half; ++iy) {
                const int64_t oiy = iy * hr_ratio_x * size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i]*0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
            if ((size_y & 0x1) == 0) {//even number
                const int iy = iy_half;
                for (int64_t ky = 0; ky < 2; ++ky) {
                    const int64_t oiy = (size_y * (hr_ratio_y - 1) * ky + iy) * hr_ratio_x * size_x;
                    for (int ix = 0; ix < ix_half; ++ix) {
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = ix + oiy + oiz;
                        hr_dest[oi] = src[i] * 0.25;
                    }
                    if ((size_x & 0x1) == 0) {//even number
                        const int ix = ix_half;
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = ix + oiy + oiz;
                        const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                        hr_dest[oi2] = hr_dest[oi] = src[i] * 0.125;
                    }
                    for (int ix = ix_half2; ix < size_x; ++ix) {
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                        hr_dest[oi] = src[i] * 0.25;
                    }
                }
            }
            for (int iy = iy_half2; iy < size_y; ++iy) {
                const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
        }
    }
    for (int iz = iz_half2; iz < size_z; ++iz) {
        const int64_t oiz = (size_z* (hr_ratio_z-1) + iz) * hr_ratio_x * size_x * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_ratio_x * size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
        if ((size_y & 0x1) == 0) {//even number
            const int iy = iy_half;
            for (int64_t ky = 0; ky < 2; ++ky) {
                const int64_t oiy = (size_y * (hr_ratio_y - 1) * ky + iy) * hr_ratio_x * size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
    }
}


template<class T>
void UpConvert_Kspace_1d_x_any(const T* src, int size_x, T* hr_dest, int new_size_x) {

    //正の領域半分と負の領域半分で2度に分けて焼き直し//
    const int ix_half = (size_x + 1) / 2;
    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//

    for (int ix = 0; ix < ix_half; ++ix) {
        const int oi = ix;
        hr_dest[oi] = src[ix];
    }
    
    for (int oi = ix_half; oi < ix_half + new_size_x - size_x; ++oi) {
        hr_dest[oi] = 0.0;
    }

    if ((size_x & 0x1) == 0) {//even number
        /*
        偶数の場合はN/2の成分を二カ所に分けて入力しないと
        実数場にもかかわらずUpConvert後に虚数成分が乗ってしまう

        また、FFTした場合、元(低解像度側)のN/2の成分は要素が1つしかないので2倍の値が入っている仕様になっている

        前提としてN/2の要素の虚数成分は0のはず(実数場でなくてもNが偶数なら).
        そうでなければhr_dest[oi]には複素共役を入れる必要がある
        */
        const int ix = ix_half;
        const int oi = new_size_x - size_x + ix;
        hr_dest[oi] = hr_dest[ix] = src[ix] * 0.5;
    }


    for (int ix = ix_half2; ix < size_x; ++ix) {
        const int oi = new_size_x - size_x + ix;
        hr_dest[oi] = src[ix];
    }

}


template<class T>
void UpConvert_Kspace_1d_y_any(const T* src, int stride, int size_y, T* hr_dest, int new_size_y) {

    //正の領域半分と負の領域半分で2度に分けて焼き直し//
    const int iy_half = (size_y + 1) / 2;
    const int iy_half2 = size_y / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//

    for (int iy = 0; iy < iy_half; ++iy) {
        const int oi = iy;
        for (int ix = 0; ix < stride; ++ix) {
            hr_dest[oi* stride + ix] = src[iy * stride + ix];
        }
    }

    for (int oi = iy_half; oi < iy_half + new_size_y - size_y; ++oi) {
        for (int ix = 0; ix < stride; ++ix) {
            hr_dest[oi * stride + ix] = 0.0;
        }
    }

    if ((size_y & 0x1) == 0) {//even number
        /*
        偶数の場合はN/2の成分を二カ所に分けて入力しないと
        実数場にもかかわらずUpConvert後に虚数成分が乗ってしまう

        また、FFTした場合、元(低解像度側)のN/2の成分は要素が1つしかないので2倍の値が入っている仕様になっている

        前提としてN/2の要素の虚数成分は0のはず(実数場でなくてもNが偶数なら).
        そうでなければhr_dest[oi]には複素共役を入れる必要がある
        */
        const int iy = iy_half;
        const int oi = new_size_y - size_y + iy;
        for (int ix = 0; ix < stride; ++ix) {
            hr_dest[oi* stride + ix] = hr_dest[iy * stride + ix] = src[iy * stride + ix] * 0.5;
        }
    }


    for (int iy = iy_half2; iy < size_y; ++iy) {
        const int oi = new_size_y - size_y + iy;
        for (int ix = 0; ix < stride; ++ix) {
            hr_dest[oi * stride + ix] = src[iy * stride + ix];
        }
    }

}


template<class T>
void UpConvert_Kspace_3d_any(const T* src, int size_x, int size_y, int size_z, T* hr_dest, int hr_size_x, int hr_size_y, int hr_size_z) {

    memset(hr_dest, 0, sizeof(T) * (int64_t)hr_size_x * (int64_t)hr_size_y * (int64_t)hr_size_z);

    const int ix_half = (size_x + 1) / 2;
    const int iy_half = (size_y + 1) / 2;
    const int iz_half = (size_z + 1) / 2;
    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//
    const int iy_half2 = size_y / 2 + 1;
    const int iz_half2 = size_z / 2 + 1;

    for (int iz = 0; iz < iz_half; ++iz) {
        const int64_t oiz = iz * hr_size_x * hr_size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
        if ((size_y & 0x1) == 0) {//even number
            const int iy = iy_half;
            for (int64_t ky = 0; ky < 2; ++ky) {
                const int64_t oiy = ((hr_size_y - size_y) * ky + iy) * hr_size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = ((hr_size_y - size_y) + iy) * hr_size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
    }
    {
        const int64_t iz = iz_half;
        for (int64_t kz = 0; kz < 2; ++kz) {

            const int64_t oiz = ((hr_size_z - size_z) * kz + iz) * hr_size_x * hr_size_y;
            for (int iy = 0; iy < iy_half; ++iy) {
                const int64_t oiy = iy * hr_size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 =  (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
            if ((size_y & 0x1) == 0) {//even number
                const int iy = iy_half;
                for (int64_t ky = 0; ky < 2; ++ky) {
                    const int64_t oiy = ( (hr_size_y - size_y) * ky + iy) * hr_size_x;
                    for (int ix = 0; ix < ix_half; ++ix) {
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = ix + oiy + oiz;
                        hr_dest[oi] = src[i] * 0.25;
                    }
                    if ((size_x & 0x1) == 0) {//even number
                        const int ix = ix_half;
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = ix + oiy + oiz;
                        const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                        hr_dest[oi2] = hr_dest[oi] = src[i] * 0.125;
                    }
                    for (int ix = ix_half2; ix < size_x; ++ix) {
                        int64_t i = ix + size_x * (iy + size_y * iz);
                        const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                        hr_dest[oi] = src[i] * 0.25;
                    }
                }
            }
            for (int iy = iy_half2; iy < size_y; ++iy) {
                const int64_t oiy = ((hr_size_y - size_y) + iy) * hr_size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
        }
    }
    for (int iz = iz_half2; iz < size_z; ++iz) {
        const int64_t oiz = ( (hr_size_z - size_z) + iz) * hr_size_x * hr_size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
        if ((size_y & 0x1) == 0) {//even number
            const int iy = iy_half;
            for (int64_t ky = 0; ky < 2; ++ky) {
                const int64_t oiy = ((hr_size_y - size_y) * ky + iy) * hr_size_x;
                for (int ix = 0; ix < ix_half; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
                if ((size_x & 0x1) == 0) {//even number
                    const int ix = ix_half;
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = ix + oiy + oiz;
                    const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi2] = hr_dest[oi] = src[i] * 0.25;
                }
                for (int ix = ix_half2; ix < size_x; ++ix) {
                    int64_t i = ix + size_x * (iy + size_y * iz);
                    const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                    hr_dest[oi] = src[i] * 0.5;
                }
            }
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = ( (hr_size_y - size_y) + iy) * hr_size_x;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            if ((size_x & 0x1) == 0) {//even number
                const int ix = ix_half;
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                const int64_t oi2 = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi2] = hr_dest[oi] = src[i] * 0.5;
            }
            for (int ix = ix_half2; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = (hr_size_x - size_x) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
        }
    }
}


/*
* k=N/2の処理が入っていないのでdeprecated
* [deprecated]
*/
template<class T>
void UpConvert_Kspace_r2c(const T* src, int size_x, int size_y, int size_z, T* hr_dest, int hr_ratio_x, int hr_ratio_y, int hr_ratio_z) {

    const int stride_cx = size_x / 2 + 1;
    const int hr_stride_cx = (size_x * hr_ratio_x) / 2 + 1;

    memset(hr_dest, 0, sizeof(T) * (int64_t)hr_stride_cx * (int64_t)size_y * (int64_t)size_z * (int64_t)hr_ratio_y * (int64_t)hr_ratio_z);


    const int ix_half = (size_x + 1) / 2;
    const int iy_half = (size_y + 1) / 2;
    const int iz_half = (size_z + 1) / 2;

    for (int iz = 0; iz < iz_half; ++iz) {
        const int64_t oiz = iz * hr_stride_cx * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_stride_cx;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + stride_cx * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            /*
            for (int ix = ix_half; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            */
        }
        for (int iy = iy_half; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_stride_cx;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + stride_cx * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            /*
            for (int ix = ix_half; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            */
        }
    }
    for (int iz = iz_half; iz < size_z; ++iz) {
        const int64_t oiz = (size_z * (hr_ratio_z - 1) + iz) * hr_stride_cx * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_stride_cx;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + stride_cx * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            /*
            for (int ix = ix_half; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            */
        }
        for (int iy = iy_half; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_stride_cx;
            for (int ix = 0; ix < ix_half; ++ix) {
                int64_t i = ix + stride_cx * (iy + size_y * iz);
                const int64_t oi = ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            /*
            for (int ix = ix_half; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
                hr_dest[oi] = src[i];
            }
            */
        }
    }
}


template<class T>
void DownConvert_Realspace(T* dest, int size_x, int size_y, int size_z, const T* hr_src, int hr_ratio_x, int hr_ratio_y, int hr_ratio_z) {
    for (int iz = 0; iz < size_z; ++iz) {
        for (int iy = 0; iy < size_y; ++iy) {
            for (int ix = 0; ix < size_x; ++ix) {
                int64_t i = ix + size_x * (iy + size_y * iz);
                int64_t oi = hr_ratio_x * (ix + size_x * (hr_ratio_y * (iy + size_y * hr_ratio_z * iz)));
                dest[i] = hr_src[oi];
            }
        }
    }
}

//実空間上の同じ点のコピー(定数倍の点のcopy)ではなく、
//k空間でLow-pass filterとするため、
//実空間上の同じ点の値はむしろ合わない
//Nが偶数の場合に、k = N / 2の点は格納先が正側しかないが、
//読み取り先の正負の成分を合成するして格納する.
//そのとき、虚数成分はゼロにする必要がある//
template<class T>
void DownConvert_Kspace_1d(const T* hr_src, int size_x, T* dest, int hr_ratio_x) {


    const int ix_half = (size_x + 1) / 2;
    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//
    
    for (int ix = 0; ix < ix_half; ++ix) {
        const int oi = ix;
        dest[ix] = hr_src[oi];
    }

    //k=N/2の成分を2倍にする(FFTの仕様に準じて)
    if ((size_x & 0x1) == 0) {//even number//
        const int ix = ix_half;
        const int oi = size_x * (hr_ratio_x - 1) + ix;
        dest[ix] = hr_src[ix] + hr_src[oi];//ここで2倍になる//
        if constexpr (sizeof(T) == 16) {
            *((double*)(&dest[ix]) + 1) = 0.0;
        } else if constexpr (sizeof(T) == 8) {
            *((float*)(&dest[ix]) + 1) = 0.0;
        }
    }
    for (int ix = ix_half2; ix < size_x; ++ix) {
        const int oi = size_x * (hr_ratio_x - 1) + ix;
        dest[ix] = hr_src[oi];
    }
}

/*
* Nが偶数の場合に、k = N/2の波数があっても、k=-N/2の波数が存在しない問題が生じる
* そこで、この関数でk = N/2のモードはゼロで落とす
*/
template<class T>
void DownConvert_Kspace(const T* hr_src, int size_x, int size_y, int size_z, T* dest, int hr_ratio_x, int hr_ratio_y, int hr_ratio_z) {

    
    const int ix_half = (size_x + 1) / 2;
    const int iy_half = (size_y + 1) / 2;
    const int iz_half = (size_z + 1) / 2;

    const int ix_half2 = size_x / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//
    const int iy_half2 = size_y / 2 + 1;
    const int iz_half2 = size_z / 2 + 1;

    auto LoopX = [&](int syz, int oiy, int oiz, double weight) {
        for (int ix = 0; ix < ix_half; ++ix) {
            int64_t i = ix + syz;
            const int64_t oi = ix + oiy + oiz;
            dest[i] = hr_src[oi] * weight;
        }

        //k=N/2の成分を2倍にする(FFTの仕様に準じて)
        if ((size_x & 0x1) == 0) {//even number//
            const int ix = ix_half;
            int64_t i = ix + syz;
            const int64_t oi = ix + oiy + oiz;
            const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
            dest[i] = (hr_src[oi] + hr_src[oi2]) * weight;//ここで2倍になる//
            if constexpr (sizeof(T) == 16) {
                *((double*)(&dest[i]) + 1) = 0.0;
            } else if constexpr (sizeof(T) == 8) {
                *((float*)(&dest[i]) + 1) = 0.0;
            }
        }

        for (int ix = ix_half2; ix < size_x; ++ix) {
            int64_t i = ix + syz;
            const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
            dest[i] = hr_src[oi] * weight;
        }
    };

    auto LoopXAdd = [&](int syz, int oiy, int oiz, double weight) {
        for (int ix = 0; ix < ix_half; ++ix) {
            int64_t i = ix + syz;
            const int64_t oi = ix + oiy + oiz;
            dest[i] = dest[i] + hr_src[oi] * weight;
        }
        if ((size_x & 0x1) == 0) {//even number//
            const int ix = ix_half;
            int64_t i = ix + syz;
            const int64_t oi = ix + oiy + oiz;
            const int64_t oi2 = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
            dest[i] = dest[i] + (hr_src[oi] + hr_src[oi2]) * weight;
            if constexpr (sizeof(T) == 16) {
                *((double*)(&dest[i]) + 1) = 0.0;
            } else if constexpr (sizeof(T) == 8) {
                *((float*)(&dest[i]) + 1) = 0.0;
            }
        }

        for (int ix = ix_half2; ix < size_x; ++ix) {
            int64_t i = ix + syz;
            const int64_t oi = size_x * (hr_ratio_x - 1) + ix + oiy + oiz;
            dest[i] = dest[i] + hr_src[oi] * weight;
        }
        };

    const double weight = 1.0 / (double)(hr_ratio_x * hr_ratio_y * hr_ratio_z);


    for (int iz = 0; iz < iz_half; ++iz) {
        const int64_t oiz = iz * hr_ratio_x * size_x * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_ratio_x * size_x;
            LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
        }
        if ((size_y & 0x1) == 0) {
            const int iy = iy_half;
            const int64_t oiy = iy * hr_ratio_x * size_x;
            const int64_t oiy2 = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
            LoopXAdd(size_x * (iy + size_y * iz), oiy2, oiz, weight);
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
        }
    }
    if ((size_z & 0x1) == 0) {
        const int iz = iz_half;
        {
            const int64_t oiz = iz * hr_ratio_x * size_x * hr_ratio_y * size_y;
            for (int iy = 0; iy < iy_half; ++iy) {
                const int64_t oiy = iy * hr_ratio_x * size_x;
                LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
            }
            if ((size_y & 0x1) == 0) {
                const int iy = iy_half;
                const int64_t oiy = iy * hr_ratio_x * size_x;
                const int64_t oiy2 = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
                LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
                LoopXAdd(size_x * (iy + size_y * iz), oiy2, oiz, weight);
            }
            for (int iy = iy_half2; iy < size_y; ++iy) {
                const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
                LoopX(size_x * (iy + size_y * iz), oiy, oiz, weight);
            }
        }
        {
            const int64_t oiz2 = (size_z * (hr_ratio_z - 1) + iz) * hr_ratio_x * size_x * hr_ratio_y * size_y;
            for (int iy = 0; iy < iy_half; ++iy) {
                const int64_t oiy = iy * hr_ratio_x * size_x;
                LoopXAdd(size_x * (iy + size_y * iz), oiy, oiz2, weight);
            }
            if ((size_y & 0x1) == 0) {
                const int iy = iy_half;
                const int64_t oiy = iy * hr_ratio_x * size_x;
                const int64_t oiy2 = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
                LoopXAdd(size_x* (iy + size_y * iz), oiy, oiz2, weight);
                LoopXAdd(size_x* (iy + size_y * iz), oiy2, oiz2, weight);
            }
            for (int iy = iy_half2; iy < size_y; ++iy) {
                const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
                LoopXAdd(size_x* (iy + size_y * iz), oiy, oiz2, weight);
            }
        }
    }
    for (int iz = iz_half2; iz < size_z; ++iz) {
        const int64_t oiz = (size_z * (hr_ratio_z - 1) + iz) * hr_ratio_x * size_x * hr_ratio_y * size_y;
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oiy = iy * hr_ratio_x * size_x;
            LoopX(size_x* (iy + size_y * iz), oiy, oiz, weight);
        }
        if ((size_y & 0x1) == 0) {
            const int iy = iy_half;
            const int64_t oiy = iy * hr_ratio_x * size_x;
            const int64_t oiy2 = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            LoopX(size_x* (iy + size_y * iz), oiy, oiz, weight);
            LoopXAdd(size_x* (iy + size_y * iz), oiy2, oiz, weight);
        }
        for (int iy = iy_half2; iy < size_y; ++iy) {
            const int64_t oiy = (size_y * (hr_ratio_y - 1) + iy) * hr_ratio_x * size_x;
            LoopX(size_x* (iy + size_y * iz), oiy, oiz, weight);
        }
    }
}


template<class T>
void CopyRect(const T* src, int src_x, int src_y, int src_z, 
    int end_x, int end_y, int end_z,
    int stride_x, int stride_y, int stride_z,
    T* dest, int dest_x, int dest_y, int dest_z,
    int stride_x2, int stride_y2, int stride_z2) {

    const int wx = end_x - src_x;
    const int wy = end_y - src_y;
    const int wz = end_z - src_z;

    for (int iz = 0; iz < wz; ++iz) {
        for (int iy = 0; iy < wy; ++iy) {
            for (int ix = 0; ix < wx; ++ix) {
                dest[ix + dest_x + stride_x2 * (iy + dest_y + stride_y2 * (iz + dest_z))] =
                    src[ix + src_x + stride_x * (iy + src_y + stride_y * (iz + src_z))];
            }
        }
    }
}


template<class T>
void UpDownConvert_Kspace_3d_any(const T* src, int size_x, int size_y, int size_z,
    T* hr_dest, int hr_size_x, int hr_size_y, int hr_size_z) {


    const int ix_half = (std::min(size_x, hr_size_x) + 1) / 2;
    const int iy_half = (std::min(size_y, hr_size_y) + 1) / 2;
    const int iz_half = (std::min(size_z, hr_size_z) + 1) / 2;
    //const int ix_half2 = std::min(size_x, hr_size_x) / 2 + 1;  //奇数ならix_halfと等しい.偶数ならix_half+1と等しい//
    //const int iy_half2 = std::min(size_y, hr_size_y) / 2 + 1;
    //const int iz_half2 = std::min(size_z, hr_size_z) / 2 + 1;

    memset(hr_dest, 0, sizeof(T) * hr_size_x * hr_size_y * hr_size_z);


    auto LoopX = [&](const T* src, int size_x, T* hr_dest, int hr_size_x) {
        for (int ix = 0; ix < ix_half; ++ix) {
            const int64_t oi = ix;
            hr_dest[oi] = src[ix];
        }

        if (size_x < hr_size_x) {
            if ((size_x & 0x1) == 0) {
                const int ix = ix_half;
                const int oi = hr_size_x - size_x + ix;
                hr_dest[oi] = hr_dest[ix] = src[ix] * 0.5;
            }
        } else if (size_x > hr_size_x) {
            if ((hr_size_x & 0x1) == 0) {
                //k=N/2の成分を2倍にする(FFTの仕様に準じて)
                const int64_t ix = ix_half;
                const int64_t oi = ix;
                const int64_t ix2 = size_x - ix_half;
                hr_dest[oi] = (src[ix] + src[ix2]);//ここで2倍になる//
                if constexpr (sizeof(T) == 16) {
                    *((double*)(&hr_dest[oi]) + 1) = 0.0;
                } else if constexpr (sizeof(T) == 8) {
                    *((float*)(&hr_dest[oi]) + 1) = 0.0;
                }
            }
        } else {
            const int64_t ix = ix_half;
            const int64_t oi = ix;
            hr_dest[oi] = src[ix];
        }

        for (int ix = size_x-ix_half + 1; ix < size_x; ++ix) {
            const int64_t oi = hr_size_x - size_x + ix;
            hr_dest[oi] = src[ix];
        }

        };

    auto tmp_x = std::make_unique<T[]>(hr_size_x);
    auto LoopY = [&](const T* src, int size_y, T* hr_dest, int hr_size_y) {
        for (int iy = 0; iy < iy_half; ++iy) {
            const int64_t oi = iy;
            LoopX(src + size_x * iy, size_x, hr_dest + hr_size_x * oi, hr_size_x);
        }

        if (size_y < hr_size_y) {
            if ((size_y & 0x1) == 0) {
                const int iy = iy_half;
                const int oi = hr_size_y - size_y + iy;

                memset(tmp_x.get(), 0, sizeof(T) * hr_size_x);
                
                LoopX(src + size_x * iy, size_x, tmp_x.get(), hr_size_x);
                
                for (int ix = 0; ix < hr_size_x; ++ix) {
                    tmp_x[ix] = tmp_x[ix] * 0.5;
                }
                memcpy(hr_dest + hr_size_x * iy, tmp_x.get(), sizeof(T) * hr_size_x);
                memcpy(hr_dest + hr_size_x * oi, tmp_x.get(), sizeof(T) * hr_size_x);
            }
        } else if (size_y > hr_size_y) {
            if ((hr_size_y & 0x1) == 0) {
                //k=N/2の成分を2倍にする(FFTの仕様に準じて)
                const int64_t iy = iy_half;
                const int64_t oi = iy;
                const int64_t iy2 = size_y - iy_half;

                memset(tmp_x.get(), 0, sizeof(T) * hr_size_x);
                LoopX(src + size_x * iy2, size_x, tmp_x.get(), hr_size_x);
                LoopX(src + size_x * iy, size_x, hr_dest + hr_size_x * oi, hr_size_x);
                for (int ix = 0; ix < hr_size_x; ++ix) {
                    hr_dest[hr_size_x * oi + ix] = hr_dest[hr_size_x * oi + ix]+ tmp_x[ix];
                    if constexpr (sizeof(T) == 16) {
                        *((double*)(&hr_dest[hr_size_x * oi + ix]) + 1) = 0.0;
                    } else if constexpr (sizeof(T) == 8) {
                        *((float*)(&hr_dest[hr_size_x * oi + ix]) + 1) = 0.0;
                    }
                }
            }
        } else {
            const int64_t iy = iy_half;
            const int64_t oi = iy;
            LoopX(src + size_x * iy, size_x, hr_dest + hr_size_x * oi, hr_size_x);
        }

        for (int iy = size_y - iy_half + 1; iy < size_y; ++iy) {
            const int64_t oi = hr_size_y - size_y + iy;
            LoopX(src + size_x * iy, size_x, hr_dest + hr_size_x * oi, hr_size_x);
        }

        };


    auto tmp_2 = std::make_unique<T[]>(hr_size_x* hr_size_y);
    auto LoopZ = [&](const T* src, int size_z, T* hr_dest, int hr_size_z) {
        for (int iz = 0; iz < iz_half; ++iz) {
            const int64_t oi = iz;
            LoopY(src + size_x* size_y * iz, size_y, hr_dest + hr_size_x* hr_size_y * oi, hr_size_y);
        }

        if (size_z < hr_size_z) {
            if ((size_z & 0x1) == 0) {
                const int iz = iz_half;
                const int oi = hr_size_z - size_z + iz;

                memset(tmp_2.get(), 0, sizeof(T)* hr_size_x* hr_size_y);

                LoopY(src + size_x * size_y * iz, size_y, tmp_2.get(), hr_size_y);

                for (int ix = 0; ix < hr_size_x * hr_size_y; ++ix) {
                    tmp_2[ix] = tmp_2[ix]* 0.5;
                }
                memcpy(hr_dest + hr_size_x * hr_size_y * iz, tmp_2.get(), sizeof(T) * hr_size_x* hr_size_y);
                memcpy(hr_dest + hr_size_x * hr_size_y * oi, tmp_2.get(), sizeof(T) * hr_size_x* hr_size_y);
            }
        } else if (size_z > hr_size_z) {
            if ((hr_size_z & 0x1) == 0) {
                //k=N/2の成分を2倍にする(FFTの仕様に準じて)
                const int64_t iz = iz_half;
                const int64_t oi = iz;
                const int64_t iz2 = size_z - iz_half;

                memset(tmp_2.get(), 0, sizeof(T)* hr_size_x* hr_size_y);
                LoopY(src + size_x * size_y * iz2, size_y, tmp_2.get(), hr_size_y);
                LoopY(src + size_x * size_y * iz, size_y, hr_dest + hr_size_x * hr_size_y * oi, hr_size_y);
                for (int ix = 0; ix < hr_size_x* hr_size_y; ++ix) {
                    hr_dest[hr_size_x* hr_size_y * oi + ix] = hr_dest[hr_size_x * hr_size_y * oi + ix] + tmp_2[ix];
                    if constexpr (sizeof(T) == 16) {
                        *((double*)(&hr_dest[hr_size_x * oi + ix]) + 1) = 0.0;
                    } else if constexpr (sizeof(T) == 8) {
                        *((float*)(&hr_dest[hr_size_x * oi + ix]) + 1) = 0.0;
                    }
                }
            }
        } else {
            const int64_t iz = iz_half;
            const int64_t oi = iz;
            LoopY(src + size_x * size_y * iz, size_y, hr_dest + hr_size_x * hr_size_y * oi, hr_size_y);
        }

        for (int iz = size_z - iz_half + 1; iz < size_z; ++iz) {
            const int64_t oi = hr_size_z - size_z + iz;
            LoopY(src + size_x * size_y * iz, size_y, hr_dest + hr_size_x * hr_size_y * oi, hr_size_y);
        }

        };


    LoopZ(src, size_z, hr_dest, hr_size_z);
}

