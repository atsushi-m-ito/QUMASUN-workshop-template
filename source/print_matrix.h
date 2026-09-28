#pragma once
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#if 0
#ifdef _WIN32
#define finite(x) _finite(x)
#endif
#endif

namespace {
	/*
	逆行列の計算
	m: 行列
	size:	行列のサイズ(size x sizeの二次元行列となる)
	inv_m:	逆行列の格納場所
	*/
	inline void OutputMatrix(double* m, int size1, int size2, const char* filepath) {

		FILE* fp = fopen(filepath, "w");

		for (int j = 0; j < size1; j++) {
			for (int i = 0; i < size2 - 1; i++) {
				fprintf(fp, "%f\t", m[j + size1* i]);
			}
			fprintf(fp, "%f\n", m[j + size1 * (size2 - 1)]);
		}

		fclose(fp);
	}

    inline void PrintMatrix(double* m, int size1, int size2) {


        for (int j = 0; j < size1; j++) {
            for (int i = 0; i < size2 - 1; i++) {
                printf("%f\t", m[j + size1 * i]);
            }
            printf("%f\n", m[j + size1 * (size2 - 1)]);
        }

    }

}
