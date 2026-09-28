#pragma once
//#include <ctchar>
#include <cstdio>
#include <cstring>
//#include <vec3.h>
//#include <mat33.h>
#include <charconv>
//#include "ATOMS_DATA.h"
#include "transpose.h"

class CubeReader3 {
public:
	CubeReader3() = default;
    ~CubeReader3() {
        if (read_buffer) {
            delete[] read_buffer;
        }
    }

	struct Frame {
		int grid_x;//グリッド数
		int grid_y;
		int grid_z;
		double boxaxis[9];
		double boxorg[3];
        char comment[1024];
	};

private:
    
    double* read_buffer = nullptr;
    Frame m_frame{ 0 };

public:


    
    Frame LoadCube(const char* filename) {
        Frame frame{ 0 };
        FILE* fp = fopen(filename, "r");

        const size_t SIZE = 1024;
        char line[SIZE];

        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//タイトル
        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//コメント
        strcpy(frame.comment, line);

        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//原子数とvolumeデータの原点座標
        int pcnt;
        sscanf(line, "%d %lf %lf %lf", &pcnt, &(frame.boxorg[0]), &(frame.boxorg[1]), &(frame.boxorg[2]));

        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//x方向のメッシュ数とx軸
        sscanf(line, "%d %lf %lf %lf", &(frame.grid_x), &(frame.boxaxis[0]), &(frame.boxaxis[1]), &(frame.boxaxis[2]));

        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//y方向のメッシュ数とx軸
        sscanf(line, "%d %lf %lf %lf", &(frame.grid_y), &(frame.boxaxis[3]), &(frame.boxaxis[4]), &(frame.boxaxis[5]));

        if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//z方向のメッシュ数とx軸
        sscanf(line, "%d %lf %lf %lf", &(frame.grid_z), &(frame.boxaxis[6]), &(frame.boxaxis[7]), &(frame.boxaxis[8]));

        //frame->boxorg[0] -= (frame->boxaxis[0] + frame->boxaxis[3] + frame->boxaxis[6]) / 2.0;
        //frame->boxorg[1] -= (frame->boxaxis[1] + frame->boxaxis[4] + frame->boxaxis[7]) / 2.0;
        //frame->boxorg[2] -= (frame->boxaxis[2] + frame->boxaxis[5] + frame->boxaxis[8]) / 2.0;
        frame.boxaxis[0] *= (double)frame.grid_x;
        frame.boxaxis[1] *= (double)frame.grid_x;
        frame.boxaxis[2] *= (double)frame.grid_x;
        frame.boxaxis[3] *= (double)frame.grid_y;
        frame.boxaxis[4] *= (double)frame.grid_y;
        frame.boxaxis[5] *= (double)frame.grid_y;
        frame.boxaxis[6] *= (double)frame.grid_z;
        frame.boxaxis[7] *= (double)frame.grid_z;
        frame.boxaxis[8] *= (double)frame.grid_z;


        {
            for (int i = 0; i < pcnt; i++) {
                if (fgets(line, 1024, fp) == NULL) { fclose(fp); return frame; }	//原子位置読み込み				
            }
        }

        //ボリュームデータの読み込み
        //const int64_t width = frame->grid_x;
        size_t mesh_sz = (frame.grid_x) * (frame.grid_y) * (frame.grid_z);
        if (read_buffer) delete[] read_buffer;
        read_buffer = new double[mesh_sz];
        //char* tp;
        double maxvalue = 0.0;
        double minvalue = 1e10;
        int64_t ix = 0;
        int64_t iy = 0;
        int64_t iz = 0;
        int64_t ioffset = 0;
        const int64_t width = (frame.grid_z);
        const char* const endptr = line + SIZE - 1;
        while (fgets(line, SIZE, fp) != NULL) {
            char* tp = line;
            const int64_t i_end = std::min(iz + 6, width);
            for (; iz < i_end; ++iz) {
                char* ep = nullptr;
                read_buffer[iz + ioffset] = strtod(tp, &ep);
                tp = ep + 1;
            }
            if (iz == width) {
                iz = 0;
                ++iy;
                if (iy == frame.grid_y) {
                    iy = 0;
                    ++ix;
                }
                ioffset = width * (ix + frame.grid_x * iy);
            }
        }

        fclose(fp);
        
        m_frame = frame;
        return frame;
    }

    void GetMeshData(double* dest) {
        const int64_t width = m_frame.grid_z;
        transpose(width, (m_frame.grid_x) * (m_frame.grid_y), read_buffer, width, dest, (m_frame.grid_x) * (m_frame.grid_y));
    }


};
