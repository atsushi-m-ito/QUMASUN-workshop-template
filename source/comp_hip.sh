#hipify-perl main_cuda.cu > main_conv.hip

#ROCBLAS_DIR=/home/carbon/install/rocm-6.4.3
#ROCSOLVER_DIR=/home/carbon/install/rocm-6.4.3/rocSOLVER/build/release/rocsolver-install

export OMPI_CC=hipcc

cp -pf main_mpi.cpp main.hip
mpicc -x hip -DGY_LOOP -DGY_APU -DGY_WITH_HIP -DUSE_MPI -O3 -std=c++17 -Wno-unused-result \
	-lrocsolver -lrocblas -lhipfft \
	--offload-arch=gfx942 \
	main.hip -o qumasun_hip.exe

#	-L${ROCBLAS_DIR}/lib -L${ROCSOLVER_DIR}/lib -L${ROCMSYS}/lib -lrocsolver -lrocblas -lhipfft \
#        -I${ROCBLAS_DIR}/include -I${ROCSOLVER_DIR}/include -I${ROCMSYS}/include \
#        -Wl,-rpath,${ROCSOLVER_DIR}/lib \
#        -Wl,-rpath,${ROCBLAS_DIR}/lib \
#        -Wl,-rpath,${ROCMSYS}/lib \


ldd ./qumasun_hip.exe | grep -E 'rocsolver|rocblas|hipfft'

cp -p qumasun_hip.exe ../
