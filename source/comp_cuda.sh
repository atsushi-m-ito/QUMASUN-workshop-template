cp -p main_mpi.cpp main.cu

#nvcc -ccbin mpic++ -DGY_LOOP -DGY_WITH_CUDA -DUSE_MPI -restrict -std=c++17 -O3 --extended-lambda -arch=sm_90 -lcudart -lcufft -lcublas -lcusolver -lstdc++ -lstdc++fs -Xcompiler -std=c++17 main.cu -o qumasun_cuda.exe
nvcc -ccbin mpic++ -DGY_LOOP -DGY_WITH_CUDA -DUSE_MPI -restrict -std=c++17 -O3 --extended-lambda -arch compute_80 -code=sm_80 -lcudart -lcufft -lcublas -lcusolver -lstdc++ -lstdc++fs -Xcompiler -std=c++17 main.cu -o qumasun_cuda.exe

#nvcc -ccbin mpic++ -DGY_LOOP -DGY_WITH_CUDA -DUSE_MPI -restrict -std=c++17 -O3 --extended-lambda -arch compute_80 -code=sm_80 -lcudart -lcufft -lcublas -lcusolver -lstdc++ main.cu -o qumasun_cuda.exe


cp -p qumasun_cuda.exe ../
