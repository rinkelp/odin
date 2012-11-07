all:  gpuwaxs

gpuwaxs: main.cu run1.cu
	nvcc -use_fast_math -Xptxas -v -arch=sm_21 run1.cu -I ./

clean:  
	rm -f ./a.out
