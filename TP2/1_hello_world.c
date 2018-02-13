#include <stdio.h>
#include <omp.h>

int main(){
	int number_threads = 5000;
	omp_set_num_threads(number_threads);
	int count = 0;
	#pragma omp parallel for shared(count)
	for (int i=0; i<number_threads; i++){
		#pragma omp critical
		{count++;}
		int me = omp_get_thread_num();
		printf("hello world from thread %d\n",me);
	}

	if(count!=number_threads){
		printf("bad count : %d\n", count);
	}else{
		printf("good count : %d\n", count);
	}
	return 0;
}
