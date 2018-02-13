#include <stdio.h>
#include <omp.h>

int main(){
	omp_set_num_threads(5000);
	int count = 0;
	#pragma omp parallel
	{
		count++;
		int me = omp_get_thread_num();
		int nb = omp_get_num_threads();
		printf("hello world from thread %d\n",me);
		if(me==0){
			printf("nb of threads : %d\n", nb);
		}
	}
	printf("bad count :%d\n", count);
	return 0;
}
