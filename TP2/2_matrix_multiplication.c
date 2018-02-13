#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define MAX_INT 2
#define N 2

int filltab(int* tab){
	#pragma omp parallel
	{
		tab[omp_get_thread_num()]=rand()%MAX_INT;
	}
	return 0;
}

int printab(int* tab){
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%d\t", tab[i*N+j]);
		}
		printf("\n");
	}
	return 0;
}

int multab(int* a, int* b, int* ab){
	#pragma omp parallel
	{
		int num = omp_get_thread_num();
		int y = num / N;
		int x = num % N;
		ab[y*N+x]=0;
		for(int i=0; i<N; i++){
			//i*N+j = num, mais pour la lisibilité de ce qui se passe
			ab[y*N+x] += a[y*N+i] * b[i*N+x];
		}
	}
	return 0;
}	

int main(){
	srand(time(NULL));
	int a[N*N], b[N*N], ab[N*N];
	omp_set_num_threads(N*N);

	filltab(a);
	filltab(b);

	printf("\nA\n\n");
	printab(a);

	printf("\nB\n\n");
	printab(b);

	multab(a, b, ab);

	printf("\nAxB\n\n");
	printab(ab);
	printf("\n");


	return 0;
}
