#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define MAX_INT 5//Borne supérieure de la valeur des cases
#define N 2//Taille de la matrice

int filltab(int* tab){
	#pragma omp parallel
	{
		//Toutes les cases du tableaux sont indépendantes, pas besoin de section critique
		tab[omp_get_thread_num()]=rand()%MAX_INT;
	}
	return 0;
}
//Non parallélisable : ordre important
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
	#pragma omp parallel for shared(ab)
	for (int x=0; x<N; x++){
		for(int y=0; y<N; y++){
			#pragma omp critical
				{
				ab[y*N+x]=0;
				for(int i=0; i<N; i++){
					ab[y*N+x] += a[y*N+i] * b[i*N+x];
				}
			}
		}
	}
	return 0;
}

int main(){
	srand(time(NULL));
	int a[N*N], b[N*N], ab[N*N];

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
