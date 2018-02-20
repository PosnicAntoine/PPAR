#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

int main(){

	int nb = 2000;
	bool prime[nb];
	int prime_counter=0;
	omp_set_num_threads(nb);
	#pragma omp parallel
	{
		prime[omp_get_thread_num()]=true;
	}
	int maxsteps = sqrt(nb)+1;
	for(int i=2; i<maxsteps; i++){
		if(prime[i]){
			#pragma omp parallel
			for(int j=0; i*(j+2)<nb; j++){
				prime[i*(j+2)]=false;//Tous les multiples de i à partir de 2x
			}
			prime_counter++;
			printf("%d, ",i);
		}
	}
	for(int i=maxsteps; i<nb; i++){
		if(prime[i]){
			prime_counter++;
			printf("%d, ",i);
		}
	}
	printf("\nnb primes :%d\n",prime_counter);
}
