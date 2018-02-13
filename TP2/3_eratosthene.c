#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

int main(){

	int nb_nb = 1000;
	bool prime[nb_nb];
	int prime_counter=0;
	omp_set_num_threads(nb_nb);
	#pragma omp parallel
	{
		prime[omp_get_thread_num()]=true;
	}
	int maxsteps = sqrt(nb_nb)+1;
	for(int i=2; i<maxsteps; i++){
		if(prime[i]){
			omp_set_num_threads(nb_nb/i);
			#pragma omp parallel
			{
				prime[i*(omp_get_thread_num()+2)]=false;
			}
			prime_counter++;
			printf("%d, ",i);	
		}
	}
	for(int i=maxsteps; i<nb_nb; i++){
		if(prime[i]){
			prime_counter++;
			printf("%d, ",i);	
		}
	}
	printf("\nnb primes :%d\n",prime_counter);
}
