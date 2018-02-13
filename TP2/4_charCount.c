#include <stdio.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

int main(){
	const char * array[] = {
    "First entry",
    "Second entry",
    "Third entry"
	};
	int arrayLength = 3;
	int recLetter[256]={0};
	int nbOfSpace = 0;
	int totalNb = 0;
	
	#pragma omp parallel for shared(recLetter,nbOfSpace, totalNb)
	for(int i = 0; i< arrayLength; i++){
		#pragma omp parallel for
		for(int c = 0; c< strlen(array[i]); c++){
			#pragma omp critical
			{
				recLetter[(int)array[i][c]]++;
				if(array[i][c] == ' ') nbOfSpace++;
				totalNb= totalNb + 1;
			}
		}
	}
		
	printf("Il y a %d caractéres 'e'\n", recLetter[101]);
	printf("%d\n", nbOfSpace);
	printf("%d\n", totalNb);
}
