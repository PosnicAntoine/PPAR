
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc,char *argv[]){
	int k, total, n;

	if (argc < 2){
	  fprintf(stderr,"Please provide k : ");
	  fscanf(stdin,"%d",&k);
	}
	else{
	  k = atoi(argv[1]);
	};

	if (k < 0){
	  fprintf(stderr,"%s: invalid value for the tree size, size is %d!\n",argv[0],k);
	  return 1;
	};

	n = 1;
	for (int i = 0; i < k; i++){
	   n *= 2;
	};
	fprintf(stdout, "le n est : %d \n", n);

	int *temp = (int *) malloc(n * sizeof(int));
	for (int i = 0; i < n; i++){
	   temp[i]=i+1;
	};
	
	for(int j = 1; j < n; j *= 2){// O(log(n))
		for(int i= 0; i<=n; i += j * 2){
			temp[i] += temp[i+j];
		};
	};
	
	total=temp[0];
	fprintf(stdout, "le total est : %d \n", total);
	return total;
};
