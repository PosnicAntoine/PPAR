
/* Sequential algorithm for converting a text in a digit sequence
 *
 * PPAR, TP4
 *
 * A. Mucherino
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int getvalue(int ascii){
	switch(ascii){
		case 49:
			return 1;
		case 50:
			return 2;
		case 51:
			return 3;
		case 52:
			return 4;
		case 53:
			return 5;
		case 54:
			return 6;
		case 55:
			return 7;
		case 56:
			return 8;
		case 57:
			return 9;
		default:
			return 0;
	}
}

int main(int argc,char *argv[])
{
   int i,n;
   int count,ascii_code;
   char *text;
   char filename[20];
   short notblank,notpoint,notnewline;
   short number,minletter,majletter;
   FILE *input;

   // getting started (we suppose that the argv[1] contains the filename related to the text)
   input = fopen(argv[1],"r");
   if (!input)
   {
      fprintf(stderr,"%s: impossible to open file '%s', stopping\n",argv[0],argv[1]);
      return 1;
   };

   // checking file size
   fseek(input,0,SEEK_END);
   n = ftell(input);
   rewind(input);

   // reading the text
   text = (char*)calloc(n+1,sizeof(char));
   for (i = 0; i < n; i++)  text[i] = fgetc(input);

   // converting the text
   int values[n];
   int curr = 0;
   count = 0;
   
   for (i = 0; i < n; i++)
   {
      ascii_code = (int)text[i];
      notblank =   (ascii_code !=  32);
      notpoint =   (ascii_code !=  46);
      notnewline = (ascii_code !=  10);
      number =     (ascii_code >=  48 && ascii_code <=  57);  // 0-9
      majletter =  (ascii_code >=  65 && ascii_code <=  90);  // A-Z
      minletter =  (ascii_code >=  97 && ascii_code <= 122);  // a-z

	  if(number){
		 if(count!=0){
			values[curr] = count;
			count = 0;
		 }
		 values[curr+1] = getvalue(ascii_code);
		 curr+=2;
	  }
	  else if(majletter || minletter)
	  {		  
		 count+=1;
	  }
	  else if(notblank && notpoint && notnewline)
	  {
		 if(count!=0){
			values[curr] = count;
			count = 0;
		 }
		 values[curr+1] = 0;
		 curr+=2; 
	  }
	  else
	  {
		 if(count!=0){
			values[curr] = count;
			count = 0;
		 }
		 curr+=1;
	  }
   }

	for(i=0; i < n;i++)
	{
		printf("%d ",values[i]);
	}
   // closing
   free(text);  fclose(input);

   // ending
   return 0;
};

