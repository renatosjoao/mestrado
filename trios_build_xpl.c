/*
 ============================================================================
 Name        : trios_build_xpl.c
 Author      : Renato Stoffalette Joao
 Version     :
 Copyright   : Your copyright notice
 Description :
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "trios.h"
#include <string.h>
#include <strings.h>

int main(int argc, char* argv[]) {
    char *training_set_descr = NULL;
    char *window_descr = NULL;
    char *output_file = NULL;


    if (argc < 4) {

            printf("\n Usage: build_xpl image_set_path window_path outputfile \n \n ");
             return -1;
      }else{
    	  training_set_descr = argv[1];
    	  window_descr = argv[2];
    	  output_file = argv[3];
    	  lcollec(training_set_descr,window_descr,NULL,1,1,1,output_file,NULL);
      }
	return EXIT_SUCCESS;
}

void printHelp(){
printf("********************\n");
printf("Usage:\n");

}
