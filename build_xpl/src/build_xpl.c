/*
 ============================================================================
 Name        : build_xpl.c
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
    /*
    FILE *fp,*wind=NULL;
    //wind=fopen("./w9x9.win", "r");
    char *train_descr = "./training.set";
    char *window_descr = "./w9x9.win";
    char *foutput = "./output.xpl";

    training = imgset_read(train_descr);

    win = win_read(window_descr);
   // printf("Height : %d\n",win->height);
    //printf("Width : %d\n",win->width);

    /**
        Collects examples from a set of images.
        \param fname_i1 IMGSET file.
        \param fname_i2 WINSPEC file.
        \param fname_i3 Optional input XPL file. It must be NULL if not used.
        \param input_type Flag to indicate if the input images are binary. YES=1, NO=0.
        \param output_type Flag to indicate if the output images are binary. YES=1, NO=0.
        \param cv_flag Flag to indicate if all pixels that form a w-pattern must have the same value as the pixel under the central point of the window. This is useful to collect w-patterns ignoring neighboring objects.
        \param fname_o1 Output XPL file.
        \param fname_o2 Optional output report file. It must be NULL if not given.
        \return 1 on success. 0 on failure.


    int result = lcollec(train_descr,window_descr,NULL,1,1,1,foutput,NULL);



    /*fp=fopen("./veja9x9.xpl", "r");
    if (fp == NULL) perror ("Error opening file");
       else {
    	char mystring [100];
    	fgets(mystring,100,fp);
    	printf("%s\n",mystring);
        while(fgets(mystring,100,fp)!=NULL){
    		 printf("%s",mystring);
    }
    fclose(fp);
       }



    */


    // if (argc < 4) {
    //     printf("Usage: my_build image_set_path window_path result_path test_set\n");
     //    return -1;
    // }
    //printf("%s\n",argv[0]);
   // int ret = strcmp(argv[1],help);
    //printf("%d\n",ret);
    //if(ret ==0 ){
    //	printHelp();
   // }

     //training = imgset_read(argv[1]);
     //win = win_read(argv[2]);

     /*if (argc == 5) {
         test = imgset_read(argv[4]);
     } else {
         test = NULL;
     }

     if (training == NULL || win == NULL) {
         fprintf(stderr, "Error reading window or training image set.\n");
         return -1;
     }

     op = image_operator_build_bb(training, win);
     if (op == NULL) {
         fprintf(stderr, "Error building operator\n");
         return -1;
     }

     if (test != NULL) {
         mae = image_operator_mae(op, test, &acc);
         printf("MAE: %u Accuraccy: %f \n", mae, acc);
     }

     image_operator_write(argv[3], op);
*/
	return EXIT_SUCCESS;
}

void printHelp(){
printf("********************\n");
printf("Usage:\n");

}
