/*
 ============================================================================
 Name        : trios_build_operator.c
 Author      : Renato Stoffalette Joao
 Version     :
 Copyright   : Your copyright notice
 Description :
 ============================================================================
 */
#include <stdio.h>
#include <trios.h>
#include <stdlib.h>


void print_usage() {
    fprintf(stderr, "\n Usage: build_operator window_FILE MINTERM_FILE Operator_OUTPUT \n");
    fprintf(stderr, "\n This tools executes the training process to learn image operators from a set of samples.\n");
}


int main(int argc,char *argv[]){
	 image_operator_t *iop = NULL;
	 window_t *win = NULL;
	 int i=0;
	 FILE *fp = NULL;;
	 int c,n;
	 mtm_t *mtm = NULL;
	 mtm_BX *table_BX = NULL;
	 int type = 0;
	 apert_t *apt;
	 int tr_set_index = 4;
	 unsigned int nmtm, sum;
	 int lines = 0;
	 int *wpattern;
	 itv_t *temp;

	 if (argc < 4) {
		 print_usage();
	     return 0;
	 }

	 win = win_read(argv[1]);
	 if (!win) {
		 trios_fatal("Invalid window %s.\n", argv[1]);
	 }

	 int wsize =  *win->wsize;
	 char *wpat[wsize];


	 trios_malloc(iop, sizeof(image_operator_t), image_operator_t *, "Failed to alloc image operator");
	 iop->type = BB;
	 iop->win = win;
	 iop->apt = NULL;
	 iop->gg = NULL;

	 char *filename = argv[2];

	 if(!(mtm  = mtm_read(filename, &win,NULL))){
        	 printf("Erro na leitura do arquivo %s. \n",filename);
        }


	 iop->decision = mtm;
	 if (iop->decision == NULL) {
	        return (image_operator_t *) trios_error(MSG, "Error in decision");
	    }

	 iop->bb = lisi_partitioned(iop->win, iop->decision, 13000);
	 if (iop->bb == NULL) {
	     return (image_operator_t *) trios_error(MSG, "Error in lisi_partitioned");
	 }

	 if (iop == NULL) {
	     trios_fatal("Error during the training process.\n\n");
	 }

	 if (image_operator_write(argv[3], iop) == 0) {
	     trios_fatal("Error writing operator %s \n", argv[3]);
	 }

	 return EXIT_SUCCESS;
}
