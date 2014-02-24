/*
 * trios_build_mtm.c
 *
 *  Created on: Feb 19, 2014
 *      Author: rsjoao
 */

#include <stdio.h>
#include <stdlib.h>
#include <trios.h>
#include <string.h>
#include <strings.h>

void print_usage() {
    fprintf(stderr, "Usage: trios_build_mtm window training_set result_path\n\n");
    fprintf(stderr, "This tools writes a minterm file from a set of samples.\n");
}

int main(int argc, char *argv[]) {
    image_operator_t *op;
    window_t *win;
    int i;
    imgset_t *training;
    image_operator_t *iop;
    itv_t *temp;

    if (argc < 2) {
        print_usage();
        return 0;
    }

    win = win_read(argv[1]);
    if (!win) {
           trios_fatal("Invalid window %s.\n", argv[1]);
     }

    training = imgset_read(argv[2]);
    if (!training) {
    	trios_fatal("Invalid image set %s.\n", argv[2]);
     }

    /*!
      Writes a classified examples set to a file.

      \param fname File name.
      \param mtm Classified examples structure.
      \param win Window used.
      \return 1 on success. 0 on failure.

       int mtm_write(char *fname, mtm_t * mtm, window_t * win, apert_t *apt);
    */


    trios_malloc(iop, sizeof(image_operator_t), image_operator_t *, "Failed to alloc image operator");
    iop->type = BB;
    iop->win = win;
    iop->apt = NULL;

    iop->collec = lcollec_memory(training, win, BB);
    if (iop->collec == NULL) {
        return (image_operator_t *) trios_error(MSG, "Error in collec");
    }
    iop->decision = ldecision_memory(iop->collec, 1, 0, 10, 0, 0);
    if (iop->decision == NULL) {
        return (image_operator_t *) trios_error(MSG, "Error in decision");
    }

    mtm_write(argv[3],iop->decision, win, NULL);

    return 0;
}
