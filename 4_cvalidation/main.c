#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "util/mnist-utils.h"

#define INPUT 784
#define L1 1024
#define L2 1024
#define L3 784
#define L4 512
#define OUTPUT 10

// intermediate values...
int input[INPUT];
int intervec1[L1];
int intervec2[L2];
int intervec3[L3];
int intervec4[L4];
int output[OUTPUT];

// weights...
int W1[INPUT*L1];
int W2[L1*L2];
int W3[L2*L3];
int W4[L3*L4];
int W5[L4*OUTPUT];

void load_weights(FILE *fp, int row, int col, int *W)
{
  int i, j;
  char buf[100000];
  char *value;
  for (i=0;i<row;i++)
  {
    fgets(buf, sizeof(buf), fp);
    value = buf;
    W[i*col ] = atoi(value); 
    value = strtok(buf, ",");

    for(j=1;j<col;j++)
    {
	W[i*col + j] = atoi(value);
	value = strtok(NULL, ",");
// 	printf("(%d,%d): %d\n",i,j, W[i*col + j]);
    }
  }
  return;
}

void pass(int *input, int *output, int *weight, int wrow, int wcol)
{
  int i, j;
  int res = 0;
  memset(output, 0, 4 * wcol);
  //printf("layer..%d %d \n", wrow, wcol);
  // MAC operation...
  for( i=0;i<wcol;i++)
  {
    for(j=0;j<wrow;j++)
    {
 	output[i] += input[j] * weight[j*wcol + i];
    }
  }
  

  // activation...
  for(i=0;i<wcol;i++)
  {
   // printf("%d activation %d ->", i, output[i]); 
    output[i] = (int)(tanh((float)output[i]) * 100.0);
   // printf(" %d\n", output[i]);
  }

} 

int main(void)
{

   // open MNIST files
    FILE *imageFile, *labelFile;
    imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);
	
    // loading weights
    FILE *w1, *w2, *w3, *w4, *w5;
    w1 = fopen("../model/W1.csv" , "r");
    w2 = fopen("../model/W2.csv", "r");
    w3 = fopen("../model/W3.csv", "r");
    w4 = fopen("../model/W4.csv", "r");
    w5 = fopen("../model/W5.csv", "r");

    // load weights..
    load_weights(w1, INPUT, L1, W1);
    load_weights(w2, L1, L2, W2);
    load_weights(w3, L2, L3, W3);
    load_weights(w4, L3, L4, W4);
    load_weights(w5, L4, OUTPUT, W5);


    // testing ...  
    int errCount = 0;
    
    // Loop through all images in the file
    for (int imgCount=0; imgCount<MNIST_MAX_TESTING_IMAGES; imgCount++){
         
	// get expected output and input..
        // Reading next image and corresponding label
        MNIST_Image img = getImage(imageFile);
	MNIST_Label lbl = getLabel(labelFile);
	
	// get image..
		
	// looping through all layers..
	pass(img.pixel, intervec1, W1, INPUT ,L1);
	pass(intervec1, intervec2, W2, L1 ,L2);
	pass(intervec2, intervec3, W3, L2 ,L3);
	pass(intervec3, intervec4, W4, L3 ,L4);
	pass(intervec4, output, W5, L4 , OUTPUT);

	//printf("inference finished ..%d\n", imgCount);	
	int max= output[0];
	int i;
	int max_idx = 0;
	for(i=0;i<10;i++)
	{
	  if(max < output[i])
	  {
	    max = output[i];
	    max_idx = i;
	  }
	}
	printf("expected %d read %d", lbl, max_idx);
	if( lbl == max_idx)
	  printf(" corect!\n");
	else
	  printf("\n");
    }
    
    fclose(imageFile);
    fclose(labelFile);

    fclose(w1);
    fclose(w2);
    fclose(w3);
    fclose(w4);
    fclose(w5);

    return 0;
}
