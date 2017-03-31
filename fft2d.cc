// Distributed two-dimensional Discrete FFT transform
// Vibha Satya Narayan
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;
void Transform1D(Complex *,int, Complex *);

void Transpose(Complex * arr, int width, int height){
 for(int i=0;i <width; i++){
   for(int j=0; j< height; j++){
	if(i<=j) continue;
	  else
	   { Complex temp=arr[i+width*j];
             arr[i+width*j]=arr[j+width*i];
             arr[j+width*i]=temp;
}
}	
}
}
void InvTransform1D(Complex* h, int w, Complex* H)
{
  for(int i = 0; i < w; ++i)
  {
    Complex sum(0, 0);
    for(int k = 0; k < w; ++k)
    {
      double theta = 2*M_PI*i*k/w;
      double real = cos(theta);
      double imag = sin(theta);
      real = real / w;
      imag = imag / w;
      Complex w_nk(real, imag);
      sum = sum + w_nk * h[k];
    }
    H[i] = sum;
  }
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9
int height,width,nCPU,rank;    
height=image.GetWidth();
width=image.GetHeight();
	
  MPI_Comm_size(MPI_COMM_WORLD,&nCPU);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Status status;

	Complex *result=new Complex[width*height];
	Complex *data=image.GetImageData();
      
int rowspercpu=height/nCPU;
int startingrow=rowspercpu*rank;
  for(int i=0; i< rowspercpu; i++){
    Transform1D(((startingrow+i)*width)+data, width, result+(width*(startingrow+i)));
}

if(rank==0){
 	for(int i=1;i<nCPU;i++){
	  MPI_Recv(data+(width*i*rowspercpu),rowspercpu*width*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);				
}
  memcpy(data, result,  rowspercpu*width*sizeof(Complex));
}

if(rank!=0){
	MPI_Send(result+startingrow*width,rowspercpu*width*sizeof(Complex),MPI_CHAR, 0, 0 , MPI_COMM_WORLD);
}

if(rank==0){
image.SaveImageData("MyAfter1D.txt",data,width,height);
}

if(rank==0){
	Transpose(data,width,height);
}

rowspercpu=width/nCPU;
startingrow=rank*rowspercpu;
if(rank==0){
 for(int i=0;i<nCPU; i++){
   MPI_Send((data+(i*height*rowspercpu)),rowspercpu*height*sizeof(Complex),MPI_CHAR,i,0,MPI_COMM_WORLD); 
}
}

if(rank!=0){
 MPI_Recv(data+(startingrow*height),rowspercpu*height*sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
}

for(int i=0;i <rowspercpu; i++){
  Transform1D(data+(height*startingrow)+(i*height),height,result+(height*startingrow)+(i*height));
}

if(rank==0){
    for(int i=1;i<nCPU;i++){
          MPI_Recv(data+(height*i*rowspercpu),rowspercpu*height*sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
}
  memcpy(data, result,  rowspercpu*height*sizeof(Complex));

}


if(rank!=0){
        MPI_Send(result+startingrow*width,rowspercpu*height*sizeof(Complex),MPI_CHAR, 0, 0 , MPI_COMM_WORLD);

}

if(rank==0){

Transpose(data, height, width);

image.SaveImageData("MyAfter2D.txt", data,width, height);
}

delete []  result;


//inverse 
result = new Complex[width * height];
  if(0 == rank)
  {
    Transpose(data, width, height);
  }

  rowspercpu = width / nCPU;
  int rowsPerCPU=rowspercpu;
  startingrow = rowsPerCPU * rank;

  if(0 != rank)
    MPI_Recv(data + startingrow * height ,rowsPerCPU * height * sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD, &status);

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Send(data + rowsPerCPU * cpu * height, rowsPerCPU * height * sizeof(Complex), MPI_CHAR,cpu,0, MPI_COMM_WORLD);
    }
  }
  
  for(int i = 0; i < rowsPerCPU; ++i)
  {
    InvTransform1D(data + (startingrow * height) + (i * height), height, result + (startingrow * height) + (i * height));
  }

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * height, rowsPerCPU * height * sizeof(Complex),MPI_CHAR,cpu, 0,MPI_COMM_WORLD,&status);
    }

    memcpy(data, result, rowsPerCPU * height * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(result + startingrow * width,rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR, 0,0,MPI_COMM_WORLD);
   }

  if(0 == rank)
  {
    Transpose(data, height, width);
  }

  rowsPerCPU = height / nCPU;
  startingrow = rowsPerCPU * rank;

  if(0 != rank)
    MPI_Recv(data + startingrow * width,rowsPerCPU * width * sizeof(Complex),MPI_CHAR,0,0,MPI_COMM_WORLD,&status);

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Send(data + rowsPerCPU * cpu * width, rowsPerCPU * width * sizeof(Complex),MPI_CHAR,cpu,0,MPI_COMM_WORLD);
     }
  }
  
  for(int i = 0; i < rowsPerCPU; ++i)
  {
    InvTransform1D(data + (startingrow * width) + (i * width), width, result + (startingrow * width) + (i * width));
  }

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * width, rowsPerCPU * width * sizeof(Complex), MPI_CHAR,cpu, 0,MPI_COMM_WORLD, &status);
    }

    memcpy(data, result, rowsPerCPU * width * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(result + startingrow * width,
             rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR,0,0,MPI_COMM_WORLD);
 }


  if(0 == rank)
  {
     image.SaveImageData("MyAfterInverse.txt", data, width, height);
  }

  delete [] result;

}


void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
 // data, w is the width (N), and H is the output array.
for (int n=0; n< w; n++){
    Complex sum(0,0);
     for(int k=0; k< w; k++){
      double theta = 2*M_PI*n*k/w;
      double wreal = cos(theta);
      double wimag = -sin(theta);
      Complex wnk(wreal, wimag);
      sum = sum + wnk * h[k];
}
   H[n]=sum;     
}
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
int rc;  
 rc=MPI_Init(&argc,&argv);
  if(rc!=MPI_SUCCESS){
   printf("Error starting MPI program, terminating \n");
   MPI_Abort(MPI_COMM_WORLD, rc);
}
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
  MPI_Finalize();
}  

