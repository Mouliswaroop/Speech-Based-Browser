// 214101050_word_recognition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string.h>			
#include <math.h>			
#include <conio.h>			
#include <stdlib.h>
#include<Windows.h>

using namespace std;
#pragma comment(lib, "Winmm.lib")
#define training 0          //change it to 1 to continue with training again
#define basepath "my_recordings\\"	
#define frame_samples 320    // number of samples in a frame
#define stride 80			
#define p 12		// number of past samples in Linear Predictive Coding
#define norm_value 5000		// normalization factor considered
#define pathsize 4096		// buffer size for file path
#define msgsize 1024		// buffer size for console message
#define training_utterances 20 		// number of training utterances of each word
#define test_utterances 10	// numner of testing utterances of each word
#define total_words	8		// total number of words
#define samples_considered 7040	

#define T  150		// maximum number of frames to be considered or max. length of observation sequence
#define codebook_size 32  //size of codebook
#define N 5					//Number of hidden states
#define training_iters 20	// number of iterations the model is going through
#define avg_model 3				// averaging models for a word

long double wave[80000];
char words_array[total_words][20] = {"browser", "facebook", "notepad", "word", "paint", "music", "excel", "photos"};
long double tokura_weights[p];
long double codebook[codebook_size][p];
int word_match, overall_match;

/* HMM PARAMETERS */
long double A[N][N];							  //state transition probability distribution
long double B[N][codebook_size];				//observation suymbol probability distribution
long double pi[N];								//initial state distribution  
long double A_avg[N][N];						// averaged state transition probability matrix
long double B_avg[N][codebook_size];			// averaged observation symbol probability matrix
long double pi_avg[N];							// averaged initial state probability matrix
long double alpha[T][N];					// forward variable
long double beta[T][N];				  // backward variable
long double gamma[T][N];			// probability being in state i at time t
long double delta[T][N]; 	 // maximum probability along a single path at time t
long double xi[T-1][N][N]; // argumnet that maximized delta
int  psi[T][N];			        	  // probability of being at state i in time t and state j at t+1	
int qtstar[training_iters][T]; 			 		
long double pstar[training_iters];					

long double s[frame_samples];	// 320 samples evaluate as of now
long double r[p+1];	
long double a[p+1];
long double c[p+1];	
int obs_seq[T];
int framecount;
int recog_word;

/*
//to make the universe file from cepstrals 
void make_universe()
{
	FILE* fin;
	int i;
	fin = fopen("my_universe.csv","a");
	for(i = 1; i < p; i++)
	{
		fprintf(fin, "%Lf,", c[i]);
	}
	fprintf(fin, "%Lf\n", c[i]);
	fclose(fin);
}
*/

//to read the initial data
void read_data(int avg_num, int word)
{
	long double curr_value;
	int i, j;
	FILE *fin_A, *fin_B, *fin_Pi;

	//If the model is generated for the first time, read the initial models provided
	if(avg_num == 0)   
	{
		fin_A = fopen("models\\A.txt", "r");
		fin_B = fopen("models\\B.txt", "r");
		fin_Pi = fopen("models\\PI.txt", "r");	
	}
	//read the generated model if its not the first run
	else
	{
		char num[20], filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];

		memset(num,0,20);
		sprintf(num,"%s",words_array[word]);

		_snprintf(filenameA,sizeof(filenameA),"%s%s%s", "models\\A_final_model_", num,".txt");
		_snprintf(filenameB,sizeof(filenameB),"%s%s%s", "models\\B_final_word_", num,".txt");
		_snprintf(filenamePI,sizeof(filenamePI),"%s%s%s", "models\\PI_final_word_", num,".txt");
		fin_A = fopen(filenameA, "r");
		fin_B = fopen(filenameB, "r");
		fin_Pi = fopen(filenamePI, "r");	

	}

	//read matrix A
	if(fin_A != NULL)
	{
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				fscanf(fin_A, "%Lf", &curr_value);
				A[i][j] = curr_value;
			}
		}
	}

	//read matrix B
	if(fin_B != NULL)
	{
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < codebook_size; j++)
			{
				fscanf(fin_B, "%Lf", &curr_value);
				B[i][j] = curr_value;
			}
		}
	}

	//read pi matrix
	if(fin_Pi != NULL)
	{
		for(i = 0; i < N; i++)
		{
			fscanf(fin_Pi, "%Lf", &curr_value);
			pi[i] = curr_value;
			
		}
	}
	
	fclose(fin_A);			
	fclose(fin_B);			
	fclose(fin_Pi);			

}

//to print the model
void show_model()
{
	int i, j;

	printf("\nA: \n");
	for(i = 0; i < N; i++)
	{
		printf("\n");
		for(j = 0; j < N; j++)
		{
			printf("%e ",A_avg[i][j]);
		}
	}

	printf("\nB: \n");
	for(i = 0; i < N; i++)
	{
		printf("\n");
		for(j = 0; j < codebook_size; j++)
		{
			printf("%e ",B_avg[i][j]);
		}
	}
	printf("\nPI: \n");
	for(i = 0; i < N; i++)
	{
		printf("%e ",pi_avg[i]);
	}

}

//retrieve the waveform into the array 
void put_wave_in_array()
{
	memset(wave, 0, sizeof(wave));
	FILE *fin;
	long double curr_value;
	int sample_count = 0;
	fin = fopen("HMM\\wave_normalized.txt", "r");

	if(fin == NULL)
		printf("wave_normalized: can't read file");
	else
	{
		while(fscanf(fin, "%Lf\n", &curr_value) != EOF)
		{
			wave[sample_count] = curr_value;
			sample_count++;
		}


		fclose(fin);
	}

	fclose(fin);
}

//to find the power and return the value
long double power(int i, int j)		
{
	long double res = 1;
	int k;
	if(j < 0)
	{
		for(k = j; k < 0; k++)
			res /= i;
	}

	else
	{
		for(k = j; k > 0; k--)
			res *= i;
	}

	return res;
}

//to read the averaged model 
void read_averaged_model(int word)
{
	long double curr_value;
	int i, j;
	FILE *fin_A, *fin_B, *fin_Pi;

	char num[20];
	char filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];

	memset(num,0,20);
	sprintf(num,"%s",words_array[word]);

	_snprintf(filenameA,sizeof(filenameA),"%s%s%s", "models\\A_final_model_", num,".txt");
	_snprintf(filenameB,sizeof(filenameB),"%s%s%s", "models\\B_final_word_", num,".txt");
	_snprintf(filenamePI,sizeof(filenamePI),"%s%s%s", "models\\PI_final_word_", num,".txt");
	
	fin_A = fopen(filenameA, "r");
	fin_B = fopen(filenameB, "r");
	fin_Pi = fopen(filenamePI, "r");	

	if(fin_A != NULL)
	{
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				fscanf(fin_A, "%Lf", &curr_value);
				A[i][j] = curr_value;
			}
		}
	}

	if(fin_B != NULL)
	{
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < codebook_size; j++)
			{
				fscanf(fin_B, "%Lf", &curr_value);
				B[i][j] = curr_value;
			}
		}
	}

	if(fin_Pi != NULL)
	{
		for(i = 0; i < N; i++)
		{
			fscanf(fin_Pi, "%Lf", &curr_value);
			pi[i] = curr_value;
			
		}
	}
	fclose(fin_A);			
	fclose(fin_B);			
	fclose(fin_Pi);			
}

//to find the max valued index in a given row in B matrix
int find_B_row_max(int j)
{
	int k, max_index;
	long double max_b;

	max_index = 0;
	max_b = B[j][0];

	for(k = 1; k < codebook_size; k++)
	{
		if(B[j][k] > max_b)
		{
			max_b = B[j][k];
			max_index = k;
		}
	}

	return k;
}

//Soluiton to problem 1
long double solution1()
{
	int i, j, t;
	long double sum_forward;	    
	long double sum_backward;	
	long double prob_O_given_model = 0;

	// forward procedure 

	memset(alpha, 0, sizeof(alpha[0][0]) * N * T);

	//initialization
	for(i = 0; i < N; i++)
	{
		alpha[0][i] = pi[i] * B[i][obs_seq[0]];	
	}

	//induction
	for(t = 0; t < framecount-1; t++)
	{
		for(j = 0; j < N; j++)
		{
			sum_forward = 0;

			for(i = 0; i < N; i++)
				sum_forward += alpha[t][i] * A[i][j];

			alpha[t+1][j] = sum_forward * B[j][obs_seq[t+1]];
		}
	}

	// termination
	for(i = 0; i < N; i++)
	{
		prob_O_given_model += alpha[framecount - 1][i];
		
	}

	printf("\nP[O/model] = %0.40e", prob_O_given_model);

	//backward procedure
	memset(beta, 0, sizeof(beta[0][0]) * N * T);

	// initialization
	for(i = 0; i < N; i++)
	{
		beta[framecount-1][i] = 1;	
	}

	//induction step
	for(t = framecount-2; t >= 0; t--)
	{
		for(i = 0; i < N; i++)
		{
			sum_backward = 0;

			for(j = 0; j < N; j++)
			{
				sum_backward += A[i][j] * B[j][obs_seq[t+1]] * beta[t+1][j];
			}

			beta[t][i] = sum_backward;
		}
	}

	return prob_O_given_model;
}

//Solution to problem 2
void solution2(int model_num)
{
	int i, j, t;
	long double max_delta, delta_result;

	//initialization 
	memset(delta, 0, sizeof(delta[0][0]) * N * T);
	memset(psi, -1, sizeof(psi[0][0]) * N * T);	

	for(i = 0; i < N; i++)
	{
		delta[0][i] = pi[i] * B[i][obs_seq[0]];	
		psi[0][i] = -1;										
	}

	// recursion
	for(t = 1; t < framecount; t++)
	{
		for(j = 0; j < N; j++)
		{
			i = 0;
			max_delta = delta[t-1][i] * A[i][j];
			psi[t][j] = i;

			for(i = 1; i < N; i++)
			{
				delta_result = delta[t-1][i] * A[i][j];
				if(delta_result > max_delta)
				{
					max_delta = delta_result;
					psi[t][j] = i;
				}
			}

			delta[t][j] = max_delta * B[j][obs_seq[t]];
		}
	}

	//termination
	long double pstar_sub;		
	i = 0;
	pstar_sub = delta[framecount-1][i];
	qtstar[model_num][framecount-1] = i;		

	for(i = 1; i < N; i++)
	{
		if(delta[framecount-1][i] > pstar_sub)
		{
			pstar_sub = delta[framecount-1][i];
			qtstar[model_num][framecount-1] = i;
		}
	}


	// path backtracking
	for(t = framecount-2; t >= 0; t--)
		qtstar[model_num][t] = psi[t+1][qtstar[model_num][t+1]];

	printf("\nP* = %e", pstar_sub);
	pstar[model_num] = pstar_sub;
	
	printf("\nState sequence : ");
	for(t = 0; t < framecount; t++)
		printf("%d ", qtstar[model_num][t]+1);
	

}

//Solution to problem 3
void solution3()
{
	int i, j, k, t, count, max_index;
	long double denom_sum, product, numer_sum, threshold;

	memset(gamma, 0, sizeof(gamma[0][0]) * N * T);	

	// computing gamma values
	for(t = 0; t < framecount; t++)
	{
		denom_sum = 0;
		for(i = 0; i < N; i++)
		{
			product = alpha[t][i] * beta[t][i];
			gamma[t][i] = product;
			denom_sum += product;
		}

		for(i = 0; i < N; i++)
			gamma[t][i] /= denom_sum;			
		
	}

	// computing xi values
	for(t = 0; t < framecount-1; t++)
	{
		denom_sum = 0;
		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				product = alpha[t][i] * A[i][j] * B[j][obs_seq[t+1]] * beta[t+1][j];
				xi[t][i][j] = product;
				denom_sum += product;
			}
		}


		for(i = 0; i < N; i++)
		{
			for(j = 0; j < N; j++)
			{
				xi[t][i][j] /= denom_sum;			
			}
		}
	}


	// re-estimation of the model
	// new A
	for(i = 0; i < N; i++)
	{
		denom_sum = 0;
		for(t = 0; t < framecount-1; t++)
			denom_sum += gamma[t][i];		


		for(j = 0; j < N; j++)
		{
			numer_sum = 0; 
			for(t = 0; t < framecount-1; t++)
				numer_sum += xi[t][i][j];

			A[i][j] = numer_sum / denom_sum;				
		}
	}


	// new B
	for(j = 0; j < N; j++)
	{
		denom_sum = 0;
		for(t = 0; t < framecount; t++)
			denom_sum += gamma[t][j];		

		count = 0;
		for(k = 0; k < codebook_size; k++)
		{
			numer_sum = 0;
			for(t = 0; t < framecount; t++)
			{
				if(obs_seq[t]==k)
					numer_sum += gamma[t][j];
			}

			B[j][k] = numer_sum / denom_sum;
		}

		max_index = find_B_row_max(j);

		//adding threshold to zero values
		threshold = power(10, -30);
		for(k = 0; k < codebook_size; k++)
		{
			if(B[j][k] <= threshold)
			{
				count++;
				B[j][k] += threshold;
			}
		}

		B[j][max_index] -= count * threshold;  
	}
}

//to write the generated model into the files
void write_generated_model(int word)
{
	char num[20];
	char filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];
	int i, j;
	FILE *foutA, *foutB, *foutPI;

	memset(num,0,20);
	sprintf(num,"%s",words_array[word]);

	_snprintf(filenameA,sizeof(filenameA),"%s%s%s", "models\\A_final_model_", num,".txt");
	_snprintf(filenameB,sizeof(filenameB),"%s%s%s", "models\\B_final_word_", num,".txt");
	_snprintf(filenamePI,sizeof(filenamePI),"%s%s%s", "models\\PI_final_word_", num,".txt");
	foutA = fopen(filenameA, "w");
	foutB = fopen(filenameB, "w");
	foutPI = fopen(filenamePI, "w");

	for(i = 0; i < N; i++)
	{
		fprintf(foutPI, "%0.40Lf\t", pi_avg[i]);
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			fprintf(foutA, "%0.40Lf\t", A_avg[i][j]);

		fprintf(foutA, "\n");
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < codebook_size; j++)
			fprintf(foutB, "%0.40Lf\t", B_avg[i][j]);

		fprintf(foutB, "\n");
	}

	fclose(foutA);
	fclose(foutB);
	fclose(foutPI);

	printf("\n\n Model generated for word : %d\n\n", word);
}

//to initialize the HMM parameters
void initialize_globals()	
{
	memset(A, 0, sizeof(A[0][0]) * N * N);
	memset(B, 0, sizeof(B[0][0]) * codebook_size * N);
	memset(pi, 0, sizeof(pi));
	memset(A_avg, 0, sizeof(A_avg[0][0]) * N * N);
	memset(B_avg, 0, sizeof(B_avg[0][0]) * codebook_size * N);
	memset(pi_avg, 0, sizeof(pi_avg));
}


void initialize_HMM_parameters()
{
	memset(A, 0, sizeof(A[0][0]) * N * N);
	memset(B, 0, sizeof(B[0][0]) * codebook_size * N);
	memset(pi, 0, sizeof(pi));
	memset(alpha, 0, sizeof(alpha[0][0]) * N * T);
	memset(beta, 0, sizeof(beta[0][0]) * N * T);
	memset(delta, 0, sizeof(delta[0][0]) * N * T);
	memset(psi, -1, sizeof(psi[0][0]) * N * T);
	memset(gamma, 0, sizeof(gamma[0][0]) * N * T);		
	memset(qtstar, -1, sizeof(qtstar[0][0]) * T * training_iters);	
	memset(pstar, 0, training_iters);
}


//keep track for averaging out the model
void add_to_average()
{
	int i, j;

	for(i = 0; i < N; i++)
		pi_avg[i] += pi[i];


	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			A_avg[i][j] += A[i][j];
	}


	for(i = 0; i < N; i++)
	{
		for(j = 0; j < codebook_size; j++)
			B_avg[i][j] += B[i][j];
	}

}

//to get the average model for number of training utterances
void get_average_model()
{
	int i, j;

	for(i = 0; i < N; i++)
		pi_avg[i] /= training_utterances;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			A_avg[i][j] /= training_utterances;
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < codebook_size; j++)
			B_avg[i][j] /= training_utterances;
	}
}


//to remove th DC shift if any from the  wave
void dc_remover(char srcfile[], char destfile[])
{

	long double count, dc,curr_value, new_value, sum;
	char srcpath[pathsize], destpath[pathsize];
	FILE *fin,*fout;
	
	fin = fopen(srcfile,"r");
	fout = fopen(destfile,"w");

	if(fin == NULL)
	{
		printf(" Can't read file!\n");
	}
	else
	{
		sum = 0;
		count = 0;
		while(fscanf(fin,"%Lf\n",&curr_value)!=EOF)
		{
			count++;
			sum += curr_value;
		}
		//calculate dc shift
		dc = sum/count;
		fseek(fin,0L,SEEK_SET);
		//remove the dc shift value from all the samples 
		while(fscanf(fin,"%Lf\n",&curr_value) != EOF)
		{
			new_value = curr_value - dc;
			fprintf(fout, "%Lf\n",new_value);
		}
	}
	
	fclose(fin);
	fclose(fout);
}

//Normalize the waveform to range -5000 to 5000
int normalize_wave(char filepath[])
{
	long double curr_value, max_energy, wave_normalized;
	int max_sample, sample_number;
	FILE *fin, *normalizer;

	fin = fopen(filepath,"r");
	normalizer = fopen("HMM\\wave_normalized.txt","w+");  

	if(fin==NULL)
		printf("normalize: Cannot open file");
	else
	{
		fscanf(fin,"%Lf\n",&curr_value);                    
		max_energy = curr_value;
		sample_number = 1;
		max_sample = sample_number;

		//find the sample with maximum energy
		while(fscanf(fin,"%Lf\n",&curr_value) != EOF)
		{
			sample_number++;	
			if(abs(curr_value)>abs(max_energy))
			{
				max_energy = curr_value;   
				max_sample = sample_number;
			}                        
		}


		fseek(fin, 0, SEEK_SET);
		//normalize the wave form around -5000 to +5000
		while(fscanf(fin,"%Lf\n",&curr_value) != EOF)
		{
			wave_normalized = (curr_value/max_energy) * norm_value;
			fprintf(normalizer,"%Lf\n",wave_normalized);	
		}

		fclose(fin);
		fclose(normalizer);
		return max_sample;		
	}

	fclose(fin);
	fclose(normalizer);
}

//extract the samples within the range
void extract_word(int start, int end)
{
	FILE *fin, *fout;
	long double curr_value;
	int sample_count = 0;

	fin = fopen("HMM\\wave_normalized.txt","r");
	fout = fopen("HMM\\word.txt","w");  

	while(fscanf(fin, "%Lf\n", &curr_value) != EOF)
	{
		//retrieve the word in the specified range
		sample_count++;
		if((sample_count >= start) && (sample_count <= end))
		{
			fprintf(fout, "%Lf\n", curr_value);
		}

		if(sample_count == end)
			break;
	}

	fclose(fin);
	fclose(fout);
}

//extract the frame into the array s
void extract_frame(int i, long double s[])
{
	FILE *fin;
	long double curr_value;
	int count = 0, start = 0, k = 0;

	fin = fopen("HMM\\word.txt","r");

	if(fin == NULL)
	{
		printf("word.txt: Can't open file!");
	}
	else
	{
		//for the first frame, extract the sample values 
		if(i == 0) 
		{
			while(count < frame_samples)
			{
				fscanf(fin,"%Lf\n",&curr_value);
				s[k] = curr_value;
				k++;
				count++;
			}
		}
		//from the next frame, stride of 80 has to be considered
		else 		
		{
			while(start < (stride * i))
			{
				fscanf(fin,"%Lf\n",&curr_value);
				start++;	
			}	
			while(count < frame_samples)
			{
				fscanf(fin,"%Lf\n",&curr_value);
				s[k] = curr_value;
				k++;
				count++;
			}
		}
	}
	fclose(fin);
}

//returns number of samples
int count_samples()
{
	FILE *fin;
	long double curr_value;
	int count = 0;

	fin = fopen("HMM\\word.txt", "r");  
	
	while(fscanf(fin, "%Lf\n", &curr_value)!= EOF)
	{
		count++;
	}

	fclose(fin);
	return count;

}

//compute ri values
void compute_r_i(long double s[], long double r[])
{
	FILE  *fout;
	int m, k;
	long double r_k;

	fout = fopen( "HMM\\r_i.txt","w");

	for(k = 0; k < p+1; k++)		
	{
		r_k = 0;
		for(m = 0; m <= (frame_samples-1-k); m++)
		{
			r_k += s[m]*s[m+k];
		}
		r[k] = r_k;
		fprintf(fout, "%Lf\n", r[k]);
	}

	fclose(fout);
}

//compute ai values using durbins algorithm
int compute_A_i(long double r[], long double a[]) 
{
	FILE  *fout;
	int i, j;
	long double e[p+1] = {0};
	long double k[p+1] = {0};
	long double b[p+1][p+1], sum;
	

	fout = fopen("HMM\\coefficients.txt","w");

	e[0] = r[0];

	if(e[0]==0)
	{
		fclose(fout);
		return 0;
	}
	for(i = 1; i <= p; i++)
	{
		sum = 0;
		for(j = 1; j <= (i-1); j++)
		{
			sum += b[i-1][j]*r[i-j];
		}

		k[i] = (r[i]-sum) / e[i-1];
		
		b[i][i] = k[i];

		for(j = 1; j <= i-1; j++)
		{
			b[i][j] = b[i-1][j] - (k[i]*b[i-1][i-j]);
		}

		e[i] = (1 - k[i] * k[i]) * e[i-1];
	}

	for(i = 1; i <= p; i++)
	{
		a[i] = b[p][i];
		fprintf(fout, "%Lf\n", a[i]);
	}

	fclose(fout);
	return 1;
}

//compute cepstral coefficients
void compute_cepstral_coefficients(long double a[], long double c[])
{
	FILE *fin, *fout;		
	long double curr_value, rsw[p+1] = {0};
	int k, q, i=1;
	long double sum, num ;

	fout = fopen("HMM\\cepstral_coefficients.txt","w");
	fin = fopen("HMM\\rsw_values.txt","r");
	
	while(fscanf(fin, "%Lf\n", &curr_value)!= EOF)
	{
		rsw[i] = curr_value;				
		i++;
	}

	q = p;
	for(i = 1; i <= q; i++)
	{
		sum = 0;
		for(k = 1; k <= (i-1); k++)
		{
			num = ((long double)k/(long double)i) * c[k] * a[i-k];
			sum += num;
		}

		c[i] = a[i] + sum;		
	}

	for(i = 1; i <= q; i++)
	{
		c[i] = c[i] * rsw[i];			
		fprintf(fout, "%Lf\n", c[i]);
	}

	//make_universe();
	
	fclose(fin);					
	fclose(fout);
}

//generate the observation sequence from the cepstrals and codebook generated
void generate_obs_seq(long double c[],int framenum)
{
	int i, j, codebook_index;
	long double diff, dist, min_distance, distances[codebook_size];

	//calculate the distortion between the cepstral coefficients and codebook
	for(j = 0; j < codebook_size; j++)		
	{
		dist = 0;
		for(i = 1; i <= p; i++) 
		{
			diff = c[i] - codebook[j][i-1];
			dist += tokura_weights[i-1] * diff * diff;
		}

		distances[j] = dist;
	}

	//consider the index of the codebook with minimum distortion
	codebook_index = 0;
	min_distance = distances[codebook_index];
	for(i = 1; i < codebook_size; i++)
	{
		if(distances[i]<min_distance)
			{
				codebook_index = i;
				min_distance = distances[i];
			}
	}

	obs_seq[framenum] = codebook_index + 1;

}


/* HMM */
void runHMM()
{
	
	int model_iter = 0;		
	long double prev_prob_O_given_model, curr_prob_O_given_model;


	memset(qtstar, -1, sizeof(qtstar[0][0]) * T * training_iters);	
	memset(pstar, 0, training_iters);

	prev_prob_O_given_model = solution1();	
	solution2( model_iter);	
	solution3( );

	for(model_iter = 1; model_iter < training_iters; model_iter++)
	{
		// checking for improvement in the model re-estimated
		curr_prob_O_given_model = solution1( );
		if (curr_prob_O_given_model > prev_prob_O_given_model)		
		{
			prev_prob_O_given_model = curr_prob_O_given_model;
		}
		else if(curr_prob_O_given_model == prev_prob_O_given_model)
		{
		}
		else
		{
			prev_prob_O_given_model = curr_prob_O_given_model;
		}
		//conitnue for better convergance of the model
		solution2( model_iter);
		solution3( );

	}

}

//preprocessing of the data
void preprocessing(char curr_word[], int utterance_num, char file_num[], int avg_num, int word)
{
	char display[msgsize], filepath[pathsize];
	int frame_num, i, temp, peak, start, end;
	long double max_energy;

	memset(obs_seq, -1, sizeof(obs_seq));

	//live testing data
	if(curr_word[0] == '-')			
	{
		//handle dc shift
		dc_remover("input_file.txt","HMM\\dc_removed.txt");
	}
	else 										
	{
		_snprintf(filepath, sizeof(filepath), "%s%s%s%s%s", basepath, curr_word, "_", file_num, ".txt");
		//handle dc shift
		dc_remover(filepath,"HMM\\dc_removed.txt");
	}
	
	//Normalizing text data
	peak = normalize_wave("HMM\\dc_removed.txt");  
	//put the waveform into the array
	put_wave_in_array();

	//generating start and end values in the waveform to be considered
	start = peak - samples_considered/2;
	end = peak + samples_considered/2 - 1;
	extract_word(start, end);

	// calculating number of frames in the extracted word
	temp = ((count_samples() - frame_samples) / stride) + 1;
	framecount = (temp < T) ? temp : T; 

	for(frame_num=0; frame_num < framecount; frame_num++)
	{
		memset(s, 0, sizeof(s));
		memset(r, 0, sizeof(r));
		memset(a, 0, sizeof(a));
		memset(c, 0, sizeof(c));
		
		//extract frame under consideration into the array
		extract_frame(frame_num, s);
		//Computing R_i values
		compute_r_i(s, r);
		//Computing A_i values
		 compute_A_i(r, a);
		//Computing the C_i values (cepstral coefficients)
		compute_cepstral_coefficients(a, c);
		//generate the observation sequence
		generate_obs_seq(c, frame_num);
		
	}
	for(i = 0; i < framecount; i++)
	{
		obs_seq[i] -= 1;
	}
}

//To recognize the word for testing
void recognize_word(char curr_word[], int utterance_num, char file_num[], int avg_num, int word)
{
	int i;
	long double prob, max_prob;
	int index = 0;		

	//preprocessing of the word utterance
	preprocessing(curr_word, utterance_num, file_num, avg_num, word);
	//observation sequence will be obtained after preprocessing 
	initialize_HMM_parameters();
	//find the model with which it has the the maximum probability given observation sequence
	read_averaged_model(0);
	printf("\n word-%s :", words_array[0]);
	prob = solution1();
	max_prob = prob;
	for(i = 1; i < total_words; i++)
	{
		initialize_HMM_parameters();
		read_averaged_model(i);
		printf("\nword-%s :", words_array[i]);
		prob = solution1();
		//keep track of the word having the maximum probability
		if(prob > max_prob)
		{
			max_prob = prob;
			index = i;
		}
	}

	if(curr_word[0] != '-')
	{
		printf("\nActual word: %s\t Recognized word: %s\n", words_array[word], words_array[index]);	

		if(word == index)
		{
			//if the word is recognized correctly
			word_match++;
			overall_match++;
		}
	}

	//for live recorded data
	else
		printf("\n\nRecognized word : %s",words_array[index]);

	recog_word = index;

}

//training of the words is performed
void training_words()
{
	char filepath[pathsize], num[3];
	int i, j, word_num, avg_num, utterance_num;

	for(word_num = 0; word_num < total_words; word_num++)
	{
		for(avg_num = 0; avg_num < avg_model; avg_num++)	
		{
			initialize_globals();	
			printf("\nIteration : %d", avg_num+1);
			printf("\n-------------------------------------------------------------\n");
			for(utterance_num=1; utterance_num <= training_utterances; utterance_num++)		
			{

				memset(num,0,3);
				sprintf(num,"%d",utterance_num);
				printf("\nword %s : Utterance %d", words_array[word_num], utterance_num);
				//preprocessing of the recordings has to be done
				preprocessing(words_array[word_num], utterance_num, num,avg_num, word_num);
				initialize_HMM_parameters();
				//read the model
				read_data( avg_num, word_num);
				//HMM for new model generation
				runHMM();
				add_to_average();

			}
			//write the averaged model
			get_average_model();
			write_generated_model(word_num);
			show_model();
			printf("\n\n=============================================================================================================\n\n");
		}	
	}

	printf("\n\nFinished training !!!!");
}

//testing of the words is performed
void testing()
{
	char filepath[pathsize], num[3], rec_time[10];
	int i, j, time, option, word_num, avg_num, utterance_num;
	long double curr_value, word_accuracy, overall_accuracy;

	do
	{
		printf("\nMenu: \n1. Recognize on own test files \n2. Recognize on live recording \n3. Exit\n\n");
		scanf("%d",&option);

		switch(option)
		{
			case 1: 
					//testing on pre-recorded input

					overall_match = 0;
					for(word_num=0; word_num < total_words; word_num++)
					{
						word_match = 0;

						printf("\n\nStarting recognition system for the word %d\n\n", word_num);

						for(utterance_num=training_utterances+1; utterance_num <= test_utterances + training_utterances; utterance_num++)
						{
			
							memset(num,0,3);
							sprintf(num,"%d",utterance_num);
							printf("\nword %s : Utterance %d", words_array[word_num], utterance_num);
							recognize_word(words_array[word_num], utterance_num, num, -1, word_num);

						}

						word_accuracy = ((long double)word_match / (long double)test_utterances) * 100;
						printf("\nAccuracy in recognizing word %s : %Lf", words_array[word_num], word_accuracy);
						
					}
		
					overall_accuracy = ((long double)overall_match / (long double)(test_utterances * total_words)) * 100;
					printf("\n\n Overall accuracy of the system : %Lf\n", overall_accuracy);
					break;

			case 2: 
					//testing on live recording

					printf("\nEnter recording time in seconds : ");
					scanf("%d",&time);

					memset(filepath,0,pathsize);
					memset(rec_time,0,sizeof(rec_time));
					sprintf(rec_time,"%d",time);
					_snprintf(filepath, sizeof(filepath), "%s %s %s %s","Recording_Module.exe", rec_time, "input_file.wav","input_file.txt");
					//Usuage - ("Recording_Module.exe 3 input_file.wav input_file.txt");
					system(filepath);
					
					recognize_word("-1", -1, "-1",-1, -1); 

					break;

			case 3: printf("\nExiting ...");
					break;

			default: printf("\nInvalid input!!!...");

		}
		
	} while(option !=3);

}
//live training
void live_training()
{
	char filepath[pathsize], num[3] ;
	FILE *fin;
	char filenameA[msgsize], filenameB[msgsize], filenamePI[msgsize];
	int i, j;
	FILE *foutA, *foutB, *foutPI;

	for(i = 1; i <= 10; i++)
	{
		memset(filepath,0,pathsize);
		memset(num,0,3);
		sprintf(num,"%d",i);
		_snprintf(filepath, sizeof(filepath), "%s %s %s %s%s%s","Recording_Module.exe","3","input_file.wav","live_recordings/live_",num,".txt");
		system(filepath);
	}

	for(i = 1; i <= 10; i++)
	{
		memset(filepath,0,pathsize);
		memset(num,0,3);
		sprintf(num,"%d",i);
		_snprintf(filepath, sizeof(filepath), "%s%s%s","live_recordings\\live_",num,".txt");
		fin = fopen(filepath, "r");
		preprocessing("-1", -1, "-1",-1, -1); 
		initialize_HMM_parameters();
		//read the model
		read_data( 0, -1);
		//HMM for new model generation
		runHMM();
		add_to_average();
	}
	

	_snprintf(filenameA,sizeof(filenameA),"%s", "live_recordings\\A_live_final_model.txt");
	_snprintf(filenameB,sizeof(filenameB),"%s", "live_recordings\\B_live_final_model.txt");
	_snprintf(filenamePI,sizeof(filenamePI),"%s", "live_recordings\\PI_live_final_model.txt");
	foutA = fopen(filenameA, "w");
	foutB = fopen(filenameB, "w");
	foutPI = fopen(filenamePI, "w");

	for(i = 0; i < N; i++)
	{
		fprintf(foutPI, "%0.40Lf\t", pi_avg[i]/10);
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < N; j++)
			fprintf(foutA, "%0.40Lf\t", A_avg[i][j]/10);

		fprintf(foutA, "\n");
	}

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < codebook_size; j++)
			fprintf(foutB, "%0.40Lf\t", B_avg[i][j]/10);

		fprintf(foutB, "\n");
	}

	fclose(foutA);
	fclose(foutB);
	fclose(foutPI);

}
//live testing 
void live_testing()
{
	char filepath[pathsize], num[3], rec_time[10];
	int i, j, time, option, word_num, avg_num, utterance_num;
	long double curr_value, word_accuracy, overall_accuracy;

	memset(filepath,0,pathsize);
	_snprintf(filepath, sizeof(filepath), "%s","Recording_Module.exe 3 input_file.wav input_file.txt");
	system(filepath);

	recognize_word("-1", -1, "-1",-1, -1);
	
	//opening the application
	if( recog_word == 0)
	{
		printf("\n OPENING BROWSER\n ");
		PlaySound(TEXT("instructions\\browser.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
		
	}
	if( recog_word == 1)
	{
		printf("\n OPENING FACEBOOK\n ");
		PlaySound(TEXT("instructions\\facebook.wav"), NULL, SND_SYNC);
		system("start https://www.facebook.com/");
	}
	if( recog_word == 2)
	{
		printf("\n OPENING NOTEPAD\n ");
		PlaySound(TEXT("instructions\\notepad.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\WINDOWS\\system32\\notepad.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}
	if( recog_word == 3)
	{
		printf("\n OPENING WORD\n ");
		PlaySound(TEXT("instructions\\word.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\Program Files (x86)\\Microsoft Office\\Office12\\WINWORD.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}
	if( recog_word == 4)
	{
		printf("\n OPENING PAINT\n ");
		PlaySound(TEXT("instructions\\paint.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\WINDOWS\\system32\\mspaint.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}
	if( recog_word == 5)
	{
		printf("\n OPENING MUSIC\n ");
		PlaySound(TEXT("instructions\\music.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}
	if( recog_word == 6)
	{
		printf("\n OPENING EXCEL\n ");
		PlaySound(TEXT("instructions\\excel.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\Program Files (x86)\\Microsoft Office\\Office12\\EXCEL.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}
	if( recog_word == 7)
	{
		printf("\n OPENING PHOTOS\n ");
		PlaySound(TEXT("instructions\\photos.wav"), NULL, SND_SYNC);
		STARTUPINFO startInfo={0};
		PROCESS_INFORMATION processInfo={0};
		BOOL bSuccess=CreateProcess(TEXT("C:\\Program Files (x86)\\FastStone Image Viewer\\FSViewer.exe"),NULL,NULL,NULL,FALSE,NULL,NULL,NULL,&startInfo,&processInfo);
	}

}
void hmm_live_training()
{
	FILE *fin_tokura, *fin_codebook;
	int i, j, option;
	long double curr_value;

	fin_tokura = fopen("HMM\\tokura_wts.txt", "r");
	fin_codebook = fopen("HMM\\word_codebook.txt", "r");
	//storing tokura weights
	i = 0;
	while(fscanf(fin_tokura, "%Lf\n", &curr_value) != EOF)
	{
		tokura_weights[i] = curr_value;
		i++;
	} 	

	// storing codebook generated from the universe created
	for(i = 0; i < codebook_size; i++)
	{
		for(j = 0; j < p; j++)
		{
			fscanf(fin_codebook, "%Lf", &curr_value);
			codebook[i][j] = curr_value;
		}
	}
	fclose(fin_codebook);
	fclose(fin_tokura);

	//training 
	
	if(training == 1 )
		training_words();

	//testing
	
	//testing();
	printf("\nLIVE TRAINING\n Utter the word 10 times\n");
	live_training();
	
}
void hmm_live_testing()
{
	FILE *fin_tokura, *fin_codebook;
	int i, j, option;
	long double curr_value;

	fin_tokura = fopen("HMM\\tokura_wts.txt", "r");
	fin_codebook = fopen("HMM\\word_codebook.txt", "r");
	//storing tokura weights
	i = 0;
	while(fscanf(fin_tokura, "%Lf\n", &curr_value) != EOF)
	{
		tokura_weights[i] = curr_value;
		i++;
	} 	

	// storing codebook generated from the universe created
	for(i = 0; i < codebook_size; i++)
	{
		for(j = 0; j < p; j++)
		{
			fscanf(fin_codebook, "%Lf", &curr_value);
			codebook[i][j] = curr_value;
		}
	}
	fclose(fin_codebook);
	fclose(fin_tokura);

	//training 
	
	if(training == 1 )
		training_words();

	//testing
	
	printf("\nLIVE TESTING\n ");
	live_testing();
}

