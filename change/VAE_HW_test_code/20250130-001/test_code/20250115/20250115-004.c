/* Project: Variational Auto Encoder             */
/* SW及びSW+HWの実装                             */
/* Press button 0. VAE Network : SW              */
/* Press button 1. VAE Network : SW+HW           */
/* Press button 2. Initialization : VAE          */
/* Press button 3. HW Test                       */

/*20250115-004の変更内容
FPGAでは学習を行わせない
MATLABで学習させた変数のみを利用する
直書き変更（SWのみ）
*/

#include <stdio.h>
#include "platform.h"
#include <xgpio.h>
#include "xparameters.h"
#include "sleep.h"

#include "xil_printf.h"
//#include "xsdps.h"
#include "xil_cache.h"
// Use ff.h to access SD card.
#include "ff.h"

#include <stdlib.h>
#include <math.h>

#define EPOCH (10000)

#define ETA (0.001)
#define BATCHSIZE (2)

#define NUM_K  (9)
#define NUM_X  (9)
#define NUM_A2 (2)
#define NUM_A3 (9)

#define FLTOFX(x) ((sint32)(x * 65536.0))
#define FXTOFL(x) ((double)(x / 65536.0))
#define FXTOFL8(x) ((double)(x / 16.0))
#define ITOFX(x) ((int)(x << 16))
#define FXTOI(x) ((int)(x >> 16))

double Uniform( void ){
	return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

double rand_normal( double mu, double sigma ){
	double z=sqrt( -2.0*log(Uniform()) ) * sin( 2.0*M_PI*Uniform() );
	return mu + sigma*z;
 }

void disp_menu(){
	printf("test\n");
	printf("\n\r");
	printf("****************************************\n\r");
	printf("Press button 0. VAE Network : SW        \n\r");
	printf("Press button 1. VAE Network : SW+HW     \n\r");
	printf("Press button 2. Initialization : VAE    \n\r");
	printf("Press button 3. HW Test                 \n\r");
	printf("****************************************\n\r");
}

void VAE_forward_929_SW(
		double w2_mean[][NUM_X], double b2_mean[],
		double w2_var[][NUM_X], double b2_var[],
		double w3[][NUM_A2], double b3[],
		double X[],
		double z2_mean[], double a2_mean[],
		double z2_var[], double a2_var[],
		double z[],
		double z3[], double a3[], double eps[])
{
	int i, j;

	/* z2，a2，z3，a3の初期化 */
	for(i = 0; i < NUM_A2; i++){
		z2_mean[i] = 0.0;    a2_mean[i] = 0.0;
		z2_var[i]  = 0.0;    a2_var[i]  = 0.0;
	}
	for(i = 0; i < NUM_A3; i++){
		z3[i] = 0.0;
		a3[i] = 0.0;
	}

	// z2_mean の計算
	// 行列演算が必要
	//  printf("z2 : \n\r");
	for(i = 0; i < NUM_A2; i++){
		for(j = 0; j < NUM_X; j++){
			z2_mean[i] = z2_mean[i] + w2_mean[i][j]*X[j];
		//      printf("0x%8x = 0x%8x + 0x%8x*0x%8x\n\r",
		//    		  *(int *)&z2[i], *(int *)&z2[i], *(int *)&w2[i][j], *(int *)&X[j]);
		}
		z2_mean[i] = z2_mean[i]+b2_mean[i];
	//    printf("\n\r0x%8x ", *(int *)&z2[i]);
	}

	// a2_mean の計算． Linear
	for(i = 0; i < NUM_A2; i++){
		a2_mean[i] = z2_mean[i];
	}

	for(i = 0; i < NUM_A2; i++){
		for(j = 0; j < NUM_X; j++){
			z2_var[i] = z2_var[i] + w2_var[i][j]*X[j];
		}
		z2_var[i] = z2_var[i]+b2_var[i];
	}

	// a2_var の計算． Softplus
	for(i = 0; i < NUM_A2; i++){
		a2_var[i] = log(1+exp(z2_var[i]));
	}

	// zの計算
	for(i = 0; i < NUM_A2; i++){
		z[i] = a2_mean[i] + sqrt(a2_var[i])*eps[i];
	}

	// z3 の計算
	// 行列計算を行っている
	for(i = 0; i < NUM_A3; i++){
		for(j = 0; j < NUM_A2; j++){
			z3[i] = z3[i] + w3[i][j]*z[j];
		}
		z3[i] = z3[i]+b3[i];
	}

	// a3 の計算．Sigmoid 関数の計算
	for(i = 0; i < NUM_A3; i++){
		a3[i] =  1.0/(1.0+exp(-1.0*z3[i]));
	}

	// a2, a3を戻す
	return;
}

void VAE_generate_929_SW(
		double w3[][NUM_A2], double b3[],
		double z[],
		double z3[], double a3[])
{
	int i, j;
	// z3 の計算
	// 行列計算を行っている
	for(i = 0; i < NUM_A3; i++){
		for(j = 0; j < NUM_A2; j++){
			z3[i] = z3[i] + w3[i][j]*z[j];
		}
		z3[i] = z3[i]+b3[i];
	}

	// a3 の計算．Sigmoid 関数の計算
	for(i = 0; i < NUM_A3; i++){
		a3[i] =  1.0/(1.0+exp(-1.0*z3[i]));
	}

	printf("z = ");
	for(i = 0; i < NUM_A2; i++){
		printf("%8f ", z[i]);
	}
	printf("\n\r");
	printf("a3 = \n\r");
	for(i = 0; i < NUM_A3; i++){
		printf("%8f ", a3[i]);
		if(i%3==2)
			printf("\n\r");
	}

	// a3を戻す
	return;
}

void AE_forward_929_HW(
		double w2[][NUM_X], double b2[], double w3[][NUM_A2], double b3[],
		double X[], double z[],
		double z2_hw[], double a2_hw[], double z3_hw[], double a3_hw[])
{

	int i, j, m, l;
	unsigned int offset_address;
	sint32 a3_tmp_hw, a2_tmp_hw;
	sint32 z3_tmp_hw, z2_tmp_hw;

	/* z2，a2，z3，a3の初期化 */
	for(i = 0; i < NUM_A2; i++){
		z2_hw[i] = 0.0;
		a2_hw[i] = 0.0;
	}
	for(i = 0; i < NUM_A3; i++){
		z3_hw[i] = 0.0;
		a3_hw[i] = 0.0;
	}
	// HW
	// Write data
	// xparameters.h
	// ae_sys_platform/export/ae_sys_platform/sw/ae_sys_platform/standalone_domain/bspinclude/include
	// XPAR_DUT_FORWA_IP_0_BASEADDR 0x43C00000
	// X (k)
	offset_address = 0x01BC;
	for(l = 0; l < NUM_X; l++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(X[l]);
		offset_address += 4;
	}
	// w2
	offset_address = 0x0100;
	for(l = 0; l < NUM_A2; l++){
		for(m = 0; m < NUM_X; m++){
		  *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(w2[l][m]);
		  offset_address += 4;
		}
	}
	// printf("Write W2.\n\r");

	// b2
	offset_address = 0x0148;
	*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(b2[0]);
	offset_address = 0x014C;
	*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(b2[1]);
	// printf("Write b2.\n\r");

	// Read data
	// z2
	offset_address = 0x01E0;
	z2_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	z2_hw[0] = FXTOFL(z2_tmp_hw);
	//               printf("0x%8x, 0x%8x\n\r", *(int *)&z2_tmp_hw, *(int *)&z2_hw[0]);

	offset_address = 0x01E4;
	z2_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	z2_hw[1] = FXTOFL(z2_tmp_hw);
	//               printf("0x%8x, 0x%8x\n\r", *(int *)&z2_tmp_hw, *(int *)&z2_hw[1]);
	//               printf("Read z2.\n\r");

	// a2
	//  printf("a2(HW) : ");
	j = 0;
	offset_address = 0x01E8;
	for(l = 0; l < NUM_A2; l++){
		a2_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
		a2_hw[l] = FXTOFL(a2_tmp_hw);
	//	    printf("0x%8x, ", *(int *)&a2_hw[l]);

		offset_address += 4;
	}
	//  printf("\n\r");
	//               printf("Read a2.\n\r");


	// z (k)
	offset_address = 0x0238;
	for(l = 0; l < NUM_A2; l++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(z[l]);
		offset_address += 4;
	}

	// w3
	offset_address = 0x0150;
	for(l = 0; l < NUM_A3; l++){
		for(m = 0; m < NUM_A2; m++){
		  *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(w3[l][m]);
		  offset_address += 4;
		}
	}
	// printf("Write w3.\n\r");

	// b3
	offset_address = 0x0198;
	for(m = 0; m < NUM_A3; m++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(b3[m]);
		offset_address += 4;
	}
	// printf("Write b3.\n\r");

	// a3
	//  printf("a3(HW) : ");
	j = 0;
	offset_address = 0x0214;
	for(l = 0; l < NUM_A3; l++){
	   a3_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	   a3_hw[l] = FXTOFL(a3_tmp_hw);
	//	   printf("0x%8x, ", *(int *)&a3_hw[l]);
	   offset_address += 4;
	}
	//  printf("\n\r");
	//               printf("Read a3.\n\r");

	// Read data
	// z3
	offset_address = 0x01F0;
	for(m = 0; m < NUM_A3; m++){
	  z3_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	  z3_hw[m] = FXTOFL8(z3_tmp_hw);
	  offset_address += 4;
	}
	//               printf("0x%8x, 0x%8x\n\r", *(int *)&z3_tmp_hw, *(int *)&z3_hw[0]);

	// a2_hw, a3_hwを戻す
	return;
}

void VAE_forward_929_HW(
		double w2_mean[][NUM_X], double b2_mean[],
		double w2_var[][NUM_X], double b2_var[],
		double w3[][NUM_A2], double b3[],
		double X[],
		double z2_mean_hw[], double a2_mean_hw[],
		double z2_var_hw[], double a2_var_hw[],
		double z[],
		double z3_hw[], double a3_hw[], double eps[])
{
	int i;
	double z2_mean_tmp_hw[NUM_A2], z2_var_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];
	double a2_mean_tmp_hw[NUM_A2], a2_var_tmp_hw[NUM_A2], a3_tmp_hw[NUM_A3];

    // Meanの計算
    z[0]=0.0;z[1]=0.0;
    AE_forward_929_HW(w2_mean, b2_mean, w3, b3, X, z, z2_mean_tmp_hw, a2_mean_tmp_hw, z3_tmp_hw, a3_tmp_hw);
    // a2_mean の計算． Linear
    //printf("VAE HW\n\r");
    for(i = 0; i < NUM_A2; i++){
  	  z2_mean_hw[i] = z2_mean_tmp_hw[i];
  	  a2_mean_hw[i] = z2_mean_tmp_hw[i];
  	  // printf(" %8f, \n\r", z2_mean_hw[i]);
    }

    // Varianceの計算
    AE_forward_929_HW(w2_var, b2_var, w3, b3, X, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
    // a2_mean の計算． Linear
    for(i = 0; i < NUM_A2; i++){
  	  z2_var_hw[i] = z2_var_tmp_hw[i];
    }
    // a2_var の計算． Softplus
    for(i = 0; i < NUM_A2; i++){
  	  a2_var_hw[i] = log(1+exp(z2_var_hw[i]));
    }
    // zの計算 (SW)
    for(i = 0; i < NUM_A2; i++){
  	  z[i] = a2_mean_hw[i] + sqrt(a2_var_hw[i])*eps[i];
    }
    // 画像の生成
    AE_forward_929_HW(w2_var, b2_var, w3, b3, X, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
    for(i = 0; i < NUM_A3; i++){
  	  z3_hw[i] = z3_tmp_hw[i];
  	  a3_hw[i] = a3_tmp_hw[i];
    }

    return;
}

void VAE_print_a2a3_sw(double a2_mean[], double a2_var[], double z[], double a3[])
{
	int i;

	printf("a2_mean : ");
	for(i = 0; i < NUM_A2; i++){			// a2を格納
	  printf("%8f, ", a2_mean[i]);
	}
	printf("\n\r");
	printf("a2_var : ");
	for(i = 0; i < NUM_A2; i++){			// a2を格納
	  printf("%8f, ", a2_var[i]);
	}
	printf("\n\r");
	printf("z : ");
	for(i = 0; i < NUM_A2; i++){			// a2を格納
	  printf("%8f, ", z[i]);
	}
	printf("\n\r");

	printf("a3 : ");
	for(i = 0; i < NUM_A3; i++){
	  printf("%8f, ", a3[i]);
	}
	printf("\n\r");
}

void VAE_print_a2a3(double a2_mean[], double a2_var[], double a3[])
{
	int i;

	printf("a2_mean : ");
	for(i = 0; i < NUM_A2; i++){			// a2を格納
	  printf("%8f, ", a2_mean[i]);
	}
	printf("\n\r");
	printf("a2_var : ");
	for(i = 0; i < NUM_A2; i++){			// a2を格納
	  printf("%8f, ", a2_var[i]);
	}
	printf("\n\r");

	printf("a3 : ");
	for(i = 0; i < NUM_A3; i++){
	  printf("%8f, ", a3[i]);
	}
	printf("\n\r");
}

void print_wb(double w2_mean[][NUM_X], double b2_mean[],
		double w2_var[][NUM_X], double b2_var[],
		double w3[][NUM_A2], double b3[])
{
  int i, j;

  /* W2_mean の表示 */
  printf("W2_mean\n\r");
  for(i = 0; i < NUM_A2; i++){
    for(j = 0; j < NUM_X; j++){
      // For PC
      // printf("%6.3f ", w2[i][j]);
      //
      // For embedded CPU
      printf("%8f ", w2_mean[i][j]);
    }
    printf("\n\r");
  }

  /* b2_mean の表示 */
  printf("b2_mean\n\r");
  for(j = 0; j <NUM_A2; j++){
    // For embedded CPU
    printf("%8f ", b2_mean[j]);
  }
  printf("\n\r");

  /* W2_var の表示 */
  printf("W2_var\n\r");
  for(i = 0; i < NUM_A2; i++){
    for(j = 0; j < NUM_X; j++){
      // For PC
      // printf("%6.3f ", w2[i][j]);
      //
      // For embedded CPU
      printf("%8f ", w2_var[i][j]);
    }
    printf("\n\r");
  }

  /* b2_var の表示 */
  printf("b2_var\n\r");
  for(j = 0; j <NUM_A2; j++){
    // For embedded CPU
    printf("%8f ", b2_var[j]);
  }
  printf("\n\r");

  /* W3 の表示 */
  printf("W3\n\r");
  for(i = 0; i < NUM_A3; i++){
    for(j = 0; j < NUM_A2; j++){
      // For PC
      // printf("%6.3f ", w3[i][j]);
      //
      // For embedded CPU
      printf("%8f ", w3[i][j]);
    }
    printf("\n\r");
  }

  /* b3 の表示 */
  printf("b3\n\r");
  for(j = 0; j < NUM_A3; j++){
    // For PC
    // printf("%6.3f ", b3[j]);
    //
    // For embedded CPU
    printf("%8f ", b3[j]);
  }
  printf("\n\r\n\r");
}

void print_a2_a3(double a2_mean_hw[], double a2_var_hw[], double a3_hw[], double a2_mean_sw[], double a2_var_sw[], double a3_sw[])
{
   int i;
   double diff_tmp;

   // a2_hw, a2_sw, diff
   printf("   a2_mean_hw  ,    a2_mean_sw  ,  Diff    \n\r");
   for(i = 0; i < NUM_A2; i++){
     diff_tmp = a2_mean_hw[i] - a2_mean_sw[i];
     printf("%8f, %8f, %8f\n\r", a2_mean_hw[i], a2_mean_sw[i], diff_tmp);
   }
   printf("   a2_var_hw  ,    a2_var_sw  ,  Diff    \n\r");
   for(i = 0; i < NUM_A2; i++){
     diff_tmp = a2_var_hw[i] - a2_var_sw[i];
     printf("%8f, %8f, %8f\n\r", a2_var_hw[i], a2_var_sw[i], diff_tmp);
   }
   // a2_hw, a2_sw, diff
   printf("   a3_hw  ,    a3_sw  ,  Diff    \n\r");
   for(i = 0; i < NUM_A3; i++){
     diff_tmp = a3_hw[i] - a3_sw[i];
     printf("%8f, %8f, %8f\n\r", a3_hw[i], a3_sw[i], diff_tmp);
   }

   return ;
}


int main()
{
   XGpio input, output;
   int button_data = 0;
   int switch_data = 0;

   int i, j, l, m, loop;

   //変更させる予定
   //初めのweightはMATLABの出力結果を使用予定
   double w2_mean_init[NUM_A2][NUM_X]  =
   {{0.00061, -0.0092, -0.0021, -0.0087, 0.0089, -0.0058, -0.1971, 1.4054, 0.3721},
	{-0.0078, 0.0012, -0.0063, 0.0017, -0.0083, 0.00047, 0.5381, -0.5612, 0.6816}};
   double w2_var_init[NUM_A2][NUM_X]  =
   {{-0.8178, -0.0893, -0.6769, 0.1298, -0.3421, -0.5389, -1.0345, -0.2477, -1.2913},
	{-1.0118, 0.0936, -1.1137, 0.0569, -0.1882, -0.7672, -0.5465, 0.00071, -1.2532}};

   double w3_init[NUM_A3][NUM_A2] =
                 	{{1.2488, 2.6317},
					{4.4647, -1.3165},
					{1.2322, 2.4744},
					{4.4139, -1.4120},
					{-4.0184, 2.2693},
					{4.5003, -1.2044},
					{1.3060, 2.7210},
					{4.4511, -1.3656},
					{1.1667, 2.3886}};
   double eps_init[NUM_A2]=
   {-0.601121103667131,-3.410037246733799};

   double b2_mean_init[NUM_A2]         = {-0.7388, -0.7314};
   double b2_var_init[NUM_A2]          = {-1.8587, -2.0374};
   double b3_init[NUM_A3]         = {3.2619, -0.4094, 3.3225, -0.3711, 0.0538, -0.4448, 3.2270, -0.3946, 3.3549};
   double k[9][BATCHSIZE]    = {{1.0, 1.0},
                                {1.0, 0.0},
							    {1.0, 1.0},
							    {1.0, 0.0},
							    {0.0, 1.0},
							    {1.0, 0.0},
							    {1.0, 1.0},
							    {1.0, 0.0},
							    {1.0, 1.0}};

   double t[9][BATCHSIZE]    = {{1.0, 1.0},
								{1.0, 0.0},
								{1.0, 1.0},
								{1.0, 0.0},
								{0.0, 1.0},
								{1.0, 0.0},
								{1.0, 1.0},
								{1.0, 0.0},
								{1.0, 1.0}};

	double w2_mean[NUM_A2][NUM_X], w2_var[NUM_A2][NUM_X];
	double w3[NUM_A3][NUM_A2];
	double b2_mean[NUM_A2], b2_var[NUM_A2];
	double b3[NUM_A3];
	double eps[NUM_A2];
	double z[NUM_A2];

	double X[NUM_X];
	double z2_mean[NUM_A2][BATCHSIZE], z2_var[NUM_A2][BATCHSIZE], z3[NUM_A3][BATCHSIZE];
	double z2_mean_tmp[NUM_A2], z2_var_tmp[NUM_A2], z3_tmp[NUM_A3];
	double z_sw[NUM_A2][BATCHSIZE];
	double a2_mean[NUM_A2][BATCHSIZE], a2_var[NUM_A2][BATCHSIZE], a3[NUM_A3][BATCHSIZE];
	double a2_mean_tmp[NUM_A2], a2_var_tmp[NUM_A2], a3_tmp[NUM_A3];


	double z_hw[NUM_A2];
	double z2_mean_hw[NUM_A2], z2_var_hw[NUM_A2], z3_hw[NUM_A3];
	double z2_mean_tmp_hw[NUM_A2], z2_var_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];
	double a2_mean_hw[NUM_A2], a2_var_hw[NUM_A2], a3_hw[NUM_A3];
	double a2_mean_tmp_hw[NUM_A2], a2_var_tmp_hw[NUM_A2], a3_tmp_hw[NUM_A3];

	double dCda3[NUM_A3][BATCHSIZE], dCdz3[NUM_A3][BATCHSIZE], delta3[NUM_A3][BATCHSIZE];
	double dCdb3[NUM_A3], dCdw3[NUM_A3][NUM_A2];
	double dCda2_mean[NUM_A2][BATCHSIZE], dCdz2_mean[NUM_A2][BATCHSIZE], delta2_mean[NUM_A2][BATCHSIZE];
	double dCda2_var[NUM_A2][BATCHSIZE], dCdz2_var[NUM_A2][BATCHSIZE], delta2_var[NUM_A2][BATCHSIZE];
	double dCdb2_mean[NUM_A2], dCdw2_mean[NUM_A2][NUM_X];
	double dCdb2_var[NUM_A2], dCdw2_var[NUM_A2][NUM_X];
	double mult_tmp;


	double eta = ETA;

	XGpio_Initialize(&input, XPAR_AXI_GPIO_0_DEVICE_ID);		//initialize input XGpio variable
	XGpio_Initialize(&output, XPAR_AXI_GPIO_1_DEVICE_ID);	//initialize output XGpio variable

	XGpio_SetDataDirection(&input, 1, 0xF);			//set first channel tristate buffer to input
	XGpio_SetDataDirection(&input, 2, 0xF);			//set second channel tristate buffer to input

	XGpio_SetDataDirection(&output, 1, 0x0);			//set first channel tristate buffer to output

	init_platform();

   // w2, b2, w3, b3の初期化
   for(l=0; l<NUM_A2; l++){
	   for(m = 0; m < NUM_X; m++){
		   w2_mean[l][m] = w2_mean_init[l][m];
		   w2_var[l][m] = w2_var_init[l][m];
	   }
   }
   for(l=0; l<NUM_A3; l++){
	   for(m = 0; m < NUM_A2; m++){
		   w3[l][m] = w3_init[l][m];
	   }
   }
   for(l=0; l<NUM_A2; l++){
	   b2_mean[l] = b2_mean_init[l];
	   b2_var[l] = b2_var_init[l];
   }
   for(l=0; l<NUM_A3; l++){
	   b3[l] = b3_init[l];
   }
   for(l=0; l<NUM_A2; l++){
	   eps[l] = eps_init[l];
   }

   printf("Start VAE (LSI Design Contest).\n\r");
   // メニュー表示
   disp_menu();

   	while(1){
      switch_data = XGpio_DiscreteRead(&input, 2);	//get switch data

      XGpio_DiscreteWrite(&output, 1, switch_data);	//write switch data to the LEDs

      button_data = XGpio_DiscreteRead(&input, 1);	//get button data

      //print message dependent on whether one or more buttons are pressed
      	if(button_data == 0b0000){} //do nothing

      	else if(button_data == 0b0001){
			printf("button 0 pressed\n\r");
			printf("AE Network (SW).\n\r");

			printf("Initial Weight \n\r");
			print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);
			/*
			for(loop = 0; loop < EPOCH; loop++){
				// Evaluation
				if(loop % 1000 == 0)
					printf("Epoch = %d\n", loop);

				for(j = 0; j < BATCHSIZE; j++){
					for(i = 0; i < NUM_X; i++)          // 入力
					X[i] = k[i][j];

					for(l=0; l<NUM_A2; l++){
					eps[l] = Uniform();
					}
					VAE_forward_929_SW(w2_mean, b2_mean, w2_var, b2_var,
							w3, b3, X,
							z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
							z, z3_tmp, a3_tmp, eps);
					//VAE_print_a2a3(a2_mean_tmp, a2_var_tmp, a3_tmp);

					for(i = 0; i < NUM_A2; i++){			// a2を格納
					a2_mean[i][j] = a2_mean_tmp[i];
					a2_var[i][j] = a2_var_tmp[i];
					z_sw[i][j] = z[i];
					}
		//               printf("a2 : 0x%8x, 0x%8x\n\r", *(int *)&a2[0][j], *(int *)&a2[1][j]);
					for(i = 0; i < NUM_A3; i++){
					a3[i][j] = a3_tmp[i];										// a3を格納

					dCda3[i][j] = -X[i]/a3[i][j]+(1.0-X[i])/(1.0-a3[i][j]);	// dC/da3　の計算
					dCdz3[i][j] = dCda3[i][j] * a3[i][j] * (1.0-a3[i][j]);	// dC/dz3 の計算
					delta3[i][j] = dCdz3[i][j];								// δ3　の計算
					}
				}

				for(i = 0; i < NUM_A3; i++){          	// バイアスb3の勾配 (3)式
				dCdb3[i] = 0.0;
				for(j = 0; j < BATCHSIZE; j++){
					dCdb3[i] += delta3[i][j];
				}
				}
		//            printf("dCdb3 : 0x%8x, 0x%8x\n\r", *(int *)&dCdb3[0], *(int *)&dCdb3[1]);

				for(i = 0; i < NUM_A2; i++){          	// 重みw3の勾配 (2)式
				for(j = 0; j < NUM_A3; j++){
					dCdw3[j][i] = 0.0;
					for(l = 0; l < BATCHSIZE; l++){
						dCdw3[j][i] += (z_sw[i][l] * delta3[j][l]);
					}
				}
				}
				// dC/da2_mean, dC/da2_var
				for(i = 0; i < NUM_A2; i++){    		// 重みw3の勾配 (2)式
				for(l = 0; l < BATCHSIZE; l++){
					dCda2_mean[i][l] = 0.0;
					dCda2_var[i][l] = 0.0;
					for(j = 0; j < NUM_A3; j++){
						dCda2_mean[i][l] += (w3[j][i] * delta3[j][l]);
						dCda2_var[i][l] += (w3[j][i] * delta3[j][l]);
					}
					dCda2_mean[i][l] += a2_mean[i][l];
					dCda2_var[i][l] += (a2_var[i][l]-1/a2_var[i][l]);
				}
				}

				for(i = 0; i < NUM_A2; i++){          	// 隠れ層の誤差 meanは線形
				for(l = 0; l < BATCHSIZE; l++){
					dCdz2_mean[i][l] = dCda2_mean[i][l];
					delta2_mean[i][l] = dCdz2_mean[i][l];

					dCdz2_var[i][l] = 1./(1+exp(-dCda2_var[i][l]));
					delta2_var[i][l] = dCdz2_var[i][l];
				}
				}

				for(i = 0; i < NUM_A2; i++){          	// バイアスb2の勾配 (6)式
				dCdb2_mean[i] = 0.0;
				dCdb2_var[i] = 0.0;
				for(l = 0; l < BATCHSIZE; l++){
					dCdb2_mean[i] += delta2_mean[i][l];
					dCdb2_var[i] += delta2_var[i][l];
				}
				}

				for(i = 0; i < NUM_X; i++){          	// 重みw2の勾配 (5)式
				for(j = 0; j < NUM_A2; j++){
					dCdw2_mean[j][i] = 0.0;
					dCdw2_var[j][i] = 0.0;
					for(l = 0; l < BATCHSIZE; l++){
						dCdw2_mean[j][i] += (k[i][l] * delta2_mean[j][l]);
						dCdw2_var[j][i] += (k[i][l] * delta2_var[j][l]);
					}
				}
				}
				// パラメータの更新
				for(i = 0; i < NUM_A3; i++){
				b3[i] = b3[i] - eta*dCdb3[i];				// b3 の更新

				for(j = 0; j < NUM_A2; j++){
					w3[i][j] = w3[i][j] - eta * dCdw3[i][j];	// w3 の更新
				}
				}

				for(i = 0; i < NUM_A2; i++){
				b2_mean[i] = b2_mean[i] - eta*dCdb2_mean[i];				// b2 の更新
				b2_var[i] = b2_var[i] - eta*dCdb2_var[i];				// b2 の更新

				for(j = 0; j < NUM_X; j++){
					w2_mean[i][j] = w2_mean[i][j] - eta * dCdw2_mean[i][j];	// w2 の更新
					w2_var[i][j] = w2_var[i][j] - eta * dCdw2_var[i][j];	// w2 の更新
				}
				}
			}
			*/
			//上記の更新を無視

			printf("Final Weight \n\r");
			print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);

			for(j = 0; j < BATCHSIZE; j++){
				for(i = 0; i < NUM_X; i++)          // 入力
				X[i] = k[i][j];

				for(l=0; l<NUM_A2; l++){
				eps[l] = Uniform();
				}
				VAE_forward_929_SW(w2_mean, b2_mean, w2_var, b2_var,
						w3, b3, X,
						z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
						z, z3_tmp, a3_tmp, eps);
				VAE_print_a2a3_sw(a2_mean_tmp, a2_var_tmp, z, a3_tmp);
			}

			printf("\n\rGenerate Image\n\r");
			z[0] = -0.5; z[1] = 1.0;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 0.0; z[1] = 0.5;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.0; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.2; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.4; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.5; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);


			disp_menu();
		}else if(button_data == 0b0010){
			printf("button 1 pressed\n\r");
			printf("Variational Auto Encoder (SW+HW).\n\r");

			printf("Initial Weight \n\r");
			print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);

				
			for(loop = 0; loop < EPOCH; loop++){
				// Evaluation
				if(loop % 1000 == 0)
					printf("Epoch = %d\n", loop);

				for(j = 0; j < BATCHSIZE; j++){
					for(i = 0; i < NUM_X; i++)          // 入力
					X[i] = k[i][j];

					for(l=0; l<NUM_A2; l++){
					eps[l] = Uniform();
					}
					VAE_forward_929_HW(w2_mean, b2_mean, w2_var, b2_var,
							w3, b3, X,
							z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
							z, z3_tmp, a3_tmp, eps);
					//VAE_print_a2a3(a2_mean_tmp, a2_var_tmp, a3_tmp);

					for(i = 0; i < NUM_A2; i++){			// a2を格納
					a2_mean[i][j] = a2_mean_tmp[i];
					a2_var[i][j] = a2_var_tmp[i];
					z_sw[i][j] = z[i];
					}
		//               printf("a2 : 0x%8x, 0x%8x\n\r", *(int *)&a2[0][j], *(int *)&a2[1][j]);
					for(i = 0; i < NUM_A3; i++){
					a3[i][j] = a3_tmp[i];										// a3を格納

					dCda3[i][j] = -X[i]/a3[i][j]+(1.0-X[i])/(1.0-a3[i][j]);	// dC/da3　の計算
					dCdz3[i][j] = dCda3[i][j] * a3[i][j] * (1.0-a3[i][j]);	// dC/dz3 の計算
					delta3[i][j] = dCdz3[i][j];								// δ3　の計算
					}
				}

				for(i = 0; i < NUM_A3; i++){          	// バイアスb3の勾配 (3)式
				dCdb3[i] = 0.0;
				for(j = 0; j < BATCHSIZE; j++){
					dCdb3[i] += delta3[i][j];
				}
				}
		//            printf("dCdb3 : 0x%8x, 0x%8x\n\r", *(int *)&dCdb3[0], *(int *)&dCdb3[1]);

				for(i = 0; i < NUM_A2; i++){          	// 重みw3の勾配 (2)式
				for(j = 0; j < NUM_A3; j++){
					dCdw3[j][i] = 0.0;
					for(l = 0; l < BATCHSIZE; l++){
						dCdw3[j][i] += (z_sw[i][l] * delta3[j][l]);
					}
				}
				}
				// dC/da2_mean, dC/da2_var
				for(i = 0; i < NUM_A2; i++){    		// 重みw3の勾配 (2)式
				for(l = 0; l < BATCHSIZE; l++){
					dCda2_mean[i][l] = 0.0;
					dCda2_var[i][l] = 0.0;
					for(j = 0; j < NUM_A3; j++){
						dCda2_mean[i][l] += (w3[j][i] * delta3[j][l]);
						dCda2_var[i][l] += (w3[j][i] * delta3[j][l]);
					}
					dCda2_mean[i][l] += a2_mean[i][l];
					dCda2_var[i][l] += (a2_var[i][l]-1/a2_var[i][l]);
				}
				}

				for(i = 0; i < NUM_A2; i++){          	// 隠れ層の誤差 meanは線形
				for(l = 0; l < BATCHSIZE; l++){
					dCdz2_mean[i][l] = dCda2_mean[i][l];
					delta2_mean[i][l] = dCdz2_mean[i][l];

					dCdz2_var[i][l] = 1./(1+exp(-dCda2_var[i][l]));
					delta2_var[i][l] = dCdz2_var[i][l];
				}
				}

				for(i = 0; i < NUM_A2; i++){          	// バイアスb2の勾配 (6)式
				dCdb2_mean[i] = 0.0;
				dCdb2_var[i] = 0.0;
				for(l = 0; l < BATCHSIZE; l++){
					dCdb2_mean[i] += delta2_mean[i][l];
					dCdb2_var[i] += delta2_var[i][l];
				}
				}

				for(i = 0; i < NUM_X; i++){          	// 重みw2の勾配 (5)式
				for(j = 0; j < NUM_A2; j++){
					dCdw2_mean[j][i] = 0.0;
					dCdw2_var[j][i] = 0.0;
					for(l = 0; l < BATCHSIZE; l++){
						dCdw2_mean[j][i] += (k[i][l] * delta2_mean[j][l]);
						dCdw2_var[j][i] += (k[i][l] * delta2_var[j][l]);
					}
				}
				}
				// パラメータの更新
				for(i = 0; i < NUM_A3; i++){
				b3[i] = b3[i] - eta*dCdb3[i];				// b3 の更新

				for(j = 0; j < NUM_A2; j++){
					w3[i][j] = w3[i][j] - eta * dCdw3[i][j];	// w3 の更新
				}
				}

				for(i = 0; i < NUM_A2; i++){
				b2_mean[i] = b2_mean[i] - eta*dCdb2_mean[i];				// b2 の更新
				b2_var[i] = b2_var[i] - eta*dCdb2_var[i];				// b2 の更新

				for(j = 0; j < NUM_X; j++){
					w2_mean[i][j] = w2_mean[i][j] - eta * dCdw2_mean[i][j];	// w2 の更新
					w2_var[i][j] = w2_var[i][j] - eta * dCdw2_var[i][j];	// w2 の更新
				}
				}
			}
			printf("Final Weight \n\r");
			print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);

			printf("Final Weight \n\r");
			print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);

			for(j = 0; j < BATCHSIZE; j++){
				for(i = 0; i < NUM_X; i++)          // 入力
				X[i] = k[i][j];

				for(l=0; l<NUM_A2; l++){
				eps[l] = Uniform();
				}
				VAE_forward_929_HW(w2_mean, b2_mean, w2_var, b2_var,
						w3, b3, X,
						z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
						z, z3_tmp, a3_tmp, eps);
				VAE_print_a2a3_sw(a2_mean_tmp, a2_var_tmp, z, a3_tmp);
			}

			printf("\n\rGenerate Image\n\r");
			z[0] = -0.5; z[1] = 1.0;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 0.0; z[1] = 0.5;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.0; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.2; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.4; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);
			z[0] = 1.5; z[1] = 0.2;
				VAE_generate_929_SW(w3, b3, z, z3_tmp, a3_tmp);

			disp_menu();
      }else if(button_data == 0b0100) {
         xil_printf("button 2 pressed\n\r");
         xil_printf("Initialization\n\r");

         // w2, b2, w3, b3の初期化
         for(l=0; l<NUM_A2; l++){
      	   for(m = 0; m < NUM_X; m++){
      		   w2_mean[l][m] = w2_mean_init[l][m];
      		   w2_var[l][m] = w2_var_init[l][m];
      	   }
         }
         for(l=0; l<NUM_A3; l++){
      	   for(m = 0; m < NUM_A2; m++){
      		   w3[l][m] = w3_init[l][m];
      	   }
         }
         for(l=0; l<NUM_A2; l++){
      	   b2_mean[l] = b2_mean_init[l];
      	   b2_var[l] = b2_var_init[l];
         }
         for(l=0; l<NUM_A3; l++){
      	   b3[l] = b3_init[l];
         }
         for(l=0; l<NUM_A2; l++){
      	   eps[l] = eps_init[l];
         }

         disp_menu();
      }else if(button_data == 0b1000){
          printf("button 3 pressed\n\r");
          printf("HW test.\n\r");

          printf("Initial Weight \n\r");
          print_wb(w2_mean, b2_mean, w2_var, b2_var, w3, b3);

          j = 0;
          for(i = 0; i < NUM_X; i++){          // 入力の設定
        	  X[i] = k[i][j];
          }

          // SW
          VAE_forward_929_SW(w2_mean, b2_mean, w2_var, b2_var,
          		w3, b3, X,
				z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
				z, z3_tmp, a3_tmp, eps);

          j = 0;
          printf("a2_mean(SW) : ");
          for(i = 0; i < NUM_A2; i++){			// a2を格納
        	  a2_mean[i][j] = a2_mean_tmp[i];
        	  printf("%8f, ", a2_mean[i][j]);
          }
          printf("\n\r");
          printf("a2_var(SW) : ");
          for(i = 0; i < NUM_A2; i++){			// a2を格納
        	  a2_var[i][j] = a2_var_tmp[i];
        	  printf("%8f, ", a2_var[i][j]);
          }
          printf("\n\r");

          j = 0;
          printf("a3(SW) : ");
          for(i = 0; i < NUM_A3; i++){
        	  a3[i][j] = a3_tmp[i];										// a3を格納
        	  printf("%8f, ", a3[i][j]);
          }
          printf("\n\r");

          VAE_forward_929_HW(w2_mean, b2_mean, w2_var, b2_var, w3, b3,
          		X, z2_mean_hw, a2_mean_hw, z2_var_hw, a2_var_hw,
          		z, z3_hw, a3_hw, eps);

/*          // Meanの計算
          z[0]=0.0;z[1]=0.0;
          AE_forward_929_HW(w2_mean, b2_mean, w3, b3, X, z, z2_mean_tmp_hw, a2_mean_tmp_hw, z3_tmp_hw, a3_tmp_hw);
          // a2_mean の計算． Linear

          for(i = 0; i < NUM_A2; i++){
        	  z2_mean_hw[i] = z2_mean_tmp_hw[i];
        	  a2_mean_hw[i] = z2_mean_tmp_hw[i];
        	  // printf(" %8f, \n\r", z2_mean_hw[i]);
          }

          // Varianceの計算
          AE_forward_929_HW(w2_var, b2_var, w3, b3, X, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
          // a2_mean の計算． Linear
          for(i = 0; i < NUM_A2; i++){
        	  z2_var_hw[i] = z2_var_tmp_hw[i];
          }
          // a2_var の計算． Softplus
          for(i = 0; i < NUM_A2; i++){
        	  a2_var_hw[i] = log(1+exp(z2_var_hw[i]));
          }
          // zの計算 (SW)
          for(i = 0; i < NUM_A2; i++){
        	  z[i] = a2_mean_hw[i] + sqrt(a2_var_hw[i])*eps[i];
          }
          // 画像の生成
          AE_forward_929_HW(w2_var, b2_var, w3, b3, X, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
          for(i = 0; i < NUM_A3; i++){
        	  z3_hw[i] = z3_tmp_hw[i];
        	  a3_hw[i] = a3_tmp_hw[i];
          }
*/

          printf("VAE HW\n\r");
          j = 0;
          printf("a2_mean(HW) : ");
          for(i = 0; i < NUM_A2; i++){			// a2を格納
        	  printf("%8f, ", a2_mean_hw[i]);
          }
          printf("\n\r");
          printf("a2_var(HW) : ");
          for(i = 0; i < NUM_A2; i++){			// a2を格納
        	  printf("%8f, ", a2_var_hw[i]);
          }
          printf("\n\r");

          j = 0;
          printf("a3(HW) : ");
          for(i = 0; i < NUM_A3; i++){
        	  printf("%8f, ", a3_hw[i]);
          }
          printf("\n\r");

          disp_menu();

      }
      else
         xil_printf("multiple buttons pressed\n\r");

      usleep(200000);			//delay

   }
   cleanup_platform();
   return 0;
}
