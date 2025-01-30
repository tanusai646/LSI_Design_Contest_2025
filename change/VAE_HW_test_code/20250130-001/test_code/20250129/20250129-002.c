/* Project: Variational Auto Encoder             */
/* SW及びSW+HWの実装                             */
/* Press button 0. VAE Network : SW              */
/* Press button 1. VAE Network : SW+HW           */
/* Press button 2. Initialization : VAE          */
/* Press button 3. HW Test                       */

/*20250129-001の変更内容
20250122-002>>
ハードウェアを変更したので、それに適応するようにプログラムを大幅変更
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
#include <string.h>
#include <math.h>


#define BATCHSIZE (2)

#define NUM_K  (16)
#define NUM_X  (256)
#define NUM_A2 (2)
#define NUM_A3 (16)
#define NUM_IN (256)

#define FLTOFX(x) ((sint32)(x * 65536.0))
#define FXTOFL(x) ((double)(x / 65536.0))
#define FXTOFY(x) ((double)(x / 16777216.0))
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
	printf("\n\r");
	printf("\n\r");
	printf("\n\r");
	printf("\n\r");
	printf("****************************************\n\r");
	printf("Press button 0. Read Data               \n\r");
	printf("Press button 1. Use 256*2*256 VAE       \n\r");
	printf("Press button 2. No pre                  \n\r");
	printf("Press button 3. No pre                  \n\r");
	printf("****************************************\n\r");
}





void AE_forward_929_HW(double w2[NUM_A2][NUM_A3], double X[NUM_A3], double b2[NUM_A2], double z2_hw[NUM_A2])
{

	int i, j, m, l;
	unsigned int offset_address;
	sint32 z2_tmp_hw;

	/* z2，a2，z3，a3の初期化 */
	for(i = 0; i < NUM_A2; i++){
		z2_hw[i] = 0.0;
	}
	// HW
	// Write data
	// xparameters.h
	// ae_sys_platform/export/ae_sys_platform/sw/ae_sys_platform/standalone_domain/bspinclude/include/xparameters.h
	// XPAR_DUT_FORWA_IP_0_BASEADDR 0x43C00000


	// w2_1
	offset_address = 0x0100;
	for(l = 0; l < NUM_A3; l++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(w2[0][l]);
		offset_address += 4;
		/*
		printf("w2_mean write.\n\r");
		printf("%f, %d\n\r",w2[0][l], FLTOFX(w2[0][l]));
		printf("write finished.\n\r");
		*/
	}
	// printf("Write W2.\n\r");

	// w2_2
	offset_address = 0x0140;
	for(l = 0; l < NUM_A3; l++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(w2[1][l]);
		offset_address += 4;
		/*
		printf("w2_var write.\n\r");
		printf("%f, %d\n\r",w2[1][l], FLTOFX(w2[1][l]));
		printf("write finished.\n\r");
		*/
	}
	// printf("Write W2.\n\r");

	// b2
	offset_address = 0x0180;
	*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(b2[0]);
	offset_address = 0x0184;
	*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(b2[1]);
	// printf("Write b2.\n\r");


	// X (k)
	offset_address = 0x0248;
	for(l = 0; l < NUM_A3; l++){
		*((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address))=FLTOFX(X[l]);
		offset_address += 4;
		/*
		printf("write X[l]\n\r");
		printf("%f, %d\n\r",X[l], FLTOFX(X[l]));
		printf("write finished\n\r");
		*/
	}


	// Read data
	// z2
	offset_address = 0x0290;
	z2_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	z2_hw[0] = FXTOFL(z2_tmp_hw);
	//printf("z2 test write.\n\r");
	//printf("0x%8x, 0x%8x\n\r", *(int *)&z2_tmp_hw, *(int *)&z2_hw[0]);
	//printf("%d, %f\n\r", z2_tmp_hw, z2_hw[0]);
	//printf("test finish.\n\r");

	offset_address = 0x0294;
	z2_tmp_hw = *((volatile unsigned int*) (XPAR_DUT_FORWA_IP_0_BASEADDR+offset_address));
	z2_hw[1] = FXTOFL(z2_tmp_hw);
	//               printf("0x%8x, 0x%8x\n\r", *(int *)&z2_tmp_hw, *(int *)&z2_hw[1]);
	//               printf("Read z2.\n\r");

	return;
}


void VAE_forward_2562256_HW(
	double w2_mean[][NUM_X], double b2_mean[],
	double w2_var[][NUM_X], double b2_var[],
	double w3[][NUM_A2], double b3[],
	double X[],
	double z2_mean_hw[], double a2_mean_hw[],
	double z2_var_hw[], double a2_var_hw[],
	double z[],
	double z3_hw[], double a3_hw[], double eps[])
{
	int i, j, k, l;
	/*
	// 929のHWに乗せるために調整用変数
	double w2_mean9[NUM_A2][NUM_A3];
	double w2_var9[NUM_A2][NUM_A3];
	double w39[NUM_A3][NUM_A2];
	double X9[NUM_A3];
	double b2_mean9[NUM_A2], b2_var9[NUM_A2];
	double b39[NUM_A3];

	// 929のHWからの出力取得
	double z2_mean_tmp_hw[NUM_A2], z2_var_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];
	double a2_mean_tmp_hw[NUM_A2], a2_var_tmp_hw[NUM_A2], a3_tmp_hw[NUM_A3];

	// 929のHWからの出力保管用
	double sum_z2_mean[NUM_A2], sum_z2_var[NUM_A2];

	sum_z2_mean[0] = 0.0; sum_z2_mean[1] = 0.0;
	sum_z2_var[0] = 0.0; sum_z2_var[1] = 0.0;
	z[0] = 0.0; z[1] = 0.0;

	//meanの計算
	for(i = 0; i < 32; i++){
		k = 0;
		// 929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w2_mean9[0][k] = w2_mean[0][j];
			w2_mean9[1][k] = w2_mean[1][j];

			X9[k] = X[j];
			k++;
		}
		w2_mean9[0][8] = 0.0; w2_mean9[1][8] = 0.0;
		X9[8] = 1.0;
		b2_mean9[0] = 0.0; b2_mean9[1] = 0.0;

		// AEを使用
		AE_forward_929_HW(w2_mean9, X9, b2_mean9, z2_mean_tmp_hw);
		//printf("%7.4f %7.4f %7.4f\n\r", z2_mean_tmp_hw[0], z2_mean_tmp_hw[1], sum);
		sum_z2_mean[0] = sum_z2_mean[0] + z2_mean_tmp_hw[0];
		sum_z2_mean[1] = sum_z2_mean[1] + z2_mean_tmp_hw[1];
		//printf("%7.4f %7.4f\n\r", sum_z2_mean[0], sum_z2_mean[1]);

	}

	for(i = 0; i < NUM_A2; i++){
		z2_mean_tmp_hw[i] = sum_z2_mean[i] + b2_mean[i];
		z2_mean_tmp_hw[i] = z2_mean_tmp_hw[i] / 16;
		//printf("b2_mean[%d]: %f, sum_z2_mean[%d]: %f\n\r", i, b2_mean[i], i, sum_z2_mean[i]);
		printf("z2_mean_hw[%d]: %f\n\r", i, z2_mean_tmp_hw[i]);
	}


	//varの計算
	for(i = 0; i < 32; i++){
		k = 0;
		// 929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w2_var9[0][k] = w2_var[0][j];
			w2_var9[1][k] = w2_var[1][j];

			X9[k] = X[j];
			k++;
		}
		w2_var9[0][8] = 0.0; w2_var9[1][8] = 0.0;
		X9[8] = 1.0;
		b2_var9[0] = 0.0; b2_var9[1] = 0.0;

		// AEを使用
		AE_forward_929_HW(w2_var9, b2_var9, w3, b3, X9, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		//printf("%7.4f %7.4f %7.4f\n\r", z2_var_tmp_hw[0], z2_var_tmp_hw[1], sum);
		sum_z2_var[0] = sum_z2_var[0] + z2_var_tmp_hw[0];
		sum_z2_var[1] = sum_z2_var[1] + z2_var_tmp_hw[1];
		//printf("%7.4f %7.4f\n\r", sum_z2_var[0], sum_z2_var[1]);

	}

	for(i = 0; i < NUM_A2; i++){
		z2_var_tmp_hw[i] = sum_z2_var[i] + b2_var[i];
		z2_var_tmp_hw[i] = z2_var_tmp_hw[i] / 16;
		//printf("b2_var[%d]: %f, sum_z2_var[%d]: %f\n\r", i, b2_var[i], i, sum_z2_var[i]);
		printf("z2_var_hw[%d]: %f\n\r", i, z2_var_tmp_hw[i]);
	}

	//meanとvarを活性化関数に代入
	//zの演算
	for(i = 0; i < NUM_A2; i++){
		z2_mean_hw[i] = z2_mean_tmp_hw[i];
		a2_mean_hw[i] = z2_mean_tmp_hw[i];
		z2_var_hw[i] = z2_var_tmp_hw[i];
		a2_var_hw[i] = log(1 + exp(z2_var_tmp_hw[i]));

		z[i] = a2_mean_hw[i] + sqrt(a2_var_hw[i])*eps[i];
		printf("%f\n\r", z[i]);
	}

	/*
	// 画像の生成
	for(i = 0; i < 32; i++){
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w39[k][0] = w3[j][0];
			w39[k][1] = w3[j][1];
			b39[k] = b3[j];
			//printf("w39 %f %f\n\r", w39[k][0], w39[k][1]);
			//printf("b39 %f \n\r", b39[k]);
			k++;
		}
		w39[8][0] = 1.0; w39[8][1] = 1.0;
		b39[8] = 1.0;

		AE_forward_929_HW(w2_mean9, b2_mean9, w39, b39, X9, z, z2_mean_tmp_hw, a2_mean_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			z3_hw[j] = z3_tmp_hw[k]/16.0;
			a3_hw[j] = 1 / (1 + exp(-z3_hw[j]));
			//printf("%f ", z3_hw[j]);
			k++;
		}
		//printf("\n\r");
	}
	*/

	return;
}


void VAE_forward_25616_HW(
	double w2_mean[][NUM_X], double b2_mean[],
	double w2_var[][NUM_X], double b2_var[],
	double w3[][NUM_A3], double b3[],
	double X[],
	double z2_mean_hw[], double a2_mean_hw[],
	double z2_var_hw[], double a2_var_hw[],
	double z[],
	double z3_hw[], double a3_hw[], double eps[])
{
	int i, j, k, l;

	// 162のHWに乗せるために調整用変数
	double w2_16[NUM_A2][NUM_A3];
	double X16[NUM_A3];
	double b2_16[NUM_A2];
	double b316[NUM_A3];

	// 162のHWからの出力取得
	double z2_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];

	// 162のHWからの出力保管用
	double sum_z2_mean[NUM_A2], sum_z2_var[NUM_A2];

	sum_z2_mean[0] = 0.0; sum_z2_mean[1] = 0.0;
	sum_z2_var[0] = 0.0; sum_z2_var[1] = 0.0;
	z[0] = 0.0; z[1] = 0.0;


	for(l = 0; l < 16; l++){
		//z_lの計算
		b2_16[0] = b2_mean[l];
		b2_16[1] = b2_var[l];
		z2_tmp_hw[0] = 0.0; z2_tmp_hw[1] = 0.0;

		for(i = 0; i < 16; i++){
			k = 0;
			// 162に入れるための準備
			for(j = i*16; j < i*16+16; j++){
				w2_16[0][k] = w2_mean[l][j];
				w2_16[1][k] = w2_var[l][j];

				X16[k] = X[j];
				k++;
				//printf("%d ", j);
			}
			//printf("\n\r");

			// AEを使用
			AE_forward_929_HW(w2_16, X16, b2_16, z2_tmp_hw);

			b2_16[0] = z2_tmp_hw[0];
			b2_16[1] = z2_tmp_hw[1];

			//sum_z2_mean[0] = sum_z2_mean[0] + z2_mean_tmp_hw[0];
			//sum_z2_mean[1] = sum_z2_mean[1] + z2_mean_tmp_hw[1];
			//printf("%7.4f %7.4f\n\r", z2_mean_tmp_hw[0], z2_mean_tmp_hw[1]);
			//printf("%7.4f %7.4f\n\r", sum_z2_mean[0], sum_z2_mean[1]);
		}
		//printf("b2_16[0]: %f, b2_16[1]: %f\n\r", b2_16[0], b2_16[1]);

		for(i = 0; i < NUM_A2; i++){
			//z2_mean_tmp_hw[i] = sum_z2_mean[i] + b2_16[i];
			z2_tmp_hw[i] = z2_tmp_hw[i] / 16;
			//printf("b2_mean[%d]: %f, sum_z2_mean[%d]: %f\n\r", i, b2_mean[i], i, sum_z2_mean[i]);
			//printf("z2_hw[%d]: %f, ", i, z2_tmp_hw[i]);
		}
		//printf("\n\r");

		z2_mean_hw[l] = z2_tmp_hw[0];
		z2_var_hw[l] = z2_tmp_hw[1];
		a2_mean_hw[l] = z2_tmp_hw[0];
		a2_var_hw[l] = log(1 + exp(z2_tmp_hw[l]));
		z[l] = a2_mean_hw[l] + sqrt(a2_var_hw[l]) * eps[l];
		//printf("eps: %f", eps[l]);
		//printf("z: %f\n\r", z[l]);

	}






	//varの計算
	/*
	for(i = 0; i < 32; i++){
		k = 0;
		// 929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w2_var9[0][k] = w2_var[0][j];
			w2_var9[1][k] = w2_var[1][j];

			X9[k] = X[j];
			k++;
		}
		w2_var9[0][8] = 0.0; w2_var9[1][8] = 0.0;
		X9[8] = 1.0;
		b2_var9[0] = 0.0; b2_var9[1] = 0.0;

		// AEを使用
		AE_forward_929_HW(w2_var9, b2_var9, w3, b3, X9, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		//printf("%7.4f %7.4f %7.4f\n\r", z2_var_tmp_hw[0], z2_var_tmp_hw[1], sum);
		sum_z2_var[0] = sum_z2_var[0] + z2_var_tmp_hw[0];
		sum_z2_var[1] = sum_z2_var[1] + z2_var_tmp_hw[1];
		//printf("%7.4f %7.4f\n\r", sum_z2_var[0], sum_z2_var[1]);

	}

	for(i = 0; i < NUM_A2; i++){
		z2_var_tmp_hw[i] = sum_z2_var[i] + b2_var[i];
		z2_var_tmp_hw[i] = z2_var_tmp_hw[i] / 16;
		//printf("b2_var[%d]: %f, sum_z2_var[%d]: %f\n\r", i, b2_var[i], i, sum_z2_var[i]);
		printf("z2_var_hw[%d]: %f\n\r", i, z2_var_tmp_hw[i]);
	}

	//meanとvarを活性化関数に代入
	//zの演算
	for(i = 0; i < NUM_A2; i++){
		z2_mean_hw[i] = z2_mean_tmp_hw[i];
		a2_mean_hw[i] = z2_mean_tmp_hw[i];
		z2_var_hw[i] = z2_var_tmp_hw[i];
		a2_var_hw[i] = log(1 + exp(z2_var_tmp_hw[i]));

		z[i] = a2_mean_hw[i] + sqrt(a2_var_hw[i])*eps[i];
		printf("%f\n\r", z[i]);
	}
	*/

	/*
	// 画像の生成
	for(i = 0; i < 32; i++){
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w39[k][0] = w3[j][0];
			w39[k][1] = w3[j][1];
			b39[k] = b3[j];
			//printf("w39 %f %f\n\r", w39[k][0], w39[k][1]);
			//printf("b39 %f \n\r", b39[k]);
			k++;
		}
		w39[8][0] = 1.0; w39[8][1] = 1.0;
		b39[8] = 1.0;

		AE_forward_929_HW(w2_mean9, b2_mean9, w39, b39, X9, z, z2_mean_tmp_hw, a2_mean_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			z3_hw[j] = z3_tmp_hw[k]/16.0;
			a3_hw[j] = 1 / (1 + exp(-z3_hw[j]));
			//printf("%f ", z3_hw[j]);
			k++;
		}
		//printf("\n\r");
	}
	*/

	return;
}


void VAE_forward_16256_HW(
	double w2_mean[][NUM_X], double b2_mean[],
	double w2_var[][NUM_X], double b2_var[],
	double w3[][NUM_A3], double b3[],
	double X[],
	double z2_mean_hw[], double a2_mean_hw[],
	double z2_var_hw[], double a2_var_hw[],
	double z[],
	double z3_hw[], double a3_hw[], double eps[])
{
	int i, j, k, l;

	// 162のHWに乗せるために調整用変数
	double w2_16[NUM_A2][NUM_A3];
	double X16[NUM_A3];
	double b2_16[NUM_A2];
	double b316[NUM_A3];

	// 162のHWからの出力取得
	double z2_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];

	//zをX16に入力。X16は変更しなくて良いのでここで設定
	for(i = 0; i < NUM_A3; i++){
		X16[i] = z[i];
		//printf("X16: %f", X16[i]);
	}


	for(l = 0; l < 128; l++){
		//z3_lの計算
		z2_tmp_hw[0] = 0.0; z2_tmp_hw[1] = 0.0;
		k = 0;
		for(i = 0; i < 16; i++){
			// 162に入れるための準備
			w2_16[0][k] = w3[l*2][k];
			w2_16[1][k] = w3[l*2+1][k];
			b2_16[0] = b3[l*2];
			b2_16[1] = b3[l*2+1];
			//printf("\n\r");

			// AEを使用

			//sum_z2_mean[0] = sum_z2_mean[0] + z2_mean_tmp_hw[0];
			//sum_z2_mean[1] = sum_z2_mean[1] + z2_mean_tmp_hw[1];
			//printf("%7.4f %7.4f\n\r", w2_16[0][k], w2_16[1][k]);
			//printf("%7.4f %7.4f\n\r", b2_16[0], b2_16[1]);
			k++;
		}
		//printf("b2_16[0]: %f, b2_16[1]: %f\n\r", b2_16[0], b2_16[1]);

		AE_forward_929_HW(w2_16, X16, b2_16, z2_tmp_hw);

		for(i = 0; i < NUM_A2; i++){
			//z2_mean_tmp_hw[i] = sum_z2_mean[i] + b2_16[i];
			z2_tmp_hw[i] = z2_tmp_hw[i] / 16;
			//printf("b2_mean[%d]: %f, sum_z2_mean[%d]: %f\n\r", i, b2_mean[i], i, sum_z2_mean[i]);
			//printf("z2_hw[%d]: %f, ", i, z2_tmp_hw[i]);
		}
		//printf("\n\r");
		z3_hw[l*2] = z2_tmp_hw[0];
		z3_hw[l*2+1] = z2_tmp_hw[1];
		a3_hw[l*2] = 1.0 / (1 + exp(-z2_tmp_hw[0]));
		a3_hw[l*2+1] = 1.0 / (1 + exp(-z2_tmp_hw[1]));

		//printf("eps: %f", eps[l]);
		//printf("z3: %f %f\n\r", z3_hw[l*2], z3_hw[l*2+1]);

	}






	//varの計算
	/*
	for(i = 0; i < 32; i++){
		k = 0;
		// 929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w2_var9[0][k] = w2_var[0][j];
			w2_var9[1][k] = w2_var[1][j];

			X9[k] = X[j];
			k++;
		}
		w2_var9[0][8] = 0.0; w2_var9[1][8] = 0.0;
		X9[8] = 1.0;
		b2_var9[0] = 0.0; b2_var9[1] = 0.0;

		// AEを使用
		AE_forward_929_HW(w2_var9, b2_var9, w3, b3, X9, z, z2_var_tmp_hw, a2_var_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		//printf("%7.4f %7.4f %7.4f\n\r", z2_var_tmp_hw[0], z2_var_tmp_hw[1], sum);
		sum_z2_var[0] = sum_z2_var[0] + z2_var_tmp_hw[0];
		sum_z2_var[1] = sum_z2_var[1] + z2_var_tmp_hw[1];
		//printf("%7.4f %7.4f\n\r", sum_z2_var[0], sum_z2_var[1]);

	}

	for(i = 0; i < NUM_A2; i++){
		z2_var_tmp_hw[i] = sum_z2_var[i] + b2_var[i];
		z2_var_tmp_hw[i] = z2_var_tmp_hw[i] / 16;
		//printf("b2_var[%d]: %f, sum_z2_var[%d]: %f\n\r", i, b2_var[i], i, sum_z2_var[i]);
		printf("z2_var_hw[%d]: %f\n\r", i, z2_var_tmp_hw[i]);
	}

	//meanとvarを活性化関数に代入
	//zの演算
	for(i = 0; i < NUM_A2; i++){
		z2_mean_hw[i] = z2_mean_tmp_hw[i];
		a2_mean_hw[i] = z2_mean_tmp_hw[i];
		z2_var_hw[i] = z2_var_tmp_hw[i];
		a2_var_hw[i] = log(1 + exp(z2_var_tmp_hw[i]));

		z[i] = a2_mean_hw[i] + sqrt(a2_var_hw[i])*eps[i];
		printf("%f\n\r", z[i]);
	}
	*/

	/*
	// 画像の生成
	for(i = 0; i < 32; i++){
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			w39[k][0] = w3[j][0];
			w39[k][1] = w3[j][1];
			b39[k] = b3[j];
			//printf("w39 %f %f\n\r", w39[k][0], w39[k][1]);
			//printf("b39 %f \n\r", b39[k]);
			k++;
		}
		w39[8][0] = 1.0; w39[8][1] = 1.0;
		b39[8] = 1.0;

		AE_forward_929_HW(w2_mean9, b2_mean9, w39, b39, X9, z, z2_mean_tmp_hw, a2_mean_tmp_hw, z3_tmp_hw, a3_tmp_hw);
		k = 0;
		//929に入れるための準備
		for(j = i*8; j < i*8+8; j++){
			z3_hw[j] = z3_tmp_hw[k]/16.0;
			a3_hw[j] = 1 / (1 + exp(-z3_hw[j]));
			//printf("%f ", z3_hw[j]);
			k++;
		}
		//printf("\n\r");
	}
	*/

	return;
}

double PSNR_cal(double X[NUM_IN], double a3[NUM_IN]){
	double mse, psnr, sum;
	int i;
	sum = 0.0;
	for(i = 0; i < NUM_IN; i++){
		sum = sum + (fabs(X[i] - a3[i])) * (fabs(X[i] - a3[i]));
	}

	mse = sum / NUM_IN;
	//printf("sum: %f\n\r", fabs(X[0] - a3[0]));
	printf("mse: %f\n\r", mse);

	psnr = 10 * log10(1*1/mse);
	printf("psnr: %f\n\r", psnr);
	return psnr;
}



int main()
{
	XGpio input, output;
	int button_data = 0;
	int switch_data = 0;

	int i, j, l, m, loop;

	//SDカード読み取り用変数用意
	FIL fil;
	FATFS fatfs;
	char Filename[64];
	char buffer[100000];

	FRESULT Res;
	UINT NumBytesRead;
	u32 FileSize = 9*1024;
	TCHAR *Path = "0:/";
	unsigned char buff[4096];
	int count = 0;


	// SDカードの読み取り開始
	xil_printf("SD Card : Read test.\n\r");
	Res = f_mount(&fatfs, Path, 0);
	if(Res != FR_OK){
		xil_printf("ERROR: f_mount\n");
		return XST_FAILURE;
	}

	// w2_var.csvを読み込む
	double w2_var[NUM_A3][NUM_X];
	// w3.csvを読み込む
	double w3[NUM_IN][NUM_A3];
	// w2_mean.csvを読み込む
	double w2_mean[NUM_A3][NUM_X];
	// b2_meanを読み込む
   	double b2_mean[NUM_A3];
	//b2_varを読み込む
   	double b2_var[NUM_A3];
	// b3を読み込む
    double b3[NUM_IN];

	double k[NUM_IN][BATCHSIZE];

	// eps.csvを読み込む
	double eps[NUM_A3];
	double z[NUM_A3];

	double X[NUM_X];
	//double z2_mean[NUM_A2][BATCHSIZE], z2_var[NUM_A2][BATCHSIZE], z3[NUM_X][BATCHSIZE];
	double z2_mean_tmp[NUM_A3], z2_var_tmp[NUM_A3], z3_tmp[NUM_X];
	//double z_sw[NUM_A2][BATCHSIZE];
	//double a2_mean[NUM_A2][BATCHSIZE], a2_var[NUM_A2][BATCHSIZE], a3[NUM_X][BATCHSIZE];
	double a2_mean_tmp[NUM_A3], a2_var_tmp[NUM_A3], a3_tmp[NUM_X];

	/*
	double z_hw[NUM_A2];
	double z2_mean_hw[NUM_A2], z2_var_hw[NUM_A2], z3_hw[NUM_A3];
	double z2_mean_tmp_hw[NUM_A2], z2_var_tmp_hw[NUM_A2], z3_tmp_hw[NUM_A3];
	double a2_mean_hw[NUM_A2], a2_var_hw[NUM_A2], a3_hw[NUM_A3];
	double a2_mean_tmp_hw[NUM_A2], a2_var_tmp_hw[NUM_A2], a3_tmp_hw[NUM_A3];
	*/
	/*
	double dCda3[NUM_A3][BATCHSIZE], dCdz3[NUM_A3][BATCHSIZE], delta3[NUM_A3][BATCHSIZE];
	double dCdb3[NUM_A3], dCdw3[NUM_A3][NUM_A2];
	double dCda2_mean[NUM_A2][BATCHSIZE], dCdz2_mean[NUM_A2][BATCHSIZE], delta2_mean[NUM_A2][BATCHSIZE];
	double dCda2_var[NUM_A2][BATCHSIZE], dCdz2_var[NUM_A2][BATCHSIZE], delta2_var[NUM_A2][BATCHSIZE];
	double dCdb2_mean[NUM_A2], dCdw2_mean[NUM_A2][NUM_X];
	double dCdb2_var[NUM_A2], dCdw2_var[NUM_A2][NUM_X];
	double mult_tmp;
	*/

	// 画像データ読み込み int型 16*16=256個
	unsigned char image_data[256];
	unsigned char image_data_g[512*512];



	XGpio_Initialize(&input, XPAR_AXI_GPIO_0_DEVICE_ID);		//initialize input XGpio variable
	XGpio_Initialize(&output, XPAR_AXI_GPIO_1_DEVICE_ID);	//initialize output XGpio variable

	XGpio_SetDataDirection(&input, 1, 0xF);			//set first channel tristate buffer to input
	XGpio_SetDataDirection(&input, 2, 0xF);			//set second channel tristate buffer to input

	XGpio_SetDataDirection(&output, 1, 0x0);			//set first channel tristate buffer to output

	init_platform();



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
			printf("Reading Datas start.\n\r");

			//w2_mean.csv
			strcpy(Filename, "w2_mean.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_A3){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("ERROR: f_gets\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				for(j = 0; j < NUM_IN && token != NULL; j++){
					w2_mean[i][j] = atof(token);
					token = strtok(NULL, ",");
				}
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_A3; i++){
				for(j = 0; j < NUM_IN; j++){
					printf("%7.4f ", w2_mean[i][j]);
				}
				printf("\n\r");
			}
			*/

			printf("w2_mean is OK\n\r");

			//w2_var.csv
			strcpy(Filename, "w2_var16.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_A3){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("ERROR: f_gets\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				for(j = 0; j < NUM_IN && token != NULL; j++){
					w2_var[i][j] = atof(token);
					token = strtok(NULL, ",");
				}
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_A3; i++){
				for(j = 0; j < NUM_IN; j++){
					printf("%7.4f ", w2_mean[i][j]);
				}
				printf("\n\r");
			}
			*/

			printf("w2_var is OK\n\r");


			//b2_mean.csv
			strcpy(Filename, "b2_mean.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_A3){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("error\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				b2_mean[i] = atof(token);
				token = strtok(NULL, ",");
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_A3; i++){
				printf("b2_mean[%d] = %7.4f\n\r", i, b2_mean[i]);
			}
			*/

			printf("b2_mean is OK\n\r");

			//b2_var.csv
			strcpy(Filename, "b2_var16.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_A3){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("error\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				b2_var[i] = atof(token);
				token = strtok(NULL, ",");
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_A3; i++){
				printf("b2_var[%d] = %7.4f\n\r", i, b2_var[i]);
			}
			*/

			printf("b2_var is OK\n\r");


			//w3.csv
			strcpy(Filename, "w316.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_IN){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("error\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				for(j = 0; j < NUM_A3 && token != NULL; j++){
					w3[i][j] = atof(token);
					token = strtok(NULL, ",");
				}
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_IN; i++){
				for(j = 0; j < NUM_A3; j++){
					printf("%7.4f ", w3[i][j]);
				}
				printf("\n\r");
			}
			*/

			printf("w3 is OK.\n\r");

			//b3.csv
			strcpy(Filename, "b316.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_IN){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("error\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				b3[i] = atof(token);
				token = strtok(NULL, ",");
				i++;
			}
			f_close(&fil);

			/*
			for(i = 0; i < NUM_IN; i++){
				printf("b3[%d] = %7.4f\n\r", i, b3[i]);
			}
			*/

			printf("b3 is OK\n\r");

			//eps.csv
			strcpy(Filename, "eps.csv");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			i = 0;
			while(i < NUM_A3){
				TCHAR* line = f_gets(buffer, sizeof(buffer), &fil);
				if(line == NULL){
					xil_printf("error\n\r");
					break;
				}

				char *token = strtok(buffer, ",");
				eps[i] = atof(token);
				token = strtok(NULL, ",");
				i++;
			}
			f_close(&fil);


			for(i = 0; i < NUM_A3; i++){
				printf("eps[%d] = %7.4f\n\r", i, eps[i]);
			}

			printf("eps is OK\n\r");



			//画像データ取得
			strcpy(Filename, "test401.raw");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			Res = f_read(&fil,image_data, FileSize, &NumBytesRead);
			if (Res) {
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			f_close(&fil);

			/*
			for(i = 0; i < 256; i++){
				printf("%d ", image_data[i]);
				if((i+1) % 16 == 0){
					printf("\n\r");
				}
			}
			*/

			printf("\n\r");

			double image_data_i[256];

			for(i = 0; i < 256; i++){
				image_data_i[i] = (double)image_data[i] / 255.0;
				k[i][0] = image_data_i[i];
				printf("%5.3f ", k[i][0]);
				if((i+1) % 16 == 0){
					printf("\n\r");
				}
			}

			printf("image data is OK.\n\r");


			//画像データ取得(512×512)
			strcpy(Filename, "image4_g.raw");
			Res = f_open(&fil, Filename, FA_READ);
			if(Res){
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			Res = f_read(&fil,image_data_g, FileSize, &NumBytesRead);
			if (Res) {
				xil_printf("ERROR: f_open\n");
				return XST_FAILURE;
			}

			f_close(&fil);

			/*
			for(i = 0; i < 256; i++){
				printf("%d ", image_data[i]);
				if((i+1) % 16 == 0){
					printf("\n\r");
				}
			}
			*/

			printf("\n\r");

			double image_g[512*512];
			int count = 0;
			for(i = 0; i < 512*512; i++){
				image_g[i] = (double)image_data_g[i] / 1.0;
				/*
				if(image_g[i] == NULL){
					printf("buffer over\n\r");
				}
				*/
				printf("%3.1f ", image_g[i]);						
				if((i+1) % 512 == 0){
					printf("--%d--\n\r", count);
					count++;
				}
			}

			printf("image data is OK.\n\r");


			printf("finished read data\n\r");

			disp_menu();
		}else if(button_data == 0b0010){
			printf("button 1 pressed\n\r");
			printf("256*2*256 VAE use.\n\r");

			for(i = 0; i < 256; i++){
				X[i] = k[i][0];
			}

			VAE_forward_25616_HW(w2_mean, b2_mean, w2_var, b2_var,
						w3, b3, X,
						z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
						z, z3_tmp, a3_tmp, eps);

			//print_a2_a3(a2_mean_tmp, a2_var_tmp, a3_tmp_hw);

			printf("z3 hw:\n\r");
			for(i = 0; i < NUM_X; i++){
				printf("%f ", z3_tmp[i]);
				if((i+1)%8 == 0){
					printf("\n\r");
				}
			}

			printf("a3 hw: \n\r");
			for(i = 0; i < NUM_X; i++){
				printf("%f ", a3_tmp[i]);
				if((i+1)%8 == 0){
					printf("\n\r");
				}
			}

			printf("finish VAE used\n\r");
			disp_menu();
      	}else if(button_data == 0b0100) {
			// VAE929利用
			xil_printf("button 2 pressed\n\r");
			xil_printf("256 16 256 test\n\r");

			printf("256*16*256 VAE use.\n\r");

			for(i = 0; i < 256; i++){
				X[i] = k[i][0];
			}

			//エンコーダの計算
			VAE_forward_25616_HW(w2_mean, b2_mean, w2_var, b2_var,
						w3, b3, X,
						z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
						z, z3_tmp, a3_tmp, eps);
			for(i = 0; i < NUM_A3; i++){
				printf("%f\n\r", z[i]);
			}


			printf("finish VAE used\n\r");

         	disp_menu();
      	}else if(button_data == 0b1000){
          	printf("button 3 pressed\n\r");
			xil_printf("256 16 256 test\n\r");

			printf("256*16*256 VAE use.\n\r");

			for(i = 0; i < 256; i++){
				X[i] = k[i][0];
			}

			//エンコーダの計算
			VAE_forward_16256_HW(w2_mean, b2_mean, w2_var, b2_var,
						w3, b3, X,
						z2_mean_tmp, a2_mean_tmp, z2_var_tmp, a2_var_tmp,
						z, z3_tmp, a3_tmp, eps);
			/*
			for(i = 0; i < NUM_IN; i++){
				printf("%7.4f ", z3_tmp[i]);
				if((i+1)%16 == 0){
					printf("\n\r");
				}
			}
			printf("\n\r");
			for(i = 0; i < NUM_IN; i++){
				printf("%f ", a3_tmp[i]);
				if((i+1)%16 == 0){
					printf("\n\r");
				}
			}
			*/
			double psnr_1;

			psnr_1 = PSNR_cal(X, a3_tmp);

			//printf("PSNR: %f\n\r", psnr_1);


			printf("finish VAE used\n\r");


          	disp_menu();

      }
      else
         xil_printf("multiple buttons pressed\n\r");

      usleep(200000);			//delay

   }
   cleanup_platform();
   return 0;
}
