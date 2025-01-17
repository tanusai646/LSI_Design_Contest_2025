/* project: sdcard_sys                           */
/* SDカードのテスト                              */
/* 　Press button 3 then read data from SD card. */
/*   File name : test.txt (as Default)           */

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

int main()
{
   XGpio input, output;
   int button_data = 0;
   int switch_data = 0;

   FIL fil;
   FATFS fatfs;
   char FileName[32] = "test.txt";

   FRESULT Res;
   UINT NumBytesRead;
   u32 FileSize = 9*1024;
   TCHAR *Path = "0:/";
   unsigned char buff[9300];
   int i;

   XGpio_Initialize(&input, XPAR_AXI_GPIO_0_DEVICE_ID);	//initialize input XGpio variable
   XGpio_Initialize(&output, XPAR_AXI_GPIO_1_DEVICE_ID);	//initialize output XGpio variable

   XGpio_SetDataDirection(&input, 1, 0xF);			//set first channel tristate buffer to input
   XGpio_SetDataDirection(&input, 2, 0xF);			//set second channel tristate buffer to input

   XGpio_SetDataDirection(&output, 1, 0x0);		//set first channel tristate buffer to output

   init_platform();

   xil_printf("Press button 3. SD Card : Read test.\n\r");

   while(1){
      switch_data = XGpio_DiscreteRead(&input, 2);	//get switch data

      XGpio_DiscreteWrite(&output, 1, switch_data);	//write switch data to the LEDs

      button_data = XGpio_DiscreteRead(&input, 1);	//get button data

      //print message dependent on whether one or more buttons are pressed
      if(button_data == 0b0000){} //do nothing

      else if(button_data == 0b0001)
         xil_printf("button 0 pressed\n\r");

      else if(button_data == 0b0010)
         xil_printf("button 1 pressed\n\r");

      else if(button_data == 0b0100)
         xil_printf("button 2 pressed\n\r");

      else if(button_data == 0b1000){
    	   // SC card test sequence
    	   xil_printf("SD Card : Read test.\n\r");

    	   // Mount SD card
    	   Res = f_mount(&fatfs, Path, 0);
    	   if (Res != FR_OK) {
    	       xil_printf("ERROR: f_mount\n");
    	       return XST_FAILURE;
    	   }

    	   Res = f_open(&fil, FileName, FA_READ);
    	   if (Res) {
    	       xil_printf("ERROR: f_open\n");
    	       return XST_FAILURE;
    	   }

    	   // Read data from SD card
    	   Res = f_read(&fil, buff, FileSize, &NumBytesRead);
    	   if (Res) {
    	       xil_printf("ERROR: f_read\n");
    	       return XST_FAILURE;
    	   }
    	   // Display the data
    	   for ( i=0; i<NumBytesRead; i++ ) {
    		   xil_printf("%02x ",buff[i]);
    	   }
    	   xil_printf("\n\r");
    	   for ( i=0; i<NumBytesRead; i++ ) {
    		   xil_printf("%c",buff[i]);
    	   }
    	   xil_printf("\n\r");
      }
      else
         xil_printf("multiple buttons pressed\n\r");

      usleep(200000);			//delay

   }
   cleanup_platform();
   return 0;
}
