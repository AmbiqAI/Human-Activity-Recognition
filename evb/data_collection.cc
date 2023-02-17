// //#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include<stdio.h>
// #include <cstdarg>
// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include "button.h"
// #include "power.h"
// #include "mpu6000_i2c_driver.h"
// #include "am_mcu_apollo.h"
// #include "am_bsp.h"
// #include "am_hal_gpio.h"

// #define MIN(a,b) (((a)<(b))?(a):(b))
// #define MAX(a,b) (((a)>(b))?(a):(b))
// #define AXES 3
// void *g_pIOMHandle_touch;
// am_hal_iom_config_t     g_sIomCfg_touch =
// {
//     // Set up IOM
//     // Initialize the Device
//     .eInterfaceMode       = AM_HAL_IOM_I2C_MODE,
//     //.ui32ClockFreq        = AM_HAL_IOM_400KHZ,
//     .ui32ClockFreq        = AM_HAL_IOM_100KHZ,
//     .pNBTxnBuf = NULL,
//     .ui32NBTxnBufLength = 0
// };
// const char* activities[] = {"Walking",
//                             "Jogging",
//                             "Stairs",
//                             "Sitting",
//                             "Standing",};
// int main(void) {
//     am_bsp_itm_printf_enable();
//   am_bsp_debug_printf_enable();
//   //am_ai_set_power_mode(&am_ai_development_default);

//   am_hal_delay_us(1000000);

//   if (am_devices_mpu6000_init(1, &g_sIomCfg_touch, &g_pIOMHandle_touch )) {
//       am_util_stdio_printf("MPU initialization failed");
//       return 1;
//   }
//   am_hal_delay_us(1000000); // give init a chance to 'take'
//   uint8_t buffer[32];
//   int16_t accelX;
//   int16_t accelY;
//   int16_t accelZ;
//   int16_t gyroX;
//   int16_t gyroY;
//   int16_t gyroZ;
//   float accelVals[AXES];
//   float gyroVals[AXES];
//   int i = 4;
//   uint32_t status = 0;
//   uint32_t errors = 0;
//   uint32_t reads = 0;
//   int16_t temperature_regval;
//   float temperature;
//   am_util_stdio_printf("Beginning MPU Calibration!\n\n");
//   am_util_stdio_printf("Please place your sensor on a flat surface until calibration is finished\n");
//   if(mpu6000_calibration()) {
//     am_util_stdio_printf("Calibration Failed!\n");
//     return 1;
//   }
//   am_util_stdio_printf("Calibration Complete!\n\n");
//   am_util_stdio_printf("Get Ready To Perform One of the Five Actvities.\n\n");
//   am_hal_delay_us(4000000);
//   am_hal_delay_us(5000000);
//     am_util_stdio_printf("%s!\n", activities[i]);
//     int16_t iter = 0;
//     while (iter < 3600) {
//             // Read
//             status = read_sensors(buffer);  // fetch raw data from the registers
//             reads++;

//             if (status == AM_DEVICES_MPU6000_STATUS_ERROR) {
//             errors++;
//             } else {
//             // decode the buffer
//             accelX = buffer[0] << 8 | buffer[1];
//             accelY = buffer[2] << 8 | buffer[3];
//             accelZ = buffer[4] << 8 | buffer[5];

//             temperature_regval = (buffer[6]<<8 | buffer[7]);
//             temperature = temperature_regval/340.0+36.53;

//             gyroX = buffer[8] << 8 | buffer[9];
//             gyroY = buffer[10] << 8 | buffer[11];
//             gyroZ = buffer[12] << 8 | buffer[13];
//             }

//             #if 1
//             am_hal_delay_us(50000);
//             if (i == 1000) {
//             //mpu6000_finish_init();
//             am_util_stdio_printf("Errors/Reads: %d/%d\n",errors, reads);
//             i = 0;
//             }
//             #endif
//             #if 1
            

//             //Convert
//             accelGravity(accelVals, accelX, accelY, accelZ, ACCEL_FS_16G);
//             gyroDegPerSec(gyroVals, gyroX, gyroY, gyroZ, GYRO_FS_500DPS);

//             // am_util_stdio_printf(" Second %f\n", accelVals[0] );
//             // Debug
//             am_util_stdio_printf("7, %f,%f,%f,%f,%f,%f,%s\n", accelVals[0], accelVals[1],accelVals[2], gyroVals[0], gyroVals[1], gyroVals[2], activities[i]);
//             iter += 1;
//             #endif
//         }

//   return 0;
// }