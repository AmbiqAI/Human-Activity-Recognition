//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <math.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "am_util_stdio.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "button.h"
#include "power.h"
// #include "test_data.h"
//#include "timers.h"
#include "scott_quantized.h"
#include "static_model.h"
#include "group_model.h"
#include "dynamic_model.h"
#include "quantized_har_model.h"
#include "mpu6050_i2c_driver.h"
#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_hal_gpio.h"

#define ACCEL_SAMPLE_SIZE 1200
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define AXES 3
void *g_pIOMHandle_touch;
float sensor_data[ACCEL_SAMPLE_SIZE];
float accelXMean = 0.0;
float accelYMean = 0.0;
float accelZMean = 0.0;
float gyroXMean = 0.0;
float gyroYMean = 0.0;
float gyroZMean = 0.0;
float accelXStd = 0.0;
float accelYStd = 0.0;
float accelZStd = 0.0;
float gyroXStd = 0.0;
float gyroYStd = 0.0;
float gyroZStd = 0.0;
am_hal_iom_config_t     g_sIomCfg_touch =
{
    // Set up IOM
    // Initialize the Device
    .eInterfaceMode       = AM_HAL_IOM_I2C_MODE,
    //.ui32ClockFreq        = AM_HAL_IOM_400KHZ,
    .ui32ClockFreq        = AM_HAL_IOM_100KHZ,
    .pNBTxnBuf = NULL,
    .ui32NBTxnBufLength = 0
};

int collect_data(void) {
//Collect 200 gyrometer samples ~ 10 secs worth of activity data
  am_bsp_itm_printf_enable();
  am_bsp_debug_printf_enable();
  //am_ai_set_power_mode(&am_ai_development_default);

  am_hal_delay_us(1000000);

  if (am_devices_mpu6050_init(1, &g_sIomCfg_touch, &g_pIOMHandle_touch )) {
      am_util_stdio_printf("MPU initialization failed");
      return 1;
  }
  am_hal_delay_us(1000000); // give init a chance to 'take'
  float gyroSum = 0.0;
  uint8_t buffer[32];
  int16_t accelX;
  int16_t accelY;
  int16_t accelZ;
  int16_t gyroX;
  int16_t gyroY;
  int16_t gyroZ;
  float accelVals[AXES];
  float gyroVals[AXES];
  int i = 0;
  uint32_t status = 0;
  uint32_t errors = 0;
  uint32_t reads = 0;
  int16_t temperature_regval;
  float temperature;
  int16_t iter = 0;
  am_util_stdio_printf("Beginning MPU Calibration!\n\n");
  am_util_stdio_printf("Please place your sensor on a flat surface until calibration is finished\n");
  if(mpu6050_calibration()) {
    am_util_stdio_printf("Calibration Failed!\n");
    return 1;
  }
  am_util_stdio_printf("Calibration Complete!\n\n");
  am_util_stdio_printf("Get Ready To Perform One of the Five Actvities.\n\n");
  am_hal_delay_us(4000000);
  while (iter < 200) {
        // Read
        status = read_sensors(buffer);  // fetch raw data from the registers
        reads++;
        i++;

        if (status == AM_DEVICES_mpu6050_STATUS_ERROR) {
          errors++;
        } else {
          // decode the buffer
          accelX = buffer[0] << 8 | buffer[1];
          accelY = buffer[2] << 8 | buffer[3];
          accelZ = buffer[4] << 8 | buffer[5];

          temperature_regval = (buffer[6]<<8 | buffer[7]);
          temperature = temperature_regval/340.0+36.53;

          gyroX = buffer[8] << 8 | buffer[9];
          gyroY = buffer[10] << 8 | buffer[11];
          gyroZ = buffer[12] << 8 | buffer[13];
        }

        #if 1
        am_hal_delay_us(50000);
        if (i == 1000) {
          //mpu6000_finish_init();
          am_util_stdio_printf("Errors/Reads: %d/%d\n",errors, reads);
          i = 0;
        }
        #endif
        #if 1
        

        //Convert
        accelGravity(accelVals, accelX, accelY, accelZ, ACCEL_FS_16G);
        gyroDegPerSec(gyroVals, gyroX, gyroY, gyroZ, GYRO_FS_500DPS);

        // am_util_stdio_printf(" Second %f\n", accelVals[0] );
        // Debug
        am_util_stdio_printf("Raw Accel Data: [%f %f %f]  \t", accelVals[0], accelVals[1],accelVals[2]);
        am_util_stdio_printf("Raw Gyro Data: [%f %f %f] \t", gyroVals[0], gyroVals[1], gyroVals[2]);
        am_util_stdio_printf("Temperature: [%f]\n", temperature);
        sensor_data[iter * 6] = accelVals[0];
        sensor_data[(iter* 6) + 1] = accelVals[1];
        sensor_data[(iter * 6) + 2] = accelVals[2];
        sensor_data[(iter * 6) + 3] = gyroVals[0];
        sensor_data[(iter * 6) + 4] = gyroVals[1];
        sensor_data[(iter * 6)+ 5] = gyroVals[2];
        accelXMean = ((float)(iter) * accelXMean + accelVals[0])/(float)(iter + 1);
        accelYMean = ((float)(iter) * accelYMean + accelVals[1])/(float)(iter + 1);
        accelZMean = ((float)(iter) * accelZMean + accelVals[2])/(float)(iter + 1);
        gyroXMean = ((float)(iter) * gyroXMean + gyroVals[0])/(float)(iter + 1);
        gyroYMean = ((float)(iter) * gyroYMean + gyroVals[1])/(float)(iter + 1);
        gyroZMean = ((float)(iter) * gyroZMean + gyroVals[2])/(float)(iter + 1);
        iter += 1;
        #endif
    }
  am_util_stdio_printf("\n\n%f", gyroSum/600.0);
  return 0;
}
void calculateStd(int col, float mean, float* std) {
    am_util_stdio_printf("Mean for Col %d: %f",col, mean);
    int index = col;
    float variance = 0.0;
    while(index < ACCEL_SAMPLE_SIZE) {
        variance += pow(sensor_data[index] - mean, 2);
        index += 6;
    }
    *std = sqrt(variance/200.0);
    am_util_stdio_printf("Standard Deviation for %d: %f",col, *std);
}
const char* activities[] = {"Walking",
                            "Jogging",
                            "Stairs",
                            "Sitting",
                            "Standing",};
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 1024*100;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

//initialize HAR model
void har_init(void)
{

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    tflite::InitializeTarget();

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(quantized_har_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      TF_LITE_REPORT_ERROR(error_reporter,
                            "Model provided is schema version %d not equal "
                            "to supported version %d.",
                            model->version(), TFLITE_SCHEMA_VERSION);
      return;
    }

//     // This pulls in all the operation implementations we need.

    //static tflite::MicroMutableOpResolver<1> resolver;
    static tflite::AllOpsResolver resolver;
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }

    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
  
    am_util_stdio_printf("input->dims->size %d\n",input->dims->size);
    am_util_stdio_printf("input->dims->data[0] %d\n", input->dims->data[0]);
    am_util_stdio_printf("input->dims->data[1] %d\n", input->dims->data[1]);
    am_util_stdio_printf("input->dims->data[2] %d\n", input->dims->data[2]);
    am_util_stdio_printf("type input->type %d\n", input->type);

    am_util_stdio_printf("\n");

    am_util_stdio_printf("output->dims->size %d\n", output->dims->size);
    am_util_stdio_printf("output->dims->data[0] %d\n", output->dims->data[0]);
    am_util_stdio_printf("debug: output->dims->data[1] %d\n", output->dims->data[1]);
    am_util_stdio_printf("output->type %d\n", output->type);

}

int main(void)
{
    am_bsp_itm_printf_enable();
    float tmp;
    // Configure power - different use modes
    // require different power configs

    #ifdef AUDIODEBUG
      // This mode uses RTT, which needs SRAM
      am_bsp_debug_printf_enable();
      am_ai_set_power_mode(&am_ai_development_default);
    #else
      #if ENERGY_MODE==1
        am_bsp_uart_printf_enable();
        am_ai_set_power_mode(&am_ai_audio_default);
     #else 
        am_bsp_debug_printf_enable(); // Leave crypto on for ease of debugging
        am_ai_set_power_mode(&am_ai_development_default);
      #endif
    #endif

    // Initialized everything else
    har_init();
    collect_data();
    button_init();

//     // This is only for measuring power using an external power monitor such as
//     // Joulescope - it sets GPIO pins so the state can be observed externally
//     // to help line up the waveforms. It has nothing to do with AI...
    #if ENERGY_MODE==1
      am_init_power_monitor_state(); 
    #endif
      calculateStd(0, accelXMean, &accelXStd);
      calculateStd(1, accelYMean, &accelYStd);
      calculateStd(2, accelZMean, &accelZStd);
      calculateStd(3, gyroXMean, &gyroXStd);
      calculateStd(4, gyroYMean, &gyroYStd);
      calculateStd(5, gyroZMean, &gyroZStd);
    float means[] = {
        accelXMean,
        accelYMean,
        accelZMean,
        gyroXMean,
        gyroYMean,
        gyroZMean
    };
    float stdevs[] = {
        accelXStd,
        accelYStd,
        accelZStd,
        gyroXStd,
        gyroYStd,
        gyroZStd
    };
    //Set input tensor
    for (uint16_t i = 0; i < ACCEL_SAMPLE_SIZE; i = i + 1) {
            //Normalize input
            tmp = (sensor_data[i] - means[i % 6])/stdevs[i%6];
            tmp = tmp/input->params.scale + input->params.zero_point;
            // tmp = accel_test_data[i] / input->params.scale + input->params.zero_point
            tmp = MAX(MIN(tmp, 127), -128);
            input->data.int8[i] = (int8_t) tmp;
        }

    //Invoke() runs the model
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      while(1) {};
    }
    
    //get output from output tensor and figure out which label it assigned the highest value to
    float acitivity_inf[5];

    uint8_t activity_max; 
    float max_val = 0.0;
    am_util_stdio_printf("\nPrediction:\n\n");
    for (uint8_t i = 0; i < 5; i = i + 1) {
      acitivity_inf[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
      am_util_stdio_printf("Activity[%i]: %f %s\n", i, acitivity_inf[i], activities[i]);
      if (acitivity_inf[i] > max_val) { 
          max_val = acitivity_inf[i]; activity_max = i; 
      }
    }

    am_util_stdio_printf("**Activity Prediction: %s", activities[activity_max]);
    
}