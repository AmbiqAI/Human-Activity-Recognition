# Human Activity Recognition

To run this example, you'll need an Apollo4 EVB. We have yet to test this model on an Apollo4P.

## Compiling and Running Model

From the HAR/gcc directory:

1. `make clean`
2. `make`
3. `make deploy` Ensure your board is connected via the JLINK USB port and
   turned on first.
4. `make view` will provide SWO output as the model is running, show 
   predicted activity, as well as the specifics of the model (dimensionality, data, ect).

Wiring MPU6050:

1. Connect the VCC pin to the 5V supply in the J17 source.
2. Connect the GND pin to GND on the J17 source.
3. Connect the SCL pin to pin 8 on the J23 source.
4. Connect the SDA pin to pin 9 on the J23 source.


While model is running:

1. Place your sensor on a flat surface until the SWO Viewer provides instruction for you to perform an activity.
2. When prompted to do so, perform one of the following five activities: Walking, Jogging, Climbing Stairs, Sitting, or Standing. Ensure your sensor is orientated correctly while this is being done.
3. After roughly 10 seconds, the SWO Viewer will supply the name of the activity it predicts you to have been performing.


See the [data/README.md](./data/README.md)
for more detailed insights into this model