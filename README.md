# Human Activity Recognition

This repository contains an example implementation of an model for classifying human activity, including scripts for processing datasets, training the model, and exporting a deployable version of it. It also includes a sample neuralSPOT-based application suitable for deploying to an Apollo4P EVB.

It is offered only as an example of how to accomplish this type of model - **this is not production-ready**.

## Caveats

We use WISDM, a publicly available dataset, but only use 5 of the many classes of activity the dataset includes. There are three caveats here:

1. The sample size for these activities is small, which results in a less-accurate model. We've augmented the dataset, but are starting to run into over-fitting.
2. Some of the activities are physically similar (sitting vs. standing, walking vs. jogging) make identification overlap between those activities.
3. The WISDM dataset collected data for wrist devices, but the sensor was different, and oriented differently. We've attempted to mitigate this by collecting data using our sensor and using a fine-tuning step in the training process.

## Training the Model

We include a pre-trained model, but if you wish to replicate this, the process is straightforward. The most complex step is installing the needed non-python tools, which will vary for each development platform. Once those tools are installed, the steps to training a model are:

```bash
$> cd python
$> pip install requirements.txt
$> python -m train
```

This will execute the following steps:

1. Load the dataset files and partition the data into training, testing, and validation sets
2. Process them into 10 second moving windows
3. Augment the training data by adding jitter, scale noise, and warping
4. Train the main model - this will take a while, depending on your compute resources (tested on both CPU and GPU platforms)
5. Fine-tune the model using MPU6050 data
6. Export all artifacts (processed datasets, trained models, quantized models, and C version of the quantized model)

See the [data/README.md](./data/README.md)for more detailed insights into this model

The train.py script supports the following options:

```bash
optional arguments:
  --seed SEED           Random Seed (default: 42)
  --num-time-steps NUM_TIME_STEPS
                        Number of Timestep Windows (default: 200)
  --num-features NUM_FEATURES
                        Number of Features (default: 6)
  --sample-step SAMPLE_STEP
                        Timestep Window Slide (default: 20)
  --trained-model-dir TRAINED_MODEL_DIR
                        Directory where trained models are stored (default: trained_models/)
  --job-dir JOB_DIR     Directory where artifacts are stored (default: artifacts/)
  --dataset-dir DATASET_DIR
                        Directory where datasets reside (default: datasets/)
  --processed-dataset PROCESSED_DATASET
                        Name of processed baseline dataset (default: processed_dataset.pkl)
  --augmented-dataset AUGMENTED_DATASET
                        Name of processed augmented dataset (default: augmented_dataset.pkl)
  --processed-ft-dataset PROCESSED_FT_DATASET
                        Name of processed baseline dataset (default: processed_ft_dataset.pkl)
  --augmented-ft-dataset AUGMENTED_FT_DATASET
                        Name of processed augmented dataset (default: augmented_ft_dataset.pkl)
  --batch-size BATCH_SIZE
                        Batch Size (default: 32)
  --augmentations AUGMENTATIONS
                        Number of augmentation passes (default: 8)
  --no-save-processed-dataset
                        Save processed datasets as pkls (default: True)
  --epochs EPOCHS       Number of training epochs (default: 35)
  --ft-epochs FT_EPOCHS
                        Number of fine-tuning epochs (default: 3)
  --model-name MODEL_NAME
                        Name of trained model (default: model)
  --training-dataset-percent TRAINING_DATASET_PERCENT
                        Percent of records used for training (default: 65)
  --no-show-training-plot
                        Show training statistics plots (default: True)
  --no-train-model      Train the model, otherwise load existing model (default: True)
  --no-fine-tune-model  Fine-tune the model, otherwise load existing model (default: True)

help:
  -h, --help            show this help message and exit
```

## Deploying the Model to an EVB

To run this example, you'll need an Apollo4 EVB and an MPU6050 connected following [these instructions](https://github.com/AmbiqAI/neuralSPOT/tree/main/neuralspot/ns-i2c).

### Compiling and Running Model

This repository includes a small application implementing HAR on an Ambiq EVB. To compile and deploy the application, follow these steps:

```bash
$> cd evb
$> make
$> make deploy # flashes the model to the EVB using Jlink
$> make view # connects to Jlink's ITM monitor to enable viewing the application's output
```


