# Grape Leaf Disease Classification 

Deep learning project that classifies grapevine leaf images into four categories using a custom CNN baseline and five transfer learning models, with Grad-CAM explainability on the best hybrid model.



## Problem

Grapevine growers lose yield to fungal diseases that look similar to the untrained eye. The notebook trains image classifiers that separate three diseases from healthy foliage:

- Black Rot
- ESCA (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

## Dataset

Grape Disease Dataset (Original), downloaded from Kaggle inside the notebook.

- Source: https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original
- Licence: CC0-1.0
- Download size: about 151 MB
- All images are 256 x 256 RGB, resized to 224 x 224 for training

The archive ships with its own train and test folders:

| Split | Black Rot | ESCA | Healthy | Leaf Blight | Total |
|---|---|---|---|---|---|
| train | 1,888 | 1,920 | 1,692 | 1,722 | 7,222 |
| test | 472 | 480 | 423 | 430 | 1,805 |

Classes are close to balanced, so plain accuracy is a fair headline metric and macro F1 is reported alongside it.

Expected folder layout after extraction:

```
grape_data/
  Original Data/
    train/
      Black Rot/  ESCA/  Healthy/  Leaf Blight/
    test/
      Black Rot/  ESCA/  Healthy/  Leaf Blight/
```

## How the data is split

The 7,222 training images are split 70 / 30 by `ImageDataGenerator(validation_split=0.30)` with `seed=42`, giving 5,057 training and 2,165 validation images. Training images get augmentation (rotation 30 degrees, width and height shift 0.2, horizontal and vertical flip, zoom 0.2, brightness 0.8 to 1.2, nearest fill). Validation images are only rescaled to [0, 1].

The 1,805 image `test/` folder is never touched during training or model selection. It is reserved for the final held-out evaluation section at the end of the notebook.

## Models

All models share the same generators, 224 x 224 input, batch size 32 and a 15 epoch budget with early stopping.

| Model | Setup | Total params | Trainable |
|---|---|---|---|
| Baseline CNN | 4 conv blocks (32/64/128/256) with batch norm, dense 256 and 128, dropout 0.5 and 0.3, Adam 1e-4 with `clipnorm=1.0` | 13,269,060 | 13,268,100 |
| MobileNetV2 | ImageNet weights frozen, GAP, dense 256 with L2 1e-4, dropout 0.3 | 2,586,948 | 328,964 |
| VGG16 | ImageNet weights frozen, GAP, dense 256 and 128, dropout 0.5 and 0.3, default Adam | 14,879,428 | 164,740 |
| Hybrid CNN + VGG16 | Two branches merged by concatenation: a 3 block CNN trained from scratch plus frozen VGG16, then dense 256 and 128 | 15,006,340 | 291,204 |
| EfficientNetB0 | ImageNet weights frozen, GAP, dense 256 with L2 1e-4, dropout 0.3 | about 4.3 M | about 0.3 M |
| ResNet50 | ImageNet weights frozen with `resnet50.preprocess_input`, GAP, dense 256 with L2 1e-4, dropout 0.3 | about 23.9 M | about 0.5 M |

A detail worth keeping in mind while reading the code: the generators already rescale pixels to [0, 1], but MobileNetV2, EfficientNetB0 and ResNet50 expect raw [0, 255] input because they apply their own preprocessing internally. Each of those three models therefore starts with a `Rescaling(255.0)` layer that undoes the generator rescale. Removing that layer is what causes transfer learning models to collapse to near chance accuracy.

## Results on the validation split (2,165 images)

Recorded from the saved notebook run.

| Model | Validation accuracy | Notes |
|---|---|---|
| EfficientNetB0 | 99.5 percent | best run, 10 misclassified out of 2,165 |
| ResNet50 | 99.3 percent best val, 98.9 percent final | 15 misclassified |
| Hybrid CNN + VGG16 | 99 percent | best val loss 0.0192 at epoch 10, restored by early stopping |
| VGG16 | 93 percent | Black Rot and ESCA are the confusion pair |
| Baseline CNN | 83 percent | Black Rot recall only 0.39, unstable validation loss early on |
| MobileNetV2 | 75.5 percent | stopped at epoch 7, weakest of the pretrained backbones |

Across every model the errors sit almost entirely between Black Rot and ESCA. Healthy and Leaf Blight are separated near perfectly.

## Explainable AI

The Grad-CAM section takes the trained hybrid model, locates the VGG16 branch and its last convolutional layer (`block5_conv3`), rebuilds the classifier head as a differentiable path from that layer to the output, and overlays the resulting heatmaps on sample leaves. This shows whether the model is looking at the lesions themselves rather than background or leaf edges.

## Held-out test evaluation

The last three cells build a test generator from `grape_data/Original Data/test` with rescale only and no shuffle, assert that the class order matches the training generator, then evaluate all six trained models once and print accuracy, loss, macro F1, weighted F1 and misclassification counts, plus per class reports, a 2 x 3 confusion matrix grid, and a validation versus test accuracy comparison chart.

These cells have no saved output in the current notebook, so they still need to be run. They depend on every model variable from the earlier sections being alive in the same session, which means the notebook has to be run top to bottom in one go before this section will execute.

## How to run

Google Colab with a GPU runtime is the intended environment. The saved run used TensorFlow 2.20.0 on a Colab GPU, with roughly 75 to 100 seconds per epoch per model.

1. Open `Topic_04_Model V2.ipynb` in Colab and set Runtime, Change runtime type, GPU.
2. Run the first cell and upload your `kaggle.json` when prompted. You get this file from your Kaggle account under Settings, API, Create New Token.
3. Run the remaining cells in order. Do not skip the constants cell or the generator cell, since every later section depends on `IMG_SIZE`, `train_generator` and `val_generator`.
4. Expect around 90 minutes of total training time for all six models on a Colab GPU.

To run locally instead, install the packages in `requirements.txt`, replace the first cell with your own Kaggle credential setup (`~/.kaggle/kaggle.json` with permission 600), and skip the `google.colab` import.

```
pip install -r requirements.txt
```

A CUDA capable GPU with at least 8 GB of memory is strongly recommended. On CPU the full run takes many hours.

## Files in this folder

| File | What it is |
|---|---|
data download, exploration, six models, Grad-CAM, held-out test section |
| `requirements.txt` | Python package versions for a local run |
| `FPR_Draft_Topic ` | Latest final project report draft |
| `FPR_Draft_Chapters1-4.docx` | Earlier report draft, chapters 1 to 4 |
| `Grape_Leaf_Disease_Presentation.pptx` | Project presentation |
| `Project Form topic 01 updatd.docx` | Project registration form |
| `skeletal_ppt_template.pptx` | University presentation template |
| `nb_imgs/` | Exported figures from the notebook, with compressed copies in `nb_imgs/opt/` |
## Reproducibility notes

- `random_state=42` for pandas sampling in the exploration section and `seed=42` on the generators, so the split is stable across runs.
- TensorFlow layer initialisation and GPU kernel ordering are not seeded, so accuracy figures can move by a few tenths of a percent between runs.
- ImageNet weights for MobileNetV2, VGG16, EfficientNetB0 and ResNet50 download automatically on first use and need an internet connection.

