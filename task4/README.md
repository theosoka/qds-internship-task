# Soil Erosion Detection
This code is created for training a model for detecting soil erosion in satellite images. The model is trained using Tensorflow Keras.

# Requirements
- rasterio
- geopandas
- shapely
- numpy
- opencv-python
- matplotlib
- tensorflow

The whole list is in the requirements.txt file.

# Getting Started
1. Clone this repository
2. Install the required libraries
3. Find a satellite image you want to train the model on and place it in the project directory
4. Download masks for the erosion areas and place it in the project directory
5. Run the script

# Additional information
- The U-Net architecture is used.
- The model is trained using binary cross-entropy loss and the Adam optimizer.
- The image and mask tiles used for training are normalized between 0 and 1.
- The trained model can be used to predict erosion areas in new satellite images.
