# Generating Micro-Doppler Maps of Patient Activities with Generative Adversarial Networks (GAN)

## Overview

This project uses Generative Adversarial Networks (GANs) to generate synthetic micro-Doppler maps of patient activities. These maps can be used in Human Activity Recognition (HAR) tasks while preserving privacy, as no cameras or intrusive sensors are required. By leveraging GANs, the project reduces the cost and difficulty of collecting real-world data from hospital environments. To see more details about the project, you can open the .pdf file which I have add.

## Objectives

1. Generate realistic micro-Doppler maps of patient activities using GANs.
2. Reduce the need for expensive and restrictive data collection in hospital settings.
3. Investigate the effectiveness of GANs for human activity data generation and classification.

## Highlights

- **Privacy-Preserving**: Utilizes radar-based micro-Doppler signatures instead of video or other intrusive methods.
- **Cost-Effective**: Reduces dependency on real-world data collection by generating synthetic data.
- **GAN Architecture**: Implements a custom GAN with up-sampling and down-sampling techniques for high-quality data generation.

## Project Structure

```
GAN_microDoppler/
├── src/                # Python scripts or helper functions
├── notebooks/          # Jupyter notebooks
├── data/               # Dataset (original and processed)
├── outputs/            # Generated images, logs, and models
├── README.md           # Project description
├── requirements.txt    # Dependencies
└── LICENSE             # License file
```

## Dataset

### Source

The dataset contains micro-Doppler maps derived from radar signals in hospital and home environments. Each sample corresponds to one of 14 distinct patient activities, such as walking, sitting, or lying down.&#x20;

The detailed dataset and the primary reference of this project can be found here: "[https://sumo.ilabt.imec.be/parrad](https://sumo.ilabt.imec.be/parrad)". The data used in this project is from one of the sets on the website mentioned. Each set is approximately 10 GB in size and contains three data types for two different radar systems. To extract the necessary micro-Doppler data from the set, the "HDFView" application was used. The specifications of the radar devices utilized can be found in the primary reference.

### Preprocessing

- Micro-Doppler maps are generated from Range Doppler (RD) maps by summing over the range dimension.
- Each sample consists of 40 frames (approximately 3.7 seconds).
- Data is normalized and structured similarly to the MNIST dataset for compatibility with various GAN implementations.

### Post-Processing

- Generated micro-Doppler maps undergo denoising using FFT (Fast Fourier Transform).

## GAN Architecture

### Components

- **Generator**: Creates synthetic micro-Doppler maps using transposed convolutional layers and ELU activation.
  - **Layers**: Uses multiple transposed convolutional layers for up-sampling the input noise.
  - **Activation**: Employs ELU (Exponential Linear Unit) activation functions in all layers except the output, ensuring smooth gradients.
  - **Output**: Produces grayscale micro-Doppler maps normalized to the dataset's scale.

- **Discriminator**: Distinguishes between real and generated data using convolutional layers with batch normalization and dropout.
  - **Layers**: Incorporates standard convolutional layers for down-sampling, followed by batch normalization.
  - **Dropout**: Set at 0.4 to prevent overfitting and stabilize learning.
  - **Activation**: ELU activation for intermediate layers, with a sigmoid output for binary classification (real vs fake).

### Training Details

- **Optimizer**: Adam optimizer with beta values of (0.5, 0.999) to maintain stability during training.
- **Learning Rate**: Fine-tuned to 0.0001 to balance convergence speed and stability.
- **Batch Size**: Set to 4, optimized for the available GPU (GTX 1650).
- **Epochs**: Trained for 350 epochs to ensure the generator and discriminator achieve equilibrium.
- **Loss Function**: Binary Cross-Entropy Loss with Logits (BCEWithLogitsLoss), combining sigmoid activation and cross-entropy in one step.

### Hyperparameter Tuning

- **Dropout**: Experimented with values between 0.1 and 0.6; found 0.4 optimal for balancing generator and discriminator.
- **Batch Size**: Compared sizes from 4 to 32; smaller sizes (4) yielded more stable training.
- **Learning Rate**: Adjusted between 0.00001 and 0.001; 0.0001 provided the best balance between stability and learning rate.
- **Epoch Count**: Limited to 350 due to GPU constraints, ensuring sufficient training without overfitting.

### Challenges and Solutions

1. **Noisy Loss Function**: Optimized learning rate and dropout to stabilize training.
2. **Discriminator Dominance**: Adjusted dropout and batch size to prevent overfitting.
3. **System Limitations**: Reduced batch size and epochs to accommodate hardware constraints.

## Results

- Successfully generated micro-Doppler maps for 14 activities.
- GAN-generated maps closely resemble real data but still contain minor noise, which is reduced through post-processing.
- Denoising improved the clarity of synthetic maps, making them more visually comparable to real data.

### Example Outputs

Generated images include activities such as:

- Walking to a chair
- Sitting down on a bed
- Lying down

Plots and visualizations of the generated data are included in the notebook.

## Future Work

1. **Improve GAN Performance**: Experiment with advanced architectures like StyleGAN or cGAN.
2. **Hardware Upgrade**: Use GPUs with higher RAM for more extensive training.
3. **Activity-Specific Generation**: Adapt the GAN to generate specific activities on demand.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/GAN_microDoppler.git
   cd GAN_microDoppler
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and follow the steps to train the GAN and analyze results.

## Contributing

Contributions are welcome! Feel free to create pull requests or report issues.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or collaboration, please contact [your email/username].

