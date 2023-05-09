# stat222-generative-ai
Repository for Group A code, working towards benchmarking generative AI models for image inpainting tasks.

## Repository Structure

- EDA_Preprocess
    - Scripts to perform basic EDA on the CelebA dataset, as well as the code to programatically create       the damage masks for the images from the Met with naturally occuring damage.

- VAE_AE
    - Code and experiment results for initial experiments to run both variational and convolutional           auto-encoders.

- FineTune
    - Code to fine-tune a latent diffusion model on images from the Egyptian collection at the Met. 

- GenerateFigures
    - Code to generate figures for both the VAE, AutoEncoder, and fine-tuned latent diffusion models.
