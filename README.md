# DrugAI-CVAE

**Generating Novel Drug Molecules with Autoregressive CVAE**  

DrugAI explores an alternative approach to drug discovery by generating novel molecular sequences (SMILES) using an Autoregressive Conditional Variational Autoencoder (CVAE) built in PyTorch. This project builds on the research presented in [Dony Ang, Hagop Atamian, and Cyril Rakovski's paper](https://www.mdpi.com/2655400) and investigates a different architectural strategy for targeted drug design.  

---

## Project Overview  
This project was conducted in collaboration with:  
- **Dylan McIntosh**  
- **Dr. Chelsea Parlett**  
- **Dr. Dony Ang**  
- **Dr. Hagop Atamian**  

DrugAI utilizes the BindingDB dataset to train on human-valid molecules, ensuring the generation of biologically relevant drug candidates. By conditioning on target protein spikes, the model aims to produce novel drug molecules with potential therapeutic value.  

---

## Key Features  
- **Alternative Approach to Prior Research:** Builds on and extends transformer based concepts from [Ang, Atamian, and Rakovski’s paper](https://www.mdpi.com/2655400).  
- **Autoregressive CVAE Architecture:** Utilizes a Conditional Variational Autoencoder that predicts the next token in a molecular sequence by conditioning on target protein spikes.  
- **SMILES Tokenization and Latent Space Sampling:** Embeds SMILES tokens and samples from the latent space for accurate and innovative molecular sequence generation.  

---

## How It Works  
This repository outlines the training, prediction, and model architecture used in DrugAI. It is not structured as an extensive package but instead provides the core files necessary to understand and reproduce the model's functionality.  

Key components include:  
- **Training Script:** Details the model training process using human-valid molecules from the BindingDB dataset.  
- **Prediction Functionality:** Demonstrates how to generate new drug molecules conditioned on target protein spikes.  

---

## Dataset  
DrugAI is trained on human-valid molecules sourced from the BindingDB dataset. This dataset is curated to ensure that generated sequences are biologically plausible and relevant for drug discovery.  

---

## Model Architecture  
DrugAI utilizes an Autoregressive Conditional Variational Autoencoder (CVAE), including:  
- **Encoder:** Captures molecular context from embedded SMILES tokens.  
- **Latent Space:** Samples from the latent distribution to introduce molecular diversity.  
- **Decoder:** Predicts the next token autoregressively, conditioned on the protein spike and previously generated tokens.  

This approach allows for target-specific drug generation and offers an alternative methodology to the techniques explored in Ang, Atamian, and Rakovski’s research.  