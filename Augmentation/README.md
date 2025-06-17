# Finetune-Whisper-base

## Project Structure
- **`train.py`**: Fine-tunes openai/whisper-base on the Dutch adult speech corpera Common Voice with option for speech augmentations (speed perturbation and SpecAugment).
- **`eval.py`**: Evaluates the fine-tuned model on the Dutch child speech copora JASMIN-CGN.
- **`requirements.txt`**: Contains the required dependecies.

## Usage
1. **Requirements**  
   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
3. **Training**  
  Set the whisper model, output directory, path to the Common Voice folder, path to the preprocessed JASIM-CGN folder and the desired modifications. Then run the code on the cluster.
4. **Evaluation**  
   Set the model directory and the path to the test data folder. Then run the code on the cluster.
