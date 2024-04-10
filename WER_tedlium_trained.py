
# This script uses a wave2vec2 model already fine-tuned on TEDLIUM dataset, so we just need to compute the WER without the need for our own fine-tunning. 

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer

# Fine-tunning the facebook speech recognition model

tedlium_eval = load_dataset("LIUM/tedlium", "release3", split="test")
model = Wav2Vec2ForCTC.from_pretrained("sanchit-gandhi/wav2vec2-large-tedlium").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("sanchit-gandhi/wav2vec2-large-tedlium")

def map_to_pred(batch):
    input_values = processor(batch["audio"]["array"], return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch

result = tedlium_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])
wer_result = wer(result["text"], result["transcription"])
print("WER:", wer_result)
