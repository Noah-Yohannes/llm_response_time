
# This script uses a wave2vec2 model already fine-tuned on TEDLIUM dataset, so we just need to compute the WER without the need for our own fine-tunning. 

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer, cer, wil, wip, mer     #word error rate, character error rate, word information lost, word information preserved and match error rate


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
cer_result = cer(result["text"], result["transcription"])
wil_result = wil(result["text"], result["transcription"])
wip_result = wip(result["text"], result["transcription"])
mer_result = mer(result["text"], result["transcription"])
print("WER:", wer_result)
print("CER:", cer_result)
print("WIL:", wil_result)
print("WIP:", wip_result)
print("MER:", mer_result)



