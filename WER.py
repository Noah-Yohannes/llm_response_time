
import torch
import datasets
import jiwer
import evaluate
import librosa
import re


from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, DownloadConfig, DownloadMode, load_dataset
from evaluate import load

tedlium = load_dataset("LIUM/tedlium", "release3", split="test")

tedlium = tedlium.remove_columns([ "id", "gender", "speaker_id"])
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch
tedlium = tedlium.map(remove_special_characters)

print("++++++++++++++++++++++++++++++================================================  Tedlium special characters removed")

def extract_all_chars(batch):
  vocab_list = []  # Empty list to store vocab for each element
  for element in batch["text"]:
    all_text = element + " "
    vocab = list(set(all_text))
    vocab_list.append(vocab)
  return {"vocab": vocab_list}


# vocabs = tedlium.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True)   # this line doesn't seem to have any role in the word error metric code 
model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a')
processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')

print("++++++++++++++++++++++++++++++================================================  Speech recognition model and processor loaded correctly")


if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="yongjian/wav2vec2-large-a",
    torch_dtype=torch_dtype,
    device=device,
)

#------------------  another section ------------------

all_predictions = []

print("++++++++++++++++++++++++++++++================================================  te quiero demasiado will be executed next ")


# run streamed inference
for prediction in tqdm(
    pipe(
        KeyDataset(tedlium, "audio"),
        max_new_tokens=128,
        generate_kwargs={"task": "transcribe"},
        batch_size=32,
    ),
    total=len(tedlium),
):
    all_predictions.append(prediction["text"])

#-------------- another cell 
wer_metric = load("wer")

wer_ortho = 100 * wer_metric.compute(
    references=tedlium["text"], predictions=all_predictions
)
print(wer_ortho)