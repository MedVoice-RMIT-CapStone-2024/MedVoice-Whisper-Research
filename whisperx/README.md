# WhisperX Setup Guide

This repository provides fast automatic speech recognition (70x realtime with large-v2) with word-level timestamps and speaker diarization.

- ⚡️ Batched inference for 70x realtime transcription using whisper large-v2
- 🪶 [faster-whisper](https://github.com/guillaumekln/faster-whisper) backend, requires <8GB gpu memory for large-v2 with beam_size=5
- 🎯 Accurate word-level timestamps using wav2vec2 alignment
- 👯‍♂️ Multispeaker ASR using speaker diarization from [pyannote-audio](https://github.com/pyannote/pyannote-audio) (speaker ID labels) 
- 🗣️ VAD preprocessing, reduces hallucination & batching with no WER degradation



**Whisper** is an ASR model [developed by OpenAI](https://github.com/openai/whisper), trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching.

**Phoneme-Based ASR** A suite of models finetuned to recognise the smallest unit of speech distinguishing one word from another, e.g. the element p in "tap". A popular example model is [wav2vec2.0](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).

**Forced Alignment** refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.

**Voice Activity Detection (VAD)** is the detection of the presence or absence of human speech.

**Speaker Diarization** is the process of partitioning an audio stream containing human speech into homogeneous segments according to the identity of each speaker.


## Prerequisites

- Python 3.10
- Conda
- Git

## Setup Instructions

### 1. Create Python3.10 environment

`conda create --name whisperx python=3.10`

`conda activate whisperx`


### 2. Install PyTorch, e.g. for Linux and Windows CUDA11.8:

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

See other methods [here.](https://pytorch.org/get-started/previous-versions/#v200)

### 3. Install this repo

`pip install git+https://github.com/m-bain/whisperx.git`

If already installed, update package to most recent commit

`pip install git+https://github.com/m-bain/whisperx.git --upgrade`

If wishing to modify this package, clone and install in editable mode:
```
$ git clone https://github.com/m-bain/whisperX.git
$ cd whisperX
$ pip install -e .
```

You may also need to install ffmpeg, rust etc. Follow openAI instructions here https://github.com/openai/whisper#setup.

### 4. Install `ffmpeg` and `rust`
It also requires the command-line tool `ffmpeg` to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```
### English

Run whisper on example segment (using default params, whisper small) add `--highlight_words True` to visualise word timings in the .srt file.

    whisperx examples/sample01.wav

https://storage.googleapis.com/medvoice-sgp-audio-bucket/Cambridge-IELTS-16-Listening-Test-1-Part-1.mp3


For increased timestamp accuracy, at the cost of higher gpu mem, use bigger models (bigger alignment model not found to be that helpful, see paper) e.g.

    whisperx examples/sample01.wav --model large-v2 --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4


To label the transcript with speaker ID's (set number of speakers if known e.g. `--min_speakers 2` `--max_speakers 2`):

    whisperx examples/sample01.wav --model large-v2 --diarize --highlight_words True

To run on CPU instead of GPU (and for running on Mac OS X):

    whisperx examples/sample01.wav --compute_type int8

### Example CLI Usage

To use the WhisperX tool for speaker diarization and transcription, you can use the following command:

```sh
whisperx "https://storage.googleapis.com/medvoice-sgp-audio-bucket/Cambridge-IELTS-16-Listening-Test-1-Part-1.mp3" --model large-v2 --diarize --highlight_words True --language en --compute_type int8 --hf_token [YOUR_HF_TOKEN]
```
This command takes an audio file URL and performs speaker diarization and transcription using the specified model and language. Here's what each option does:

- --model large-v2: Specifies the model to use for transcription. In this case, it's large-v2.
- --diarize: Enables speaker diarization, which separates the audio into segments based on the speaker.
- --highlight_words True: Enables word highlighting in the transcription.
- --language en: Sets the language for transcription. In this case, it's English (en). You can use other language codes as well, such as fr for French, de for German, etc.
- --compute_type int8: Sets the compute type for the model. In this case, it's int8.
- --hf_token [YOUR_HF_TOKEN]: Specifies the Hugging Face token to use for accessing models from the Hugging Face model hub.
***Please ensure that you have the necessary permissions to access the audio file and the Hugging Face model.***

### Other languages

The phoneme ASR alignment model is *language-specific*, for tested languages these models are [automatically picked from torchaudio pipelines or huggingface](https://github.com/m-bain/whisperX/blob/e909f2f766b23b2000f2d95df41f9b844ac53e49/whisperx/transcribe.py#L22).
Just pass in the `--language` code, and use the whisper `--model large`.

Currently default models provided for `{en, fr, de, es, it, ja, zh, nl, uk, pt}`. If the detected language is not in this list, you need to find a phoneme-based ASR model from [huggingface model hub](https://huggingface.co/models) and test it on your data.

#### E.g. German
    whisperx --model large-v2 --language de examples/sample_de_01.wav

## Python usage  🐍

```python
import whisperx
import gc 

device = "cuda" 
audio_file = "audio.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
```


<h2 align="left" id="whisper-mod">Technical Details 👷‍♂️</h2>

For specific details on the batching and alignment, the effect of VAD, as well as the chosen alignment model, see the preprint [paper](https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf).

To reduce GPU memory requirements, try any of the following (2. & 3. can affect quality):
1. Reduce batch size, e.g. `--batch_size 4`
2. use a smaller ASR model `--model base`
3. Use lighter compute type `--compute_type int8`
4. Use different device `--device DEVICE` device to use for PyTorch inference (default: cpu or change to cuda)

Transcription differences from openai's whisper:
1. Transcription without timestamps. To enable single pass batching, whisper inference is performed `--without_timestamps True`, this ensures 1 forward pass per sample in the batch. However, this can cause discrepancies the default whisper output.
2. VAD-based segment transcription, unlike the buffered transcription of openai's. In Wthe WhisperX paper we show this reduces WER, and enables accurate batched inference
3.  `--condition_on_prev_text` is set to `False` by default (reduces hallucination)

