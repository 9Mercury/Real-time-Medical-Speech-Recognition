import asyncio
import queue
import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ————————————
# 0) Whisper ASR pipeline setup
# ————————————
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "Crystalcareai/Whisper-Medicalv1"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# ————————————
# 1) Real-time recording params
# ————————————
FS = 16000              # sample rate
CHUNK_DURATION = 20    # seconds per inference chunk
CHUNK_SAMPLES = int(FS * CHUNK_DURATION)

# Shared between callback and asyncio
audio_buffer = []
audio_queue  = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """ sounddevice callback: accumulate until CHUNK_SAMPLES then enqueue """
    global audio_buffer
    audio_buffer.append(indata.copy())
    total = sum(buf.shape[0] for buf in audio_buffer)
    if total >= CHUNK_SAMPLES:
        data = np.concatenate(audio_buffer, axis=0)
        chunk = data[:CHUNK_SAMPLES]
        remainder = data[CHUNK_SAMPLES:]
        audio_buffer = [remainder] if remainder.size else []
        audio_queue.put(chunk.flatten())

async def transcribe_loop():
    """ Async task: pull small audio chunks and transcribe """
    loop = asyncio.get_running_loop()
    while True:
        # get next chunk (blocks in threadpool)
        audio_array = await loop.run_in_executor(None, audio_queue.get)
        # run ASR in threadpool
        result = await loop.run_in_executor(
            None,
            lambda: pipe({"array": audio_array, "sampling_rate": FS})
        )
        print("Recognized:", result["text"])
        audio_queue.task_done()

async def main():
    # start streaming input
    stream = sd.InputStream(
        samplerate=FS,
        channels=1,
        callback=audio_callback
    )
    stream.start()
    print("Listening (Ctrl+C to stop)...")
    try:
        await transcribe_loop()  # runs until cancelled
    finally:
        stream.stop()
        stream.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")
