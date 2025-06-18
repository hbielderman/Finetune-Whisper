import whisperx
import librosa
import numpy as np
import soundfile as sf

class WhisperChunker:
    def __init__(self, model_name="openai/whisper-small", device="cuda", language_code="nl", chunk_duration=30):
        self.device = device
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.language_code = language_code

        print("🔎 Loading Whisper model...")
        self.model = whisperx.load_model(self.model_name, device=self.device)

        print("📏 Loading alignment model...")
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code=self.language_code, device=self.device)

    def transcribe_with_vad(self, audio_path):
        result = self.model.transcribe(
            audio_path, 
            vad_filter=True, 
            vad_parameters={"max_speech_duration_s": self.chunk_duration}
        )
        return result

    def align(self, transcription_result, audio_path):
        segments = transcription_result["segments"]
        aligned_result = whisperx.align(
            segments, self.alignment_model, self.metadata, audio_path, self.device
        )
        return aligned_result["word_segments"]

    def chunk_audio(self, audio_path):
        transcription_result = self.transcribe_with_vad(audio_path)
        word_segments = self.align(transcription_result, audio_path)

        print("\n📋 Word Segments:")
        for word in word_segments:
            print(f"Word: '{word['text']}' | Start: {word['start']} | End: {word['end']} | Duration: {word['end'] - word['start']}")

        chunks = []
        current_chunk = []
        current_start = word_segments[0]['start']
        current_end = current_start

        for word in word_segments:
            word_start = word['start']
            word_end = word['end']
            word_text = word['text']

            if (word_end - current_start) <= self.chunk_duration:
                print(f"Adding '{word_text}' to current chunk (current_start={current_start:.2f}, word_end={word_end:.2f})")
                current_chunk.append(word_text)
                current_end = word_end
            else:
                print(f"Chunk full: [{current_start:.2f} - {current_end:.2f}] -> '{' '.join(current_chunk)}'")
                chunks.append({
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_chunk)
                })
                current_start = word_start
                current_end = word_end
                current_chunk = [word_text]

        if current_chunk:
            print(f"Final chunk: [{current_start:.2f} - {current_end:.2f}] -> '{' '.join(current_chunk)}'")
            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_chunk)
            })

        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_results = []
        for chunk in chunks:
            start_sample = int(chunk["start"] * sr)
            end_sample = int(chunk["end"] * sr)
            chunk_audio = audio[start_sample:end_sample]
            chunk_results.append((chunk_audio, chunk["text"]))

        print("\n✅ Chunks created:")
        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx+1}: Start={chunk['start']:.2f}, End={chunk['end']:.2f}, Duration={chunk['end']-chunk['start']:.2f} sec, Text='{chunk['text']}'")

        return chunk_results
