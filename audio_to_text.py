import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class AudioToText:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def load_audio(self, file_path):
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000)  # Resample to 16kHz
        audio = audio.set_channels(1)  # Convert to mono

        # Convert to numpy array
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
        audio_array = audio_array / (2 ** 15)  # Normalize to [-1, 1]

        return audio_array

    def convert(self, file_path):
        audio_array = self.load_audio(file_path)

        # Tokenize input
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

        # Perform inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # Decode the predicted ids to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]


# Example usage
if __name__ == "__main__":
    audio_to_text = AudioToText()
    transcription = audio_to_text.convert("path_to_the_audio_file.wav")
    print("Transcription: ", transcription)
