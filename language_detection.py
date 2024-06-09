import langid
from audio_to_text import AudioToText


class LanguageDetector:
    def __init__(self):
        pass

    def detect_language(self, text):
        language, confidence = langid.classify(text)
        return language, confidence


# Example usage
if __name__ == "__main__":
    # Initialize the AudioToText class
    audio_to_text = AudioToText()

    # Convert audio to text
    transcription = audio_to_text.convert("path_to_your_audio_file.wav")
    print("Transcription: ", transcription)

    # Initialize the LanguageDetector class
    detector = LanguageDetector()

    # Detect language of the transcription
    language, confidence = detector.detect_language(transcription)
    print(f"Detected Language: {language}, Confidence: {confidence}")
