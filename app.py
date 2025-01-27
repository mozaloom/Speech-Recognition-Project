import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WhisperProcessor, WhisperForConditionalGeneration

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Wav2Vec2 model and processor
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load Whisper model and processor
whisper_processor = WhisperProcessor.from_pretrained(r"C:\Users\mzlwm\Downloads\SST\SST\Model")
whisper_model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\mzlwm\Downloads\SST\SST\Model")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio_wav2vec(filepath):
    """Transcribe the uploaded audio file using Wav2Vec2."""
    audio, sampling_rate = librosa.load(filepath, sr=16000)
    input_values = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    logits = wav2vec_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec_processor.batch_decode(predicted_ids)[0]
    return transcription

def transcribe_audio_whisper(filepath):
    """Transcribe the uploaded audio file using Whisper."""
    try:
        # Load the audio data using librosa
        audio_data, sampling_rate = librosa.load(filepath, sr=16000)  # Resample to 16kHz

        # Process the audio data using the Whisper processor
        audio_input = whisper_processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")  # Prepare input

        # Generate transcription
        generated_ids = whisper_model.generate(audio_input.input_features)  # Transcribe
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]  # Decode transcription

        return transcription

    except Exception as e:
        raise RuntimeError(f"Whisper transcription error: {e}")

@app.route("/", methods=["GET"])
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/", methods=["POST"])
def upload_audio():
    """Handle audio file upload and transcription."""
    if "audioFile" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["audioFile"]
    model_choice = request.form.get("model", "wav2vec2")  # Default to wav2vec2

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            if model_choice == "wav2vec2":
                transcription = transcribe_audio_wav2vec(filepath)
            elif model_choice == "whisper":
                transcription = transcribe_audio_whisper(filepath)
            else:
                return jsonify({"error": "Invalid model choice"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to process audio file: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

        # Return the transcription as JSON
        return jsonify({"transcription": transcription})

    return jsonify({"error": "Invalid file type. Only .wav and .mp3 files are supported."}), 400

if __name__ == "__main__":
    app.run(debug=True)
