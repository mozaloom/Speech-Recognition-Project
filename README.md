# Speech Recognition Application

## Overview
This project is a web-based speech recognition application that transcribes audio files into text using AI models Wav2Vec2 and Whisper.

## Features
- Supports Wav2Vec2 and Whisper models for transcription.
- Accepts `.wav` and `.mp3` audio files.
- User-friendly web interface.

## Step-by-Step Setup

### Prerequisites
- Python 3.8 or higher installed.
- Basic knowledge of command-line usage.

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/speech-recognition-app.git
   cd speech-recognition-app
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**:
   - Wav2Vec2:
     ```bash
     transformers-cli download facebook/wav2vec2-base-960h
     ```
   - Whisper:
     Ensure Whisper model files are placed in `./whisper-small-hi`.

5. **Run the Application**:
   ```bash
   python app.py
   ```

6. **Access the Application**:
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

7. **Use the Application**:
   - Upload a `.wav` or `.mp3` file.
   - Select the desired model (Wav2Vec2 or Whisper).
   - Click "Transcribe" to get the transcription.

## File Structure
```
.
├── app.py              # Main Flask application
├── templates
│   └── index.html      # Frontend HTML file
├── static
│   ├── styles.css      # CSS for styling
├── uploads             # Directory for uploaded audio files
├── whisper-small-hi    # Directory containing the Whisper model files
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Troubleshooting
- Ensure all dependencies are installed and Python version is compatible.
- Check the `uploads` folder has write permissions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Hugging Face for their Transformers library.
- OpenAI for the Whisper model.
- Flask for enabling rapid development of web applications.

