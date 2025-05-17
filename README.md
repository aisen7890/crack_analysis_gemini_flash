# Concrete Crack Analysis AI

A Streamlit-based web application that uses Google's Gemini AI model to analyze concrete cracks through images and text. The application features voice input, PDF knowledge base integration, and automatic report generation.

## Features

- Image analysis using Google's Gemini AI model
- Multiple input methods:
  - Image upload
  - Camera capture
  - Text input
  - Voice input
- PDF knowledge base integration using ChromaDB
- Automatic text-to-speech response
- Report generation capability
- Conversation history tracking

## Prerequisites

- Python 3.12 or higher
- Google API key for Gemini AI
- FFmpeg (required for Whisper voice recognition)
- VS Code (optional, for using launch configuration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Option 1: Using Streamlit CLI
1. Start the Streamlit application from terminal:
```bash
streamlit run main.py
```

### Option 2: Using VS Code
1. Open the project in VS Code
2. Open launch.json file insidei.vscode folder.
3. Press F5 or click the "Run and Debug" button 

Both methods will:
- Start the Streamlit server
- Open your web browser automatically to the provided local URL (typically http://localhost:8000)

### Using the Application
- Upload or capture an image of a concrete crack
- Type your question or use voice input
- View the AI's analysis and response
- Use the sidebar to manage the knowledge base
- Generate reports by asking the AI to "save to reports"

## Knowledge Base Management

- Upload PDF documents through the sidebar
- View the number of documents in the knowledge base
- Clear the knowledge base when needed

## Report Generation

To generate a report:
1. Upload or capture an image
2. Ask your question
3. Include the phrase "save to reports" in your query
4. The system will automatically save the image, response, and audio to the reports directory

## Directory Structure

- `main.py`: Main application file
- `requirements.txt`: Project dependencies
- `conversation_history.json`: Stores chat history
- `chroma_db/`: ChromaDB database directory
- `reports/`: Generated reports directory
- `.vscode/launch.json`: VS Code launch configuration

## Notes

- The application requires an active internet connection for AI processing
- Voice recognition requires FFmpeg to be installed on your system
- Large PDF files may take longer to process
- Reports are saved with timestamps in the filename

## License

[Your chosen license]

## Contributing

[Your contribution guidelines] 