# Smart Masker

An intelligent image masking tool powered by Segment Anything 2 (SAM2) model.

## Features

- Interactive image masking with SAM2
- Multiple mask management
- Real-time mask visualization
- Export masks as PNG files
- User-friendly interface

## Project Structure

```
smart-masker/
├── frontend/           # React frontend application
│   ├── src/           # Source code
│   └── public/        # Static files
├── backend/           # FastAPI backend server
│   ├── app/          # Application code
│   └── models/       # SAM2 model files
└── requirements.txt   # Python dependencies
```

## Setup

### Backend

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

## Usage

1. Open the application in your browser
2. Click "Select Image" to upload an image
3. Left-click on objects to create masks
4. Right-click on masks to delete them
5. Click "Export Mask" to save the mask as PNG

## Requirements

- Python 3.8+
- Node.js 14+
- Modern web browser (Chrome, Firefox, Edge) 