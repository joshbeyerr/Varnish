# Next.js + FastAPI Test Application


Start backend by just cd into the backend folder and running the file (python main.py)
Launch website by cd into frontender folder and npm run dev







A simple test application demonstrating how to connect a Next.js frontend to a Python FastAPI backend.

## Project Structure

```
fastApi-test/
├── backend/                 # Python FastAPI backend
│   ├── main.py             # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js app directory
│   │   ├── globals.css    # Global styles
│   │   ├── layout.tsx     # Root layout
│   │   └── page.tsx       # Main page
│   ├── package.json       # Node.js dependencies
│   ├── tsconfig.json      # TypeScript configuration
│   ├── tailwind.config.js # Tailwind CSS configuration
│   ├── postcss.config.js  # PostCSS configuration
│   └── next.config.js     # Next.js configuration
└── README.md              # This file
```

## Features

- **Backend (FastAPI)**:
  - RESTful API with CRUD operations for items
  - CORS configured for frontend communication
  - Health check endpoint
  - In-memory data storage for simplicity

- **Frontend (Next.js)**:
  - Modern React with TypeScript
  - Tailwind CSS for styling
  - API integration with Axios
  - Real-time backend status checking
  - Add and view items functionality

## Setup Instructions

### Prerequisites

- Python 3.8+ installed
- Node.js 18+ installed
- npm or yarn package manager

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the FastAPI server:**
   ```bash
   python main.py
   ```

   The backend will be available at: `http://localhost:8000`

   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the Next.js development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at: `http://localhost:3000`

## Usage

1. **Start both servers:**
   - Start the backend server first (port 8000)
   - Then start the frontend server (port 3000)

2. **Open your browser:**
   - Navigate to `http://localhost:3000`
   - You should see the main application page

3. **Test the connection:**
   - The page will show the backend connection status
   - If connected, you'll see "connected ✅"
   - If not connected, you'll see "disconnected ❌"

4. **Add items:**
   - Use the form at the top to add new items
   - Items will appear in the list below
   - You can refresh the list to see all items

## API Endpoints

### Backend API (FastAPI)

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /items` - Get all items
- `GET /items/{item_id}` - Get specific item
- `POST /items` - Create new item

### Example API Usage

```bash
# Get all items
curl http://localhost:8000/items

# Create a new item
curl -X POST http://localhost:8000/items \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Item", "description": "A test item"}'

# Health check
curl http://localhost:8000/health
```

## Troubleshooting

### Backend Issues

1. **Port already in use:**
   - Change the port in `backend/main.py` (line with `uvicorn.run`)
   - Update the `API_BASE_URL` in `frontend/app/page.tsx`

2. **CORS errors:**
   - Make sure the frontend URL is in the `allow_origins` list in `backend/main.py`

3. **Python dependencies not found:**
   - Make sure you're in the virtual environment
   - Run `pip install -r requirements.txt` again

### Frontend Issues

1. **Cannot connect to backend:**
   - Make sure the backend server is running on port 8000
   - Check the backend status indicator on the frontend

2. **Build errors:**
   - Delete `node_modules` and `package-lock.json`
   - Run `npm install` again

3. **TypeScript errors:**
   - Make sure you have the correct TypeScript version
   - Check that all dependencies are properly installed

## Development Notes

- The backend uses in-memory storage, so data will be lost when the server restarts
- For production, you'd want to add a proper database (PostgreSQL, MongoDB, etc.)
- The CORS configuration allows all origins - in production, you should restrict this
- Consider adding authentication and authorization for production use

## Next Steps

- Add a database (SQLite, PostgreSQL, etc.)
- Implement user authentication
- Add more complex CRUD operations
- Add form validation
- Implement error handling and loading states
- Add unit tests
- Deploy to production (Vercel for frontend, Railway/Heroku for backend)
