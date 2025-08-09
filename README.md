# 🚀 AI-Powered Full-Stack Application

A full-stack web application powered by **Svelte (frontend)**, **FastAPI (backend)**, and **Redis** for caching. This project integrates advanced AI capabilities and a blazing-fast modern UI for real-time interaction.

---

## ✨ Features

### 💻 Frontend (Svelte + Vite)

- ⚡ **Fast and reactive UI** with Vite bundling
- 🎨 **Modern, responsive design**
- 🔄 **Live API integration** with the backend
- 📊 **Real-time updates** from AI computations

### 🧠 Backend (FastAPI)

- 🚀 **High-performance Python API**
- 🔍 **AI-powered responses** using LLM APIs (Gemini API integrated)
- ⚙️ **Redis caching** for ultra-fast data retrieval
- 📡 **REST endpoints** for flexible integration

### 🗄️ Redis

- ⚡ **In-memory data store** for rapid key-value access
- 🛠 **Used for caching AI responses** and storing session data

---

## 🛠 Tech Stack

- **Frontend:** [Svelte](https://svelte.dev/) + [Vite](https://vitejs.dev/)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/)
- **Database/Cache:** [Redis](https://redis.io/)
- **AI Integration:** [Google Gemini API](https://ai.google.dev/gemini-api)
- **Containerization:** [Docker](https://www.docker.com/)

---

## 📂 Project Structure

```
project-root/
│
├── frontend/                # Svelte frontend
│   ├── src/                # UI components & pages
│   └── Dockerfile
│
├── backend/                 # FastAPI backend
│   ├── app/                # API routes & services
│   └── Dockerfile
│
├── docker-compose.yml
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Create environment file

Create a `.env` file in the backend folder:

```env
REDIS_HOST=redis
REDIS_PORT=6379
GEMINI_API_KEY=your_api_key_here
```

### 3️⃣ Start with Docker

```bash
docker-compose up --build
```

### 4️⃣ Access the application

- **Frontend:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redis:** localhost:6379

---

## 🧪 Development

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## 🌟 Project Highlights

- 🔥 **End-to-End AI Integration** — seamless flow from UI to AI response
- ⚡ **Real-time Caching** with Redis for optimal performance
- 📱 **Responsive & Modern UI** built with Svelte
- 🛳 **Fully Dockerized** — consistent deployment anywhere
- 📖 **Well-structured & Extensible** codebase

---

## 📜 License

This project is licensed under the MIT License — feel free to use and modify.

---

💡 *Made with love and coffee.*
