# 🚀 Build a PDF ingestion and Question/Answering system with LangChain and gemini

This is a Dockerized Node.js application built using the [Express.js](https://expressjs.com/) framework.

## 📁 Folder Structure

```
/your-app
│
├── bin/              # Startup files or utilities
├── node_modules/     # Node.js dependencies
├── public/           # Static files (images, JS, CSS)
├── routes/           # Express route handlers
├── services/         # Business logic/services
├── views/            # View templates (EJS, Pug, etc.)
│
├── .dockerignore     # Files to ignore in Docker builds
├── .env              # Environment variables (do not commit)
├── app.js            # Main application file
├── Dockerfile        # Dockerfile (consider renaming Dockerfile.txt to Dockerfile)
├── package.json      # NPM dependencies and scripts
├── package-lock.json # Dependency lock file
└── README.md         # Project documentation
```

---

## ⚙️ Features

- Express.js web framework
- Docker support for easy deployment
- Environment variable support via `.env`
- Structured folder architecture
- Scalable route & service layers

---

## 🧪 Local Development

### 1. Install Dependencies

```bash
npm install
```

### 2. Create `.env` file

```env
PORT=3000
GOOGLE_API_KEY=gemini API KEY
```

### 3. Run Locally

```bash
node app.js or npm start
```

Or with `nodemon`:

```bash
npx nodemon app.js
```

---

## 🐳 Docker Instructions

### ✅ Build Docker Image

```bash
docker build -t my-express-app Dockerfile

### ▶️ Run Docker Container

```bash
docker run -p 3000:3000 --env-file .env my-express-app
```

---

## 🪪 License

This project is licensed under the MIT License.