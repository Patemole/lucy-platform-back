import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("Starting the correct file")
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from airs.server_files import create_app as create_files_app
from airs.server_academic_advisor import create_app as create_academic_advisor_app


# Configurer le logging
logging.basicConfig(level=logging.DEBUG)  # Passer à DEBUG pour plus de détails
logger = logging.getLogger(__name__)

print("Démarrage de l'application")
logger.info("Démarrage de l'application")

app = FastAPI()

# Configurer CORS avant de monter les sous-applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour journaliser les requêtes
@app.middleware("http")
async def log_request(request, call_next):
    print(f"Request: {request.method} {request.url}")
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    logger.info(f"Response: {response.status_code}")
    return response


# Configurer le serveur pour servir les fichiers statiques - un pour l'academic advisor, l'autre pour le teacher
static_dir = "/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html" #Le graphe d'origine pour les professeurs
static_dir_academic_advisor = "/Users/gregoryhissiger/pinecone_client_test" #Le graphe d'origine pour les professeurs

app.mount("/static/teacher", StaticFiles(directory=static_dir), name="static_teacher")
app.mount("/static/academic_advisor", StaticFiles(directory=static_dir_academic_advisor), name="static_academic_advisor")

try:
    print("Création de l'application files")
    logger.info("Création de l'application files")
    files_app = create_files_app()
    if files_app is None:
        print("Application files n'a pas été créée")
        logger.error("Application files n'a pas été créée")
    else:
        print("Application files créée avec succès")
        logger.info("Application files créée avec succès")

    print("Création de l'application academic advisor")
    logger.info("Création de l'application academic advisor")
    chat_app = create_academic_advisor_app()
    if chat_app is None:
        print("Application academic advisor n'a pas été créée")
        logger.error("Application academic advisor n'a pas été créée")
    else:
        print("Application academic advisor créée avec succès")
        logger.info("Application academic advisor créée avec succès")

    print("Montage des applications")
    logger.info("Montage des applications")
    if files_app:
        app.mount("/files", files_app)
    if chat_app:
        app.mount("/chat", chat_app)
    print("Applications montées avec succès")
    logger.info("Applications montées avec succès")
except Exception as e:
    print(f"Erreur lors de la création ou du montage des applications: {e}")
    logger.exception("Erreur lors de la création ou du montage des applications: %s", e)
    raise e

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    print(f"Démarrage du serveur sur le port {port}")
    logger.info(f"Démarrage du serveur sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")  # Passer log_level à debug pour plus de détails
