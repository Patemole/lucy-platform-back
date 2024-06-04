import logging
from fastapi import FastAPI
from server_cluster import create_app as create_cluster_app
from server_files import create_app as create_files_app
from server_academic_advisor import create_app as create_academic_advisor_app

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Démarrage de l'application")

app = FastAPI()

try:
    logger.info("Création de l'application cluster")
    cluster_app = create_cluster_app()
    logger.info("Application cluster créée avec succès")

    logger.info("Création de l'application files")
    files_app = create_files_app()
    logger.info("Application files créée avec succès")

    logger.info("Création de l'application academic advisor")
    chat_app = create_academic_advisor_app()
    logger.info("Application academic advisor créée avec succès")

    logger.info("Montage des applications")
    app.mount("/cluster", cluster_app)
    app.mount("/files", files_app)
    app.mount("/chat", chat_app)
    logger.info("Applications montées avec succès")
except Exception as e:
    logger.exception("Erreur lors de la création ou du montage des applications: %s", e)
    raise e

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Démarrage du serveur sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
