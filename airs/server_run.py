'''
import logging
from fastapi import FastAPI
from airs.server_cluster import create_app as create_cluster_app
from airs.server_files import create_app as create_files_app
from airs.server_academic_advisor import create_app as create_academic_advisor_app
from fastapi.middleware.cors import CORSMiddleware

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Démarrage de l'application")

app = FastAPI()

origins = [
    "http://usyd.localhost:3001",  # Your frontend's URL
    # Add more origins if needed
]

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],  # Vous pouvez spécifier une liste d'origines spécifiques au lieu de ["*"]
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Vous pouvez spécifier des méthodes spécifiques comme ["GET", "POST"]
    allow_headers=["*"],  # Vous pouvez spécifier des headers spécifiques comme ["Content-Type"]
)

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
'''


'''
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from airs.server_cluster import create_app as create_cluster_app
from airs.server_files import create_app as create_files_app
from airs.server_academic_advisor import create_app as create_academic_advisor_app

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

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
'''



'''
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from airs.server_cluster import create_app as create_cluster_app
from airs.server_files import create_app as create_files_app
from airs.server_academic_advisor import create_app as create_academic_advisor_app

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

try:
    logger.info("Création de l'application cluster")
    cluster_app = create_cluster_app()
    if cluster_app is None:
        logger.error("Application cluster n'a pas été créée")
    else:
        logger.info("Application cluster créée avec succès")

    logger.info("Création de l'application files")
    files_app = create_files_app()
    if files_app is None:
        logger.error("Application files n'a pas été créée")
    else:
        logger.info("Application files créée avec succès")

    logger.info("Création de l'application academic advisor")
    chat_app = create_academic_advisor_app()
    if chat_app is None:
        logger.error("Application academic advisor n'a pas été créée")
    else:
        logger.info("Application academic advisor créée avec succès")

    logger.info("Montage des applications")
    if cluster_app:
        app.mount("/cluster", cluster_app)
    if files_app:
        app.mount("/files", files_app)
    if chat_app:
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
'''



'''
print("Starting the correct file")
import sys
import os
# Ajouter le répertoire du module 'airs' à sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from airs.server_cluster import create_app as create_cluster_app
from airs.server_files import create_app as create_files_app
from airs.server_academic_advisor import create_app as create_academic_advisor_app

# Ajouter le répertoire du module 'airs' à sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

try:
    print("Création de l'application cluster")
    logger.info("Création de l'application cluster")
    cluster_app = create_cluster_app()
    if cluster_app is None:
        print("Application cluster n'a pas été créée")
        logger.error("Application cluster n'a pas été créée")
    else:
        print("Application cluster créée avec succès")
        logger.info("Application cluster créée avec succès")

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
    if cluster_app:
        app.mount("/cluster", cluster_app)
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
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 5001))
    print(f"Démarrage du serveur sur le port {port}")
    logger.info(f"Démarrage du serveur sur le port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")  # Passer log_level à debug pour plus de détails
'''




#MODIFICATION DU CODE POUR SERVIR LE FICHIER STATIQUE ET ENLEVER LE MONTAGE DE SERVER_CLUSTER QUI NE SERT À RIEN
# Ajouter le répertoire du module 'airs' à sys.path
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

# Configurer le serveur pour servir les fichiers statiques
static_dir = "/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
