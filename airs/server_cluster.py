'''
#SERVEUR POUR AFFICHER LE GRAPHE CLUSTER SUR LE DASHBOARD PROF 

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Configurer le serveur pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html"), name="static")

@app.get("/get_graph")
async def get_graph():
    file_path = os.path.join("/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html", "cluster_plot.html")
    if not os.path.exists(file_path):
        return {"error": "Graph file not found."}
    return FileResponse(file_path, media_type='text/html')


def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

if __name__ == "__main__":
    run()
'''


'''
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Configurer le serveur pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html"), name="static")

@app.get("/get_graph")
async def get_graph():
    file_path = os.path.join("/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html", "cluster_plot.html")
    if not os.path.exists(file_path):
        return {"error": "Graph file not found."}
    return FileResponse(file_path, media_type='text/html')

def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
'''


'''
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vous pouvez spécifier une liste d'origines spécifiques au lieu de ["*"]
    allow_credentials=True,
    allow_methods=["*"],  # Vous pouvez spécifier des méthodes spécifiques comme ["GET", "POST"]
    allow_headers=["*"],  # Vous pouvez spécifier des headers spécifiques comme ["Content-Type"]
)

# Définir le chemin de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le dossier analytics/generated_html
STATIC_DIR = os.path.join(BASE_DIR, 'analytics', 'generated_html')

# Configurer le serveur pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/get_graph")
async def get_graph():
    file_path = os.path.join(STATIC_DIR, "cluster_plot.html")
    if not os.path.exists(file_path):
        return {"error": "Graph file not found."}
    return FileResponse(file_path, media_type='text/html')

def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''


'''

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)  # Passer à DEBUG pour plus de détails
logger = logging.getLogger(__name__)

app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Définir le chemin de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chemin vers le dossier analytics/generated_html
STATIC_DIR = os.path.join(BASE_DIR, 'analytics', 'generated_html')

# Configurer le serveur pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/get_graph")
async def get_graph():
    file_name = "cluster_plot.html"
    file_path = os.path.join(STATIC_DIR, file_name)
    
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "Graph file not found."}, status_code=404)
    
    # URL complète du fichier
    file_url = f"/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html/{file_name}"
    logger.info(f"Graph file URL: {file_url}")
    
    # Retourner l'URL complète dans la réponse JSON
    return JSONResponse(content={"url_cluster": file_url})


def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''

'''
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import logging

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Définir le chemin de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, 'analytics', 'generated_html')

@app.get("/cluster/get_graph")
async def get_graph():
    file_name = "cluster_plot.html"
    file_path = os.path.join(STATIC_DIR, file_name)
    
    if not os.path.exists(file_path):
        logger.error("Graph file not found.")
        return JSONResponse(content={"error": "Graph file not found."}, status_code=404)
    
    # URL complète du fichier
    base_url = "/static/"
    file_url_part1 = "cluster"
    file_url_part2 = "_plot"
    file_url_part3 = ".html"
    
    logger.info(f"Graph file URL parts: {base_url}, {file_url_part1}, {file_url_part2}, {file_url_part3}")
    
    # Retourner les parties de l'URL dans la réponse JSON
    return JSONResponse(content={
        "base_url": base_url,
        "file_url_part1": file_url_part1,
        "file_url_part2": file_url_part2,
        "file_url_part3": file_url_part3
    })


def create_app():
    return app


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
'''



from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurer le serveur pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html"), name="static")

@app.get("/get_graph")
async def get_graph():
    file_path = os.path.join("/Users/gregoryhissiger/Lucy-platform-v1/back_socratic/analytics/generated_html", "cluster_plot.html")
    if not os.path.exists(file_path):
        return {"error": "Graph file not found."}
    return FileResponse(file_path, media_type='text/html')

def create_app():
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)