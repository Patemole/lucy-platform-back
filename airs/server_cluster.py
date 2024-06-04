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



from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

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
