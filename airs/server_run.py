'''
import multiprocessing
import server_cluster
import server_files
import server_academic_advisor

def run_serveur_cluster():
    server_cluster.run()
    

def run_serveur_file():
    server_files.run()


def run_serveur_chat():
    server_academic_advisor.run()


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_serveur_cluster)
    p2 = multiprocessing.Process(target=run_serveur_file)
    p3 = multiprocessing.Process(target=run_serveur_chat)
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()
'''


from fastapi import FastAPI
from server_cluster import create_app as create_cluster_app
from server_files import create_app as create_files_app
from server_academic_advisor import create_app as create_academic_advisor_app

app = FastAPI()

cluster_app = create_cluster_app()
files_app = create_files_app()
chat_app = create_academic_advisor_app()

app.mount("/cluster", cluster_app)
app.mount("/files", files_app)
app.mount("/chat", chat_app)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
