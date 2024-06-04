import multiprocessing
import server_cluster
import server_files

def run_serveur_cluster():
    server_cluster.run()
    

def run_serveur_file():
    server_files.run()


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_serveur_cluster)
    p2 = multiprocessing.Process(target=run_serveur_file)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()