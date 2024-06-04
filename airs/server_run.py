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