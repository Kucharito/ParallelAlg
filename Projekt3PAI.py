"""
Paralelna implementacia algoritmu PageRank pre Berkeley-Stanford web graph.
- Paralelne nacitanie grafu do riedkej matice susednosti
- Paralelny vypocet PageRank (data-driven pristup)
"""

import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Konfiguracia
FILENAME = "web-BerkStan.txt"
NUM_THREADS = min(8, (os.cpu_count() or 4))
DAMPING = 0.85
THRESHOLD = 1e-6
MAX_ITERATIONS = 100


class SparseGraph:
    """Riedka reprezentacia orientovaneho grafu."""
    def __init__(self):
        self.out_edges = defaultdict(list)  # slovnik: vrchol -> zoznam vrcholov kam vedu hrany z neho
        self.in_edges = defaultdict(list)  # slovnik: vrchol -> zoznam vrcholov odkial vedu hrany do neho
        self.vertices = set()  # mnozina vsetkych vrcholov v grafe
        self.lock = threading.Lock()

    def add_edges_batch(self, edges):
        """Prida viacero hran naraz (thread-safe)."""
        with self.lock:
            for from_v, to_v in edges:  # pre kazdu hranu (od, do)
                self.out_edges[from_v].append(to_v)  # pridaj do odchadzajucich hran
                self.in_edges[to_v].append(from_v)  # pridaj do prichadzajucich hran
                self.vertices.add(from_v)  # pridaj zdrojovy vrchol do mnoziny
                self.vertices.add(to_v)  # pridaj cielovy vrchol do mnoziny


def get_file_chunks(filename, num_chunks):
    """Rozdeli subor na casti pre paralelne nacitanie."""
    file_size = os.path.getsize(filename)  # zisti velkost suboru v bajtoch
    chunk_size = file_size // num_chunks  # velkost jednej casti (celociselne delenie)
    chunks = []  # zoznam casti (start, end)

    with open(filename, 'rb') as f:  # otvor subor v binarnom mode
        start = 0  # zaciatok prvej casti
        for i in range(num_chunks):  # pre kazdu cast
            if i == num_chunks - 1:  # posledna cast
                end = file_size  # konci na konci suboru
            else:
                end = start + chunk_size  # predpokladany koniec casti
                f.seek(end)  # presun sa na tuto poziciu
                f.readline()  # docitaj do konca riadku (aby sme nerozdelili riadok)
                end = f.tell()  # skutocny koniec casti (za koncom riadku)
            chunks.append((start, end))  # pridaj cast do zoznamu
            start = end  # dalsia cast zacina kde predchadzajuca skoncila
    return chunks  # vrat zoznam casti


def load_chunk(filename, start_pos, end_pos, graph):
    """Nacita cast suboru a prida hrany do grafu."""
    edges = []  # lokalny zoznam hran pre tuto cast

    with open(filename, 'rb') as f:  # otvor subor v binarnom mode
        f.seek(start_pos)  # presun sa na zaciatok casti
        data = f.read(end_pos - start_pos)  # nacitaj celu cast naraz (rychlejsie)

    for line in data.splitlines():
        if not line or line[0] == 35:  # 35 = '#'
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue

    graph.add_edges_batch(edges)  # pridaj vsetky hrany do grafu (thread-safe)
    return len(edges)  # vrat pocet nacitanych hran


def load_graph_parallel(filename, num_threads):
    """Paralelne nacita graf zo suboru."""
    graph = SparseGraph()  # vytvor prazdny graf
    chunks = get_file_chunks(filename, num_threads)  # rozdel subor na casti

    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # vytvor pool vlakien
        futures = [executor.submit(load_chunk, filename, s, e, graph) for s, e in chunks]  # spusti nacitanie kazde casti v samostatnom vlakne
        total = sum(f.result() for f in as_completed(futures))  # pockaj na dokoncenie a scitaj pocet hran

    print(f"Nacitanych {total} hran, {len(graph.vertices)} vrcholov")  # vypis statistiku
    return graph  # vrat nacitany graf


def compute_pagerank_chunk(vertices_chunk, graph, current_pr, damping, num_vertices):
    """Vypocita PageRank pre podmnozinu vrcholov."""
    new_pr = {}  # slovnik pre nove hodnoty PR
    base_value = (1 - damping) / num_vertices  # zakladna hodnota: (1-d)/N - teleportacna zlozka

    for vertex in vertices_chunk:  # pre kazdy vrchol v tejto casti
        pr_sum = sum(  # sucet prispevkov od vsetkych donorov
            current_pr[donor] / len(graph.out_edges[donor])  # PR donora / pocet jeho odchadzajucich hran
            for donor in graph.in_edges[vertex]  # pre kazdy vrchol ktory ma hranu do tohto vrcholu
            if graph.out_edges[donor]  # len ak ma donor nejake odchadzajuce hrany (nie je dangling)
        )
        new_pr[vertex] = base_value + damping * pr_sum  # novy PR = (1-d)/N + d * suma prispevkov

    return new_pr  # vrat slovnik s novymi hodnotami PR


def pagerank_parallel(graph, damping, threshold, max_iterations, num_threads):
    """Paralelny vypocet PageRank (data-driven pristup)."""
    vertices = list(graph.vertices)  # zoznam vsetkych vrcholov
    num_vertices = len(vertices)  # pocet vrcholov N

    current_pr = {v: 1.0 / num_vertices for v in vertices}  # inicializacia: PR(v) = 1/N pre vsetky vrcholy

    chunk_size = (num_vertices + num_threads - 1) // num_threads  # velkost casti pre kazde vlakno (zaokruhlene hore)
    vertex_chunks = [vertices[i*chunk_size:(i+1)*chunk_size] for i in range(num_threads)]  # rozdelenie vrcholov na casti
    vertex_chunks = [c for c in vertex_chunks if c]  # odstranenie prazdnych casti

    for iteration in range(max_iterations):  # hlavny cyklus iteracii
        new_pr = {}  # novy slovnik pre PR hodnoty

        with ThreadPoolExecutor(max_workers=num_threads) as executor:  # pool vlakien
            futures = [  # spustenie vypoctu pre kazdu cast paralelne
                executor.submit(compute_pagerank_chunk, chunk, graph, current_pr, damping, num_vertices)
                for chunk in vertex_chunks
            ]
            for future in as_completed(futures):  # ked sa vlakno dokonci
                new_pr.update(future.result())  # pridaj jeho vysledky do noveho PR slovnika

        max_diff = max(abs(new_pr[v] - current_pr[v]) for v in vertices)  # najdi maximalnu zmenu PR
        current_pr = new_pr  # aktualizuj aktualny PR

        if (iteration + 1) % 10 == 0:  # kazdu 10. iteraciu
            print(f"Iteracia {iteration + 1}: max zmena = {max_diff:.10f}")  # vypis progress

        if max_diff < threshold:  # ak je zmena mensia ako prah
            print(f"Konvergencia po {iteration + 1} iteraciach")  # oznam konvergenciu
            break  # ukonci iteracie

    return current_pr  # vrat finalne PR hodnoty


def main():
    print("=" * 50)
    print("PARALELNY PAGERANK")
    print("=" * 50)

    # Nacitanie grafu
    print("\n[1/2] Nacitanie grafu...")
    start = time.time()  # zaciatok merania casu
    graph = load_graph_parallel(FILENAME, NUM_THREADS)  # paralelne nacitaj graf
    print(f"Cas: {time.time() - start:.2f}s")  # vypis cas nacitania

    # Vypocet PageRank
    print("\n[2/2] Vypocet PageRank...")
    start = time.time()  # zaciatok merania casu
    pagerank = pagerank_parallel(graph, DAMPING, THRESHOLD, MAX_ITERATIONS, NUM_THREADS)  # paralelny vypocet PR
    print(f"Cas: {time.time() - start:.2f}s")  # vypis cas vypoctu

    # Top 10 vysledkov
    top10 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]  # zorad podla PR zostupne a vyber top 10
    print("\nTop 10 vrcholov:")
    for i, (v, pr) in enumerate(top10, 1):  # pre kazdy z top 10 vrcholov
        print(f"{i}. Vrchol {v}: {pr:.10f}")  # vypis poradie, ID vrcholu a jeho PR


if __name__ == "__main__":
    main()  # spusti hlavnu funkciu
