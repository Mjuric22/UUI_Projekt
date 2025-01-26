import numpy as np
import matplotlib.pyplot as plt # da mogu vizualno prikazat labirint
from heapq import heappop, heappush # da mogu prioritet stavit na algoritam A*

def generiraj_prazan_labirint(visina, sirina): 
    return np.zeros((visina, sirina), dtype=int)

def dodaj_zidove_labirintu(labirint, gustoca=0.40):
    visina, sirina = labirint.shape
    broj_zidova = int(gustoca * visina * sirina)
    for _ in range(broj_zidova):
        x, y = np.random.randint(0, visina), np.random.randint(0, sirina)
        labirint[x, y] = 1
    return labirint

def postavi_posebne_tocke(labirint, broj, vrijednost):
    visina, sirina = labirint.shape
    tocke = []  # lista da spremam koordinate posebnih točaka (start, izlaz, ključ itd.)
    while len(tocke) < broj:
        x, y = np.random.randint(0, visina), np.random.randint(0, sirina)
        if labirint[x, y] == 0:
            tocke.append((x, y)) # koordinate dajemo listi
            labirint[x, y] = vrijednost
    return tocke

def postavi_kljuc_kraj_cudovista(labirint, cudovista, vrijednost_kljuce=5):
    visina, sirina = labirint.shape
    for cudoviste in cudovista:
        cx, cy = cudoviste
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # gledam susjedne ćelije oko trenutne
            kx, ky = cx + dx, cy + dy # susjedne ćelije
            if 0 <= kx < visina and 0 <= ky < sirina and labirint[kx, ky] == 0:  # je li unutar labirinta
                labirint[kx, ky] = vrijednost_kljuce
                return (kx, ky)
    return None

def generiraj_labirint(visina, sirina):
    labirint = generiraj_prazan_labirint(visina, sirina)
    labirint = dodaj_zidove_labirintu(labirint)

    start = postavi_posebne_tocke(labirint, 1, -2)[0]
    kraj = postavi_posebne_tocke(labirint, 1, -3)[0]
    zakljucana_vrata = (kraj[0], kraj[1] - 1) if kraj[1] > 0 else (kraj[0], kraj[1] + 1) # ako mi je kraj(1) to znači da je granica labirinta te da stavim na drugu stranu
    labirint[zakljucana_vrata[0], zakljucana_vrata[1]] = 2
    mac = postavi_posebne_tocke(labirint, 1, 4)[0]
    broj_cudovista = max(1, (visina * sirina) // 20 // 10)
    cudovista = postavi_posebne_tocke(labirint, broj_cudovista, 3)
    kljuc = postavi_kljuc_kraj_cudovista(labirint, cudovista)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # zidovi oko izlaza da mogu samo kroz vrata proći
        ex, ey = kraj[0] + dx, kraj[1] + dy
        if 0 <= ex < visina and 0 <= ey < sirina:
            if (ex, ey) != zakljucana_vrata:
                labirint[ex, ey] = 1

    return labirint, start, kraj, zakljucana_vrata, cudovista, mac, kljuc

def heuristika(a, b): # pretraživanje puta A*
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) # Manhattan distanca udaljenost dviju točaka

def a_star_korak(labirint, pocetak, cilj, ima_mac=False, ima_kljuc=False):
    visina, sirina = labirint.shape
    smjerovi = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    otvoreni = []
    heappush(otvoreni, (0 + heuristika(pocetak, cilj), 0, pocetak)) # chat rekao ovako
    dosao_od = {}
    trosak_dosad = {pocetak: 0}
    posjeceni = set()

    while otvoreni:
        _, trosak, trenutni = heappop(otvoreni)

        if trenutni in posjeceni:
            continue
        posjeceni.add(trenutni)

        if trenutni == cilj:
            put = []
            while trenutni in dosao_od:
                put.append(trenutni)
                trenutni = dosao_od[trenutni]
            put.reverse()
            return True, put

        for dx, dy in smjerovi: # hvatam susjede
            susjed = (trenutni[0] + dx, trenutni[1] + dy)
            if 0 <= susjed[0] < visina and 0 <= susjed[1] < sirina: # unutar matrice?
                celija = labirint[susjed[0], susjed[1]]

                if celija == 1:
                    continue
                if celija == 3 and not ima_mac:
                    continue
                if celija == 2 and not ima_kljuc:
                    continue

                novi_trosak = trosak + 1  
                if susjed not in trosak_dosad or novi_trosak < trosak_dosad[susjed]:
                    trosak_dosad[susjed] = novi_trosak
                    prioritet = novi_trosak + heuristika(susjed, cilj)
                    heappush(otvoreni, (prioritet, novi_trosak, susjed))
                    dosao_od[susjed] = trenutni

    return False, []

def rijesi_labirint(labirint, start, mac, cudovista, kljuc, vrata, kraj):
    put = []
    posjeceni = set()

    rijeseno, dio_puta = a_star_korak(labirint, start, mac)
    if not rijeseno:
        return False, []
    put.extend(dio_puta)
    posjeceni.update(dio_puta)

    najblize_cudoviste = None # maknut možda
    min_udaljenost = float('inf')
    for cudoviste in cudovista:
        dist = heuristika(put[-1], cudoviste)
        if dist < min_udaljenost:
            min_udaljenost = dist
            najblize_cudoviste = cudoviste

    if najblize_cudoviste:
        rijeseno, dio_puta = a_star_korak(labirint, put[-1], najblize_cudoviste, ima_mac=True)
        if not rijeseno:
            return False, []
        dio_puta = [p for p in dio_puta if p not in posjeceni]
        put.extend(dio_puta)
        posjeceni.update(dio_puta)

    rijeseno, dio_puta = a_star_korak(labirint, put[-1], kljuc, ima_mac=True)
    if not rijeseno:
        return False, []
    dio_puta = [p for p in dio_puta if p not in posjeceni]
    put.extend(dio_puta)
    posjeceni.update(dio_puta)

    rijeseno, dio_puta = a_star_korak(labirint, put[-1], vrata, ima_kljuc=True)
    if not rijeseno:
        return False, []
    dio_puta = [p for p in dio_puta if p not in posjeceni]
    put.extend(dio_puta)
    posjeceni.update(dio_puta)

    rijeseno, dio_puta = a_star_korak(labirint, put[-1], kraj, ima_kljuc=True)
    if not rijeseno:
        return False, []
    dio_puta = [p for p in dio_puta if p not in posjeceni]
    put.extend(dio_puta)

    return True, put

def prikazi_labirint_s_putem(labirint, start, kraj, put):
    visina, sirina = labirint.shape
    plt.figure(figsize=(sirina // 2, visina // 2)) # dimenzije slike
    ax = plt.gca() # trenutna osovina chat dao
    
    for i in range(visina):
        for j in range(sirina):
            if labirint[i, j] == 1:
                plt.text(j, i, "█", ha='center', va='center', color='black', fontsize=7)
            elif (i, j) == start:
                plt.text(j, i, "START", ha='center', va='center', color='green', fontsize=8)
            elif (i, j) == kraj:
                plt.text(j, i, "KRAJ", ha='center', va='center', color='red', fontsize=7)
            elif labirint[i, j] == 2:
                plt.text(j, i, "VRATA", ha='center', va='center', color='blue', fontsize=7)
            elif labirint[i, j] == 3:
                plt.text(j, i, "CUDOVISTE", ha='center', va='center', color='purple', fontsize=8)
            elif labirint[i, j] == 4:
                plt.text(j, i, "MAC", ha='center', va='center', color='orange', fontsize=8)
            elif labirint[i, j] == 5:
                plt.text(j, i, "KLJUC", ha='center', va='center', color='grey', fontsize=8)
    
    for x, y in put:
        plt.text(y, x, "·", ha='center', va='center', color='red', fontsize=12)

    ax.set_xlim(-1, sirina) # chat dao za osi i prikaz
    ax.set_ylim(-1, visina)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    visina = int(input("Unesite visinu labirinta: "))
    sirina = int(input("Unesite širinu labirinta: "))
    labirint, start, kraj, vrata, cudovista, mac, kljuc = generiraj_labirint(visina, sirina)

    rijeseno, put = rijesi_labirint(labirint, start, mac, cudovista, kljuc, vrata, kraj)

    if rijeseno:
        prikazi_labirint_s_putem(labirint, start, kraj, put)
    else:
        print("Labirint nije moguće riješiti.")
