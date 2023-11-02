import xml.etree.ElementTree as ET #eisagei th elementTree monada apo to xml.etree paketo.Dinei eykolo tropo analisis dedomenwn XML.
import pprint #eisagei thn omada poy ektipwnei
import math #eisagei thn monada poy exei diafores mathimatikes sinartiseis kai statheres
from mpl_toolkits import mplot3d #parexei leitoyrgia 3d sxediasis apo to matplotlib
from matplotlib import pyplot as plt #parexei diepafh san to Matlab gia th dimioyrgia apeikoniewn
import numpy as np #bibliothiki me pseydomino np gia arithmitikoys ypologismoys
import sys #einai leitoyrgikh monada gia prosvasi se parametroys kai leitoyegies poy aforoyn thn python

#gia na ginei epilogi ektipwshs sxetika me toys pinakes nympy
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf)


def read_data(file_path: str) -> dict:
    data = {}  # arxikopoihsh enos kenoy 'data' gia na mporesoyme na apothikeysoyme ta dedomena
    namespace = "{http://www.opengis.net/kml/2.2}"  # kathorismos xml namespace to opoio xrisimopoieitai sto arxeio toy kml
    tree = ET.parse(file_path)  # analiei to arxeio xml
    root = tree.getroot()  # pairnei to riziko stoxeio toy dendroy xml
    placemarks = root.findall(f".//{namespace}Placemark")  # briskei ola ta stoixeia poy einai simademena ws kadoi sto xml

    for placemark in placemarks:  # kanei epanalipsi se kathe kado poy einai pinned
        name = placemark.find(f".//{namespace}name").text  # dinetai to onoma toy kadoy
        coordinates = placemark.find(f".//{namespace}coordinates").text  # dinontai oi sintetasgmenes toy kadoy
        coordinate = [float(coord) for coord in
                      coordinates.split(",")[:2]]  # metatropi twn sintetagmenwn se floats kai apothikeyei tis 2 prwtes times
        data[name] = coordinate  # prosthiki toy onomatos kai ton sintetagmenwn sto leksiko dedomenwn

    return data  # kanei return to simpliromeno leksiko dedomenwn poy onomasame data panw


def distance(cords1: list, cords2: list): #ayth h synarthsh ypologizei thn apostasti metaksi 2 sintetagmenwn
    d = (math.sqrt((cords2[0] - cords1[0]) ** 2 + (cords2[1] - cords1[1]) ** 2)) #ypologizei thn apostash me ton tipo ths apostasis. Aytos o tipos dinei thn eytheia apostasi metaksi twn shmeiwn
    return round(d, 7) #stroggilopoieitai h apostasi se 7 dekadika psifia. Ayto ginetai gia na mhn yparxoyn lathi akriveias


def plot_map(points: dict): #Ayth h synarthsh sxediazei ta shmeia ston xarth plot
    x = [points[point][0] for point in points] #bazei tis sintetagmenes x twn pinned shmeiwn apo ta data
    y = [points[point][1] for point in points] #bazei tis sintetagmenes y twn pinned shmeiwn apo ta data

    # sindeei ta points
    plt.plot(x, y, 'bo-')

    # prosthetei sta shmeia ta onomata toys dhladh toys arithmoys sto plot diagram
    for point, (x_val, y_val) in zip(points.keys(), zip(x, y)):
        plt.text(x_val, y_val, point, ha='center', va='bottom')

    # Prosthetei titlo panw apo plot diagram
    plt.title("Bins Agios Athanasios, Kavala")

    plt.show() #Gia na fanei to plot

def make_distance_array(data: list): #ayth h sinartisi ypologizei ton pinaka apostasis metaksi 2 simeiwn
    num_points = len(data) #lamvanei ton arithmo twn shmeiwn
    distance_matrix = np.zeros((num_points, num_points)) #arxikoopoiei enan pinaka gia na apothikeysei tis apostaseis

    for i, point1 in enumerate(data): #epanalipsi se kathe simeio

        for j, point2 in enumerate(data): #Sigkrinetai kathe shmeio me kathe allo shmeio
            if i == j:
                continue #gia na deiksei oti an ta shmeia einai idia paraleipete

            distance_matrix[i][j] = distance(data[point1], data[point2])
            # ipologizei thn apostasi metaksi toy shmeioy 1 kai toy simeioy 2 xrhsimopoiontas th sinartisi ths apostashs
    np.fill_diagonal(distance_matrix, 0) #midenizei ta diagwnia stoixeia toy pinaka
    return distance_matrix #epistrefei ton ypologismeno pinaka apostashs


def ac(data: list, #h sinartisi ayth ektelnei ton algori8mo ant colony optimization
       n_ants: int = 30,
       n_iterations=100,
       decay: float = 0.5,
       alpha: int = 1,
       beta: int = 2,
       ):
    point_names = [i for i in data] #dimioyrgei lista me onomata apo ta pinned bins ta opoia exoyn mpei sto data
    print(data) #ektipwnei to data gia na doyme an exei sfalma

    distances = make_distance_array(data) #ipologizei ton pinaka apostasis xrhsimopoiontas th sinartisi make distance array

    pheromone = np.ones((len(data), len(data))) #arxikopoihsh feromonhs

    best_path = None #metabliti gia thn apothikeusi ths kalyterhs diadromis
    best_distance = np.inf #metabliti gia thn apothikeisi ths kalyterhs diadromis me arxikopoihsh sto apeiro

        # to loop gia ta iterations
    for iteration in range(n_iterations):
        # Arxikopoioume ta monopatia twn mirmigkiwn kai tis apostaseis
        ant_paths = np.zeros((n_ants, len(distances)), dtype=int)
        ant_distances = np.zeros(n_ants)

        # to loop gia thn kinhsh twn mirmigkiwn
        for ant in range(n_ants):
            current_node = np.random.randint(len(distances))  # topoteteitai to mirmigki se tyxaia thesi

            visited = [current_node] #lista gia na blepoyme ta bins poy exoyme paei
            # gia ola ta points
            for i in range(len(distances) - 1):
                unvisited = list(
                    set(range(len(distances))) - set(visited))  # pairnoume tin lista twn shmeiwn poy den exoyme paei
                pheromone_values = np.power(pheromone[current_node, unvisited],
                                            alpha)  # pairnoume tis times twn pheromones gia ta shmeia poy den exoyme paei
                distance_values = np.power(1.0 / distances[current_node, unvisited],
                                           beta)  # pairnoume tis apostaseis gia shmeia poy den exoyme paei
                probabilities = pheromone_values * distance_values / np.sum(
                    pheromone_values * distance_values)  # pairnoume tis pithanotites gia to epomeno node
                next_node = np.random.choice(unvisited, p=probabilities) #epilogh epomenou merous me vash tis pithanotites
                visited.append(next_node) #prosthiki epomenoy merous sthn lista me ta merh
                current_node = next_node #enhmerwsh meroys sto opoio eimaste

            ant_paths[ant] = visited #apothikeysi twn merwn poy exoyn paei ta mirmigia ws diadromh
            ant_distances[ant] += distances[visited[-1], visited[0]] # prosthiki apostasis apo to teleytaio sto prwto meros
            #kanei calculate thn telikh apostash poy ekanan ta mirmigkia
            for i in range(len(visited) - 1):
                ant_distances[ant] += distances[visited[i], visited[i + 1]]

            #enhmerwsh feromonhs me vash to monopati twn mirmigkiwn
        delta_pheromone = np.zeros(pheromone.shape)
        for ant in range(n_ants):
            for i in range(len(distances) - 1):
                delta_pheromone[ant_paths[ant, i], ant_paths[ant, i + 1]] += 1.0 / ant_distances[ant]
            delta_pheromone[ant_paths[ant, -1], ant_paths[ant, 0]] += 1.0 / ant_distances[ant]

        pheromone = (1.0 - decay) * pheromone + delta_pheromone

        #elegxei an yparxei kalytero monopati
        if ant_distances.min() < best_distance:
            best_path = ant_paths[ant_distances.argmin()].copy()
            best_distance = ant_distances.min()

        print('iteration {} : {}'.format(iteration, best_distance))

    # Return the best path and distance
    best_path = [point_names[i] for i in best_path]
    return (best_path, best_distance)


data = read_data("AgiosAthanasios.kml") #diavazei to arxeio kml me tis topothesies
plot_map(data) #kalei th sinartisi kai metabibazei to data gia na doyme ta sotieia sto plot diagram
print(ac(data, n_ants=50, n_iterations=200)) #dhlwnei to apotelesma kai ginetai dilwsh twn poswn iterations theloyme na kanei