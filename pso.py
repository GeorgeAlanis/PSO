import numpy as np
import utils
import classes
from copy import deepcopy
from sklearn.cluster import KMeans
from tqdm import tqdm


class Particle:
    def __init__(self, clusters, elements, fitness_by='variance'):
        self.number_of_clusters = len(clusters)
        self.fitness = float("-inf")
        self.centroids = np.zeros([self.number_of_clusters, 2])
        self.areas = np.zeros(self.number_of_clusters)
        self.store_times = np.zeros(self.number_of_clusters)
        self.clusters = deepcopy(clusters)
        self.elements = elements.copy()
        self.number_of_elements = len(self.elements)
        self.vel = np.zeros([self.number_of_clusters, 2])
        self.fitness_by = fitness_by

        for i in xrange(self.number_of_clusters):
            self.centroids[i] = clusters[i].centroid
            self.store_times[i] = clusters[i].total_time
            self.areas[i] = clusters[i].area

        self.calculate_fitness()
        self.best_fitness = self.fitness
        self.best_centroids = self.centroids.copy()

    def calculate_fitness(self):
        if self.fitness_by == 'variance':
            self.fitness = -1.0 * np.var(self.store_times)

    def move(self, w, c1, c2, global_best_centroids):
        r1, r2 = np.random.ranf(size=2)

        for i in xrange(self.number_of_clusters):
            self.vel[i] = (w * self.vel[i]) + (c1 * r1 * (utils.euclidean(self.best_centroids[i], self.centroids[i])))\
                          + (c2 * r2 * (utils.euclidean(global_best_centroids[i], self.centroids[i])))
            self.centroids[i] += self.vel[i]

        self.recluster()
        self.calculate_fitness()

        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_centroids = self.centroids.copy()

    def recluster(self):
        for i in xrange(self.number_of_elements):
            smallest_distance = [float('inf'), None]

            for j in xrange(self.number_of_clusters):
                current_distance = utils.euclidean(self.elements[i], self.clusters[j].centroid)

                if current_distance < smallest_distance[0]:
                    smallest_distance = [current_distance, j]

            if smallest_distance[1] != self.elements[i][-1]:
                self.clusters[self.elements[i][-1]].remove(self.elements[i])
                self.clusters[j].push_back(self.elements[i])


class ClusteringPSO:
    def __init__(self, data, n_particles, n_clusters=13, seed=42, max_iter=10, w=1, c1=.5, c2=.5, fit_by='variance'):
        self.data = data.copy()
        self.number_of_elements = len(self.data)
        self.number_of_particles = n_particles
        self.number_of_clusters = n_clusters
        self.seed = np.random.RandomState(seed)
        self.max_iterations = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fit_by = fit_by
        self.population = []
        self.global_best_centroids = np.zeros([self.number_of_clusters,2])
        self.global_best_fitness = float("-inf")

        self.generate_population()
        self.find_best()

    def generate_population(self):
        for _ in tqdm(xrange(self.number_of_particles), desc='Generando Poblacion inicial', unit=' Particle'):
            elements = self.data.copy()
            cluster = KMeans(n_clusters=self.number_of_clusters, random_state=self.seed)
            cluster_labels = cluster.fit_predict(self.data[:, 0:2])
            clusters = [classes.ClusterPdv(center, 0.0, 0.0) for center in cluster.cluster_centers_]

            for i in xrange(self.number_of_elements):
                elements[i][-1] = cluster_labels[i]
                clusters[cluster_labels[i]].push_back(elements[i])

            self.population.append(Particle(clusters, elements, self.fit_by))

    def find_best(self):
        for i in xrange(self.number_of_particles):
            if self.population[i].fitness > self.global_best_fitness:
                self.global_best_fitness = self.population[i].fitness
                self.global_best_centroids = self.population[i].centroids.copy()

    @property
    def search(self):
        best_particle = None
        with tqdm(total=self.max_iterations, unit=' Epoch') as pb:
            pb.set_postfix(fitness=self.global_best_fitness, refresh=True)
            for _ in xrange(self.max_iterations):
                for particle in self.population:
                    particle.move(self.w, self.c1, self.c2, self.global_best_centroids)
                    if particle.fitness > self.global_best_fitness:
                        best_particle = deepcopy(particle)
                        self.global_best_fitness = particle.fitness
                        self.global_best_centroids = particle.centroids.copy()
                        pb.set_postfix(fitness=self.Gb_fit, refresh=True)
                pb.update()
            return best_particle
