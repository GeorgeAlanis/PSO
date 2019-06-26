class ClusterPdv:
    def __init__(self, centroid, total_time, area):
        self.centroid = centroid.copy()
        self.total_time = total_time
        self.area = area

    def push_back(self, pdv):
        self.total_time += pdv[2]

    def remove(self, pdv):
        self.total_time -= pdv[2]
