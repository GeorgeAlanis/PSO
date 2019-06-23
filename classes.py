import copy


class PdV:
    def __init__(self, x, y, time_store):
        self.x = x
        self.y = y
        self.time_store = time_store

    def __repr__(self):
        return "[%f, %f %.2f]\n" % (self.x, self.y, self.time_store)


class ClusterPdV:
    def __init__(self, pdvs, centroid=None):
        self.centroid = copy.deepcopy(centroid) or [None, None]
        self._elements = copy.deepcopy(pdvs)
        self.total_time = 0

        for i in pdvs:
            self.total_time += i.time_store

    def __repr__(self):
        return "[" + ", ".join([str(i) for i in self._elements]) + "]"

    def push_back(self, pdv):
        self.total_time += pdv.time_store
        self._elements.append(pdv)

    def remove(self, pdv):
        self.total_time -= pdv.time_store
        self._elements.remove(pdv)

    def size(self):
        return len(self._elements)

    def set_center(self, new):
        if isinstance(new, list) and len(new) == 2:
            self.centroid = new
        else:
            raise ValueError("New center must be a list of length 2.")
