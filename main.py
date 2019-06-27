import utils as utl
import pso

data = utl.get_data(file_path="assets/sprint7ToroideMixto.csv")
pso = pso.ClusteringPSO(data, 50, max_iter=100)
print(pso.search)
