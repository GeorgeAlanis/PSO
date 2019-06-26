import utils as utl
import pso

data = utl.getData(file_path="assets/sprint7ToroideMixto.csv")
pso = pso.ClusteringPSO(data, 30, max_iter=100)
print(pso.search)
