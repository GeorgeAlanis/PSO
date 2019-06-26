import utils as utl
import pso

data = utl.getData(file_path="assets/sprint7ToroideMixto.csv")
pso = pso.PSO(data, 10, max_iter=100, use_var=True)
print(pso.search)
