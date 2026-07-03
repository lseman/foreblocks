using JuMP, Gurobi, TickTock, DataFrames
using LinearAlgebra;

using Cbc

const GRB_ENV = Gurobi.Env()

atitude = "CLASSICA"

include("$(T)_$(jobs).jl")

#  BATTERY DATA 


soc_inicial = 0.7 
limite_inferior = 0.3 
bat_usage = 5
#d = 0.05

q =10



ef = 0.9 
k = 1 
beta = 10 
v_bat = 3.6 

restringe = "987"



server = true


global recursot = recurso_p
global vetor1t = []
global vetor2t = []
global vetor3t = []
global vetor4t = []
global vetor5t = []
global vetor6t = []
global vetor7t = []
global vetor8t = []
global vetor9t = []
global vetor10t = []
global vetor11t = []
global vetor12t = []
global vetor13t = []
global vetor14t = []
global vetor15t = []
global vetor16t = []
global vetor17t = []
global vetor18t = []

global orbita = 1
global soc_inicial1 = 0.7 # [%] de carga (0-1)
global soc_total = []



global z = 1

global uy = 1

global os_equal = 0

global iterations_limit = 30000

global node_depth_limit = round(Int, T * 1.4)



d = 1
ub = 0
global iterations = 0
global node_cut = 0
global g_lower_bound = 33

global time_hist = []
global gap_hist = []
global ub_hist = []
global lb_hist = []

global global_optimal = []
global todas_exploracoes = []
global os_rodas = []
global os_rodas_2 = []
global os_rodas_finalizados = []
global return_0 = false
global return_1 = true 
global upper_bound = 0
global cortes = []
global g_optimal = 0

for m in 1:node_depth_limit
    push!(os_rodas, 0)
end 

for m in 1:node_depth_limit
    push!(os_rodas_2, 0)
end 

for p in 1:node_depth_limit
    push!(os_rodas_finalizados, 0)
end 

global tree_hist = []
start = [1]



