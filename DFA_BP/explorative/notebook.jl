# Run this again if update results in a pakage update
begin
	using Markdown
	using InteractiveUtils
	using Base
	using Images
	using CUDA
	using Flux: gradient
	using Metalhead
	using Random, Distributions
	using PlutoUI
	using JLD
	using ColorTypes
	using FixedPointNumbers
	using Plots
	using PrettyTables
end

# load from here if data has been precompiled in data.jld
function loadData(a::T, b::T) where T
	if !isfile("DFA_BP/data/data.jld")
		cd("DFA_BP/data/kagglecatsanddogs_3367a/processed")
		if isempty(a)
			for (root, dirs, files) in walkdir("DFA_BP/data/kagglecatsanddogs_3367a/processed/Cat")
				for file in files
					push!(a, load(joinpath(root, file))) # path to files
				end
			end
		end
		if isempty(b)
			for (root, dirs, files) in walkdir("DFA_BP/data/kagglecatsanddogs_3367a/processed/Dog")
				for file in files
					push!(b, load(joinpath(root, file))) # path to files
				end
			end
		end
		@timev begin
			cd("DFA_BP/data/")
			jldopen("data.jld", "w") do file
				addrequire(file, ColorTypes)
				addrequire(file, FixedPointNumbers)
				addrequire(file, Core)
				write(file, "catIm", a)
				write(file, "dogIm", b)
			end
		end
	end
	if isfile("DFA_BP/data/data.jld")
		jldopen("DFA_BP/data/data.jld", "r") do file
			if isempty(a)
				append!(a, read(file, "catIm"))
			end
			if isempty(b)
				append!(b, read(file, "dogIm"))
			end
		end
	end
end

mutable struct synapse
	w::AbstractFloat
	viability::AbstractFloat
	s_viability::AbstractFloat
	synapse(w::AbstractFloat = rand(truncated(Normal(0, 0.5), -1, 1), 1)[1], viability::AbstractFloat = rand(truncated(Normal(0, 0.5), -1, 1), 1)[1], s_viability::AbstractFloat = rand(truncated(Normal(0.0, 0.5), -1, 1), 1)[1]) = new(w, viability, s_viability)
end

mutable struct neuron
	accumulator::Int
	cycle_timer::Int
end

# load data
dogIm = Array{Array{Gray{Normed{UInt8,8}},2},1}()
catIm = Array{Array{Gray{Normed{UInt8,8}},2},1}()
@timev loadData(catIm, dogIm)
# when to fire the correct output signal?
# when any forward signal reaches any neuron neighbouring any of the correct_output neurons (neigbour meaning connected by a synapse not including itself)

input_size = 50 * 50
n_size = input_size + 64 + 128 + 16 + 64 + 64 # network size with pytorch
network_synapses = [synapse() for i = 1:n_size, j = 1:n_size]
# Overall synapse connectivity ratio
overall_connectivity_ratio = count(i -> (i.viability >= i.s_viability), network_synapses) / count(i -> true, network_synapses)
# Synapse self connection ratio
self_connectivity_ratio = count(i -> (i.viability >= i.s_viability), [network_synapses[i,i] for i = 1:n_size]) / count(i -> true, [network_synapses[i,i] for i = 1:n_size])

synapse_state = [(if ((network_synapses[i,j].viability - network_synapses[i,j].s_viability) > 0.0) 1.0 else 0.0 end) for i in 1:n_size, j in 1:n_size]
self_connectivity = permutedims([(if ((network_synapses[i,i].viability - network_synapses[i,i].s_viability) > 0.0) 1.0 else 0.0 end) for i in 1:n_size])

heatmap(synapse_state);
heatmap(self_connectivity);
typeof(network_synapses)
function propagate(synapses::T, input::AT) where {T <: Array{synapse,2},AT <: AbstractArray}
	# Initialize propagation variables
	local input_size = length(input[1])
	local n_size = size(synapses, 1)
	input_indicies = sample(1:n_size, input_size, replace = false)
	correct_output_indicies = sample(filter(i -> !(i in input_indicies), 1:n_size), 10, replace = false)

	return input_indicies
end

propagate(network_synapses, catIm)
