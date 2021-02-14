# Run this again if update results in a pakage update
using Markdown, Images, CUDA, Flux, Random, Distributions, JLD, ColorTypes, FixedPointNumbers, Plots, StructArrays, ProgressBars, GPUArrays, LinearAlgebra

allowscalar(false)
mutable struct Synapse
	w::Float32 # Synapse weight
    δw::Float32 # weight learning rate (relation to w is transposed)
    α::Bool # Synapse alive or not
	function Synapse(w::Float32 = Float32(rand(truncated(Normal(0, 0.5), -1, 1))), δw = Float32(rand(truncated(Normal(0.0, 0.5), -1, 1))), α::Bool = rand(Binomial(1, 0.02)))
        this = new()
        this.w, this.δw, this.α = w, δw, α
        return this
    end
end

mutable struct Neuron
	accumulator::Int
	cycle_timer::Int
    b::Float32 # Neuron bias
    δb::Float32 # bias learing rate
end

mutable struct Network
    neurons::StructVector{Neuron}
    synapses::StructArray{Synapse}
    inputNeurons::Array{Int,1}
    outputNeurons::Array{Int,1}
    size::Int
    connectivity::AbstractFloat
    selfConnectivity::AbstractFloat
    function Network(size::Int = 10, connectivity::AbstractFloat = 0.1, selfConnectivity::AbstractFloat = 0.1)
        this::Network = new()
        this.neurons = StructArray{Neuron}(
            (
                (
                    zeros(Int, n_size), # accumulator
                    zeros(Int, n_size), # cycle_timer
                    Float32.(rand(truncated(Normal(0.0, 0.5), -1, 1), (n_size,))), # b
                    Float32.(rand(truncated(Normal(0.0, 0.001), 0, 1), (n_size,))) # δb
                )
            )
        )
        this.synapses = StructArray{Synapse}(
            (
                (
                    Float32.(rand(truncated(Normal(0.0, 0.5), -1, 1), (n_size, n_size))), # w
                    Float32.(rand(truncated(Normal(0, 0.1), 0, 1), (n_size, n_size))), # δw
                    Bool.(rand(Binomial(1, connectivity), (n_size, n_size)))
                )
            )
        )
        if selfConnectivity != connectivity
            for i in 1:size
                this.synapses[i,i].w *= rand(Binomial(1, selfConnectivity / connectivity))
            end
        end
        this.inputNeurons = zeros(Int, 0)
        this.outputNeurons = zeros(Int, 0)
        this.size = size
        this.connectivity = connectivity
        this.selfConnectivity = selfConnectivity
        this
    end
end

function loadData(dir::String, folder::String)::Array
    retval = []
    cd(dir)
    if !isfile(folder * ".jld")
        # loading from folder
		if isempty(retval)
			for (root, dirs, files) in walkdir(folder)
				for file in files
                    push!(retval, load(joinpath(root, file))) # path to files
                    # println(file)
				end
			end
		end
		@timev begin
			jldopen(folder * ".jld", "w") do file
				addrequire(file, ColorTypes)
				addrequire(file, FixedPointNumbers)
				addrequire(file, Core)
				write(file, folder, retval)
			end
		end
	else isfile(folder * ".jld")
        # loading from cached binary
        println("loading "*folder*" from cached binary")
		jldopen(folder * ".jld", "r") do file
			if isempty(retval)
				append!(retval, read(file, folder))
			end
		end
	end
    return retval
end

# supervised
# TODO: create supervised method of propagation
# unsupervised
function propagate!(network::Network, χ::Array{Float16}, outputLength::Integer, batchSize::Integer = 1, ϵ::AbstractFloat = 0.01)
	# Initialize propagation variables
	local inputLength = Base.length(χ[:,:,1])
    local inputDims = Base.size(χ[:,:,1])
    if length(network.inputNeurons) < inputLength
        append!(network.inputNeurons, sample(filter(i -> (!(i in network.inputNeurons) && !(i in network.outputNeurons)), 1:network.size), inputLength - length(network.inputNeurons), replace = false))
        println("current network input length is smaller than input length")
    elseif length(network.inputNeurons) > inputLength
        for i in 1:(length(network.inputNeurons) - inputLength) pop!(network.inputNeurons) end
        println("current network input length is larger than input length")
    end
    if length(network.outputNeurons) < outputLength
        append!(network.outputNeurons, sample(filter(i -> (!(i in network.inputNeurons) && !(i in network.outputNeurons)), 1:network.size), outputLength - length(network.outputNeurons), replace = false))
        println("current network output length is smaller than output length")
    elseif length(network.outputNeurons) > outputLength
        for i in 1:(length(network.outputNeurons) - outputLength) pop!(network.outputNeurons) end
        println("current network output length is larger than output length")
    end
    local out = Array{Array{Float32,1},1}()
    local ∑w = Array{Array{Float16,2},1}()
    local ∑b = Array{Array{Float16,1},1}()
    # multiply the input with the input neurons' weights
    local gpuSynapses::StructArray{Synapse} = replace_storage(CuArray, network.synapses)
    local gpuNeurons::StructVector{Neuron} = replace_storage(CuVector, network.neurons)
    local gpuχ = CuArray(χ)
    print(string("batch_size: $batchSize\ninput_number: ", size(gpuχ)[3], "\ninput_length: $inputLength\ninput_dims: ", inputDims, "\noutput_size: $outputLength\n"))
    local it = 1
    local signal::typeof(view(gpuχ[:,:,1], :)) = view(CUDA.zeros(inputDims), :)
    for xBatch in tqdm(1:batchSize:size(gpuχ)[3])
        # TODO: implement batch parallelization
        local ∇w::CuArray{Float32,2} = CUDA.zeros(Float32, (network.size, network.size))
        local ∇b::CuArray{Float32,1} = CUDA.zeros(Float32, (network.size))
        for x in xBatch:min(xBatch + batchSize - 1, size(gpuχ)[3])
            it = 1
            # set the first signal to the input vector
            # TODO: implement combination of new input with recurring signals from previous inputs
            @inbounds signal = view(gpuχ[:,:,x], :)
            local srcNeurons = network.inputNeurons
            # then do for the rest of the layers until a sygnal reaches the output
            while isempty(filter(el -> (el in network.outputNeurons), srcNeurons))
                @inbounds ∇w[srcNeurons, :] = σ.(∇w[srcNeurons, :] + (gpuSynapses.δw[srcNeurons, :] .* signal))
                @inbounds ∇b[srcNeurons] = σ.(∇b[srcNeurons] + (gpuNeurons.δb[srcNeurons] .* signal))
                @inbounds signal = (gpuSynapses.α[:, srcNeurons] .* gpuSynapses.w[:, srcNeurons]) * signal
                @inbounds srcNeurons = Array{Int,1}(findall(el -> el, .!CUDA.isapprox.(signal, 0.0)))
                @inbounds signal = σ.(signal[srcNeurons] + gpuNeurons.b[srcNeurons])
                it += 1
            end
            # do once for signal of output layer
            @inbounds ∇w[srcNeurons, :] = σ.(∇w[srcNeurons, :] + (gpuSynapses.δw[srcNeurons, :] .* signal))
            @inbounds ∇b[srcNeurons] = σ.(∇b[srcNeurons] + (gpuNeurons.δb[srcNeurons] .* signal))
            @inbounds signal = (gpuSynapses.α[:, srcNeurons] .* gpuSynapses.w[:, srcNeurons]) * signal
            @inbounds srcNeurons = Array{Int,1}(CUDA.findall(el -> el, .!CUDA.isapprox.(signal, 0.0)))
            @inbounds signal = σ.(signal[srcNeurons] + gpuNeurons.b[srcNeurons])
            @inbounds local _out = Base.zeros(outputLength)
            @inbounds _out[CUDA.findall(el -> (el in srcNeurons), network.outputNeurons)] = Array(signal)[Base.filter(el -> (el in network.outputNeurons), srcNeurons)]
            @inbounds push!(out, _out)
            
            # println("signal#$x: $it iterations\n")
        end
        # TODO:update active synapses matrix α
        # track weight changes
        @inbounds push!(∑w, Array(Float16.(∇w)))
        @inbounds push!(∑b, Array(Float16.(∇b)))
        # update weights & biases
        gpuSynapses.w[CUDA.findall(i -> i, gpuSynapses.α)] = σ.(gpuSynapses.w .+ transpose(∇w .* ϵ))[CUDA.findall(i -> i, gpuSynapses.α)]
        gpuNeurons.b[:] = σ.(gpuNeurons.b .+ (∇b .* ϵ))
    end
    network.synapses = replace_storage(Array, gpuSynapses)
    network.neurons = replace_storage(Array, gpuNeurons)
    # GC.gc()
    # CUDA.reclaim()
	return out, it, ∑w, ∑b
end

# load data
dogIm = Array{Array{Gray{Normed{UInt8,8}},2},1}(loadData("c:/Users/ahmad/OneDrive/Documents/GitHub/open-ended-neural-nets/DFA_BP/data/kagglecatsanddogs_3367a/processed/", "Dog"))
catIm = Array{Array{Gray{Normed{UInt8,8}},2},1}(loadData("c:/Users/ahmad/OneDrive/Documents/GitHub/open-ended-neural-nets/DFA_BP/data/kagglecatsanddogs_3367a/processed/", "Cat"))

# merge input into one 3d array
images = Float16.(reshape(reduce(hcat, [catIm;dogIm]), tuple(size(catIm[1])..., length([catIm; dogIm]))))
# when to fire the correct output signal?
# when any forward signal reaches any neuron neighbouring any of the correct_output neurons (neigbour meaning connected by a synapse not including itself)
# specify network size
n_size = length(dogIm[1]) + 64 + 128 + 16 + 64 + 64 # network size with pytorch
# Initialize network instance
network = Network(n_size, 0.0218, 0.0001)
og_network = network
# Decay Rate of synapse viability
decay_rate = 0.5
# Growth Rate of synapse viability when a signal is fired through the synapse
growth_rate = 2.0 * decay_rate
# analyze network
sum(Int.(network.synapses.α))
heatmap(network.synapses.α)
heatmap(network.synapses.w .* network.synapses.α)
maximum(network.synapses.w .* network.synapses.α)
minimum((network.synapses.w .* network.synapses.α)[network.synapses.w .* network.synapses.α .!= 0.0])
median((network.synapses.w .* network.synapses.α)[network.synapses.w .* network.synapses.α .!= 0.0])
mean((network.synapses.w .* network.synapses.α)[network.synapses.w .* network.synapses.α .!= 0.0])

idx = sample(1:size(images)[3], 10000, replace = false)
samp = images[:,:, idx]
result = propagate!(network, samp, 2, 100, 0.1)

GC.gc()
