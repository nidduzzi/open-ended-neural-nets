# Run this again if update results in a pakage update
using Markdown, Images, CUDA, Flux, Random, Distributions, JLD, ColorTypes, FixedPointNumbers, Plots, StructArrays, ProgressBars, GPUArrays, LinearAlgebra, MLDatasets

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
    δb::Float32 # bias learning rate
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
        println("loading " * folder * " from cached binary")
		jldopen(folder * ".jld", "r") do file
			if isempty(retval)
				append!(retval, read(file, folder))
			end
		end
	end
    cd(@__DIR__)
    return retval
end

# supervised
# TODO: create supervised method of propagation
# unsupervised
function propagate_kernel!()

end
function propagate!(network::Network, χ::Array{Float16,2}, outputLength::Integer, ϵ::AbstractFloat = 0.01)
	# Initialize propagation variables
	local inputLength = Base.length(χ[:,1])
    local inputNum = size(χ)[2]
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
    print(string("number of input: $inputNum\ninput_length: $inputLength\noutput_size: $outputLength\n"))
    local it = 1
    # local signal::typeof(view(gpuχ[:,:,1], :)) = CUDA.zeros(inputLength)
    # local signalSrcNeurons::Union{Array{Int,1},CuArray{Int,1}} = zeros(Int, 0)
    NamedTuple
    local signals::Array{NamedTuple{(:signal, :signalSrcNeurons),Tuple{Union{Array{Float32,1},CuArray{Float32,1}},Union{Array{Int,1},CuArray{Int,1}}}}} = Array{Tuple{Union{Array{Float32,2},CuArray{Float32,2}},Union{Array{Int,1},CuArray{Int,1}}}}()
    for x in tqdm(1:inputNum)
        # initialize weight change matrix and bias change vector for each batch with zeros
        local ∇w::CuArray{Float32,2} = CUDA.zeros(Float32, (network.size, network.size))
        local ∇b::CuArray{Float32,1} = CUDA.zeros(Float32, (network.size))
        local inboundSignalIntegrated::Bool = false
        # TODO: implement pipelining of inputs and immediate updates of weights
        for i in 1:length(signals)
            local signal::Ref{Union{Array{Float32,1},CuArray{Float32,1}}} = Ref{Union{Array{Float32,1},CuArray{Float32,1}}}(signals[i].signal)
            local signalSrcNeurons::Ref{Union{Array{Int,1},CuArray{Int,1}}} = Ref{Union{Array{Int,1},CuArray{Int,1}}}(signals[i].signalSrcNeurons)
            # TODO: implement combination/merger/integration of inbound signal (next input) with current signals that pass through the input neurons
            local inboundSignalSrc = findall(el -> !(el in network.inputNeurons), signalSrcNeurons[])
            if !isempty(inboundSignalSrc)
                # modify signal elements that pass through input neurons with the inbound signals
                @inbounds signal[][inboundSignalSrc] = tanh.(signal[][inboundSignalSrc] .+ gpuχ[findall(el -> (el in signalSrcNeurons[]), network.inputNeurons),x])
                # append the rest of the inbound signal to the current signal
                local newSignalSrcIndex = findall(el -> !(el in signalSrcNeurons[]), network.inputNeurons)
                @inbounds append!(signal[], gpuχ[newSignalSrcIndex,x])
                # append the signalSrcNeurons of the inbound signal to that of the current signal
                @inbounds append!(signalSrcNeurons[], network.inputNeurons[newSignalSrcIndex])
                inboundSignalIntegrated = true
            end
            # TODO: implement weight and bias change array update
            @inbounds ∇w[signalSrcNeurons[], :] = tanh.(∇w[signalSrcNeurons[], :] + (gpuSynapses.δw[signalSrcNeurons[], :] .* signal[]))
            @inbounds ∇b[signalSrcNeurons[]] = tanh.(∇b[signalSrcNeurons[]] + (gpuNeurons.δb[signalSrcNeurons[]] .* signal[]))
            # TODO: update signal
            @inbounds signal[] = (gpuSynapses.α[:, signalSrcNeurons[]] .* gpuSynapses.w[:, signalSrcNeurons[]]) * signal[]
            @inbounds signalSrcNeurons[] = Array{Int,1}(findall(el -> el, .!CUDA.isapprox.(signal[], 0.0)))
            @inbounds signal[] = tanh.(signal[][signalSrcNeurons[]] + gpuNeurons.b[signalSrcNeurons[]])
            #= 
            # TODO: Network needs to know why the output neurons are special
            # * this can be done by specifying a structure in the network explicitly that either calculates the error, or mutual information
            # * or it can be done by processing and mutating the signal passing through the output neurons
            # ** mutating for UNSUPERVISED learning can be done by things like mutual information in Invariant information clustering (Ji, X., Henriques, J.F. & Vedaldi, A., 2019. Invariant Information Clustering for Unsupervised Image Classification and Segmentation. arXiv:1807.06653 [cs]. Available at: http://arxiv.org/abs/1807.06653.)
            # ** mutating for SUPERVISED learning can be done by replacing the orginal signal with the error =#
            # extract outbound signals from output neurons
            @inbounds local outboundSignalSrc = filter(el -> (el in network.outputNeurons), signalSrcNeurons[])
            if !isempty(outboundSignalSrc)
                @inbounds local _out = Base.zeros(outputLength)
                @inbounds _out[CUDA.findall(el -> (el in signalSrcNeurons[]), network.outputNeurons)] = Array(σ(signal[]))[outboundSignalSrc]
                @inbounds push!(out, _out)
            end
        end
        if !inboundSignalIntegrated
            push!(signals, (signal = gpuχ[:,x], signalSrcNeurons = network.inputNeurons))
            # TODO: Implement signal propagation like the loop above
        end
        # isempty(filter(el -> (el in network.outputNeurons), signalSrcNeurons))
        # # do once for signal of output layer
        # @inbounds ∇w[signalSrcNeurons, :] = tanh.(∇w[signalSrcNeurons, :] + (gpuSynapses.δw[signalSrcNeurons, :] .* signal))
        # @inbounds ∇b[signalSrcNeurons] = tanh.(∇b[signalSrcNeurons] + (gpuNeurons.δb[signalSrcNeurons] .* signal))
        # @inbounds signal = (gpuSynapses.α[:, signalSrcNeurons] .* gpuSynapses.w[:, signalSrcNeurons]) * signal
        # @inbounds signalSrcNeurons = Array{Int,1}(CUDA.findall(el -> el, .!CUDA.isapprox.(signal, 0.0)))
        # @inbounds signal = tanh.(signal[signalSrcNeurons] + gpuNeurons.b[signalSrcNeurons])
        
        # TODO:update active synapses matrix α
        # track weight changes
        @inbounds push!(∑w, Array(Float16.(∇w)))
        @inbounds push!(∑b, Array(Float16.(∇b)))
        # update weights & biases
        gpuSynapses.w[CUDA.findall(i -> i, gpuSynapses.α)] = tanh.(gpuSynapses.w .+ transpose(∇w .* ϵ))[CUDA.findall(i -> i, gpuSynapses.α)]
        gpuNeurons.b[:] = tanh.(gpuNeurons.b .+ (∇b .* ϵ))
    end
    # TODO: propagate until the last input reaches the output
    while false
        
    end
    network.synapses = replace_storage(Array, gpuSynapses)
    network.neurons = replace_storage(Array, gpuNeurons)
    # GC.gc()
    # CUDA.reclaim()
	return out, it, ∑w, ∑b
end

function saveData(data, name::String, path::String = ".", xs::Module...)
    cd(@__DIR__)
    cd(path)
    jldopen(name * ".jld", "w") do file
        for i in xs
            addrequire(file, i)
        end
        write(file, name, data)
    end
    cd(@__DIR__)
end
# load data
# Kaggle cats and dogs
# dogIm = Array{Array{Gray{Normed{UInt8,8}},2},1}(loadData("c:/Users/ahmad/OneDrive/Documents/GitHub/open-ended-neural-nets/DFA_BP/data/kagglecatsanddogs_3367a/processed/", "Dog"))
# catIm = Array{Array{Gray{Normed{UInt8,8}},2},1}(loadData("c:/Users/ahmad/OneDrive/Documents/GitHub/open-ended-neural-nets/DFA_BP/data/kagglecatsanddogs_3367a/processed/", "Cat"))
# merge input into one 3d array
# images = Float16.(reshape(reduce(hcat, [catIm;dogIm]), tuple(size(catIm[1])..., length([catIm; dogIm]))))
# MNIST
# data = MNIST.download()
train_x, train_y = MNIST.traindata()
train_x = Float16.(train_x)
train_y = [Flux.onehot(i, collect(0:9)) for i in train_y]
# when to fire the correct output signal?
# when any forward signal reaches any neuron neighbouring any of the correct_output neurons (neigbour meaning connected by a synapse not including itself)
# specify network size
n_size = length(train_x[:,:,1]) + 64 + 128 + 16 + 64 + 64 + 2000 # network size with pytorch
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

idx = sample(1:size(train_x)[3], 100, replace = false)
samp = train_x[:,:, idx]
result = propagate!(network, samp, 10, 10, 0.001)

err = result[1] .- [Flux.onehot(i, collect(0:9)) for i in train_y[idx]]
fig = scatter([mean(i.^2) for i in err], ms = [1], msc = [:blue], mc = [:blue])
cd(@__DIR__)
savefig(fig, "mse_1_s1000_MNIST")
saveData(result, "result_1_s1000@10_dMNIST",".", Core, FixedPointNumbers)
GC.gc()
heatmap(result[3][1] .* network.synapses.α, c = :delta)
