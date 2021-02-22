using Random, Distributions, Markdown, Images, GPUArrays, CUDA, Flux, ColorTypes, FixedPointNumbers, Plots, JLD, StructArrays, ProgressBars, MLDatasets, Profile

# test for cuda
CUDA.version()

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
	accumulator::Int64
	cycle_timer::Int64
    b::Float32 # Neuron bias
    δb::Float32 # bias learning rate
end

mutable struct Network
    neurons::StructVector{Neuron}
    synapses::StructArray{Synapse}
    inputNeurons::BitArray{1}
    outputNeurons::BitArray{1}
    signalSrc::BitArray{1}
    signal::Array{Float16,1}
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
                    Float32.(rand(truncated(Normal(0.0, connectivity), -1, 1), (n_size,))), # b
                    Float32.(rand(truncated(Normal(0.0, 0.001 * connectivity), 0, 1), (n_size,))) # δb
                )
            )
        )
        this.synapses = StructArray{Synapse}(
            (
                (
                    Float32.(rand(truncated(Normal(0.0, connectivity), -1, 1), (n_size, n_size))), # w
                    Float32.(rand(truncated(Normal(0, 0.001 * connectivity), 0, 1), (n_size, n_size))), # δw
                    Bool.(rand(Binomial(1, connectivity), (n_size, n_size)))
                )
            )
        )
        if selfConnectivity != connectivity
            for i in 1:size
                this.synapses[i,i].w *= rand(Binomial(1, selfConnectivity / connectivity))
            end
        end
        this.inputNeurons = falses(size)
        this.outputNeurons = falses(size)
        this.signal = zeros(Float16, size)
        this.signalSrc = falses(size)
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
function propagate!(network::Network, χ::Array{Float16,2}, outputLength::Integer, ϵ::AbstractFloat = 0.01, samplingRate::Int = 10)
	# Initialize propagation variables
	local inputLength = Base.length(χ[:,1])
    local inputNum = size(χ)[2]
    if sum(network.inputNeurons) < inputLength
        println("current network input length is smaller than input length, adding $(inputLength - sum(network.inputNeurons))")
        @inbounds network.inputNeurons[sample(findall(el -> !el, network.inputNeurons .& (.!network.outputNeurons)), inputLength - sum(network.inputNeurons), replace = false)] .= true
        @inbounds network.signalSrc = network.inputNeurons
    elseif sum(network.inputNeurons) > inputLength
        println("current network input length is larger than input length, removing $(sum(network.inputNeurons) - inputLength)")
        @inbounds network.inputNeurons[sample(findall(el -> el, network.inputNeurons), sum(network.inputNeurons) - inputLength, replace = false)] .= false
        @inbounds network.signalSrc = network.inputNeurons
    end
    if sum(network.outputNeurons) < outputLength
        println("current network output length is smaller than output length, adding $(outputLength - sum(network.outputNeurons))")
        @inbounds network.outputNeurons[sample(findall(el -> !el, network.outputNeurons .& (.!network.inputNeurons)), outputLength - sum(network.outputNeurons), replace = false)] .= true
    elseif sum(network.outputNeurons) > outputLength
        println("current network output length is larger than output length, removing $(sum(network.outputNeurons) - outputLength)")
        @inbounds network.outputNeurons[sample(findall(el -> el, network.outputNeurons), sum(network.outputNeurons) - outputLength, replace = false)] .= false
    end
    
    local out = Array{Array{Float32,1},1}()
    local ∑w = Array{Array{Float16,2},1}()
    local ∑b = Array{Array{Float16,1},1}()
    local it = 1

    local gpuSynapses::StructArray{Synapse} = replace_storage(CuArray, network.synapses)
    local gpuNeurons::StructVector{Neuron} = replace_storage(CuVector, network.neurons)
    local gpuInputNeurons::CuArray{Bool,1} = cu(network.inputNeurons)
    local gpuOutputNeurons::CuArray{Bool,1} = cu(network.outputNeurons)
    local gpuχ::CuArray{Float16,2} = CuArray(χ)
    local signal::CuArray{Float16,1} = cu(network.signal)
    local signalSrcNeurons::CuArray{Bool,1} = cu(network.signalSrc)
    local inboundSignalSrc::CuArray{Bool,1} = falses(network.size)
    local ∇w::CuArray{Float32,2} = CUDA.zeros(Float32, (network.size, network.size))
    local ∇b::CuArray{Float32,1} = CUDA.zeros(Float32, (network.size))
    print(string("number of input: $inputNum\ninput_length: $inputLength\noutput_size: $outputLength\n"))
    local srcIdx = CUDA.findall(el -> el, signalSrcNeurons)
    local notSrcIdx = CUDA.findall(el -> !el, signalSrcNeurons)
    for x in tqdm(1:inputNum)
        # initialize weight change matrix and bias change vector for each input with zeros
        # TODO: implement pipelining of inputs and immediate updates of weights
        # TODO: implement combination/merger/integration of inbound signal (next input) with current signals that pass through the input neurons
        inboundSignalSrc = signalSrcNeurons .& gpuInputNeurons
        if CUDA.reduce(|,inboundSignalSrc)
            # * modify signal elements that pass through input neurons with the inbound signals
            # ! inconsistent sizes between inboundSignalSrc and findall(el -> (el in Array(signalSrcNeurons)), network.inputNeurons)
            # TODO: Fix this
            let tmp = ((σ.((signal[inboundSignalSrc] .+ (gpuχ[:,x][inboundSignalSrc[gpuInputNeurons]])) .* 2) .* 2) .- 1)
                @inbounds signal[findall(el -> el, inboundSignalSrc)] = tmp
            end
            # * concatenate the rest of the inbound signal to the current signal
            local newSignalSrcIndex::CuArray{Bool,1} = (.!signalSrcNeurons) .& gpuInputNeurons
            @inbounds signal[CUDA.findall(el -> el, newSignalSrcIndex)] = gpuχ[:,x][newSignalSrcIndex[gpuInputNeurons]]
            # * concatenate the signalSrcNeurons of the inbound signal to that of the current signal
            @inbounds signalSrcNeurons .+= newSignalSrcIndex
            srcIdx = CUDA.findall(el -> el, signalSrcNeurons)
            notSrcIdx = CUDA.findall(el -> !el, signalSrcNeurons)
        end
        # TODO: implement weight and bias change array update
        # * pool the updates of an iteration to the weight & bias change arrays
        @inbounds ∇w[srcIdx,:] = tanh.(∇w + (gpuSynapses.δw .* signal))[srcIdx,:]
        @inbounds ∇b[srcIdx] = tanh.(∇b + (gpuNeurons.δb .* signal))[srcIdx]
        # * increment accumulator and stop clock for neurons that fire
        @inbounds gpuNeurons.accumulator[srcIdx] = gpuNeurons.accumulator[srcIdx] + CUDA.ones(Int, length(srcIdx))
        @inbounds gpuNeurons.cycle_timer[srcIdx] = CUDA.zeros(Int, length(srcIdx))
        # * and decrement accumulator and start clock for neurons that don't fire
        @inbounds gpuNeurons.accumulator[notSrcIdx] = gpuNeurons.accumulator[notSrcIdx] - CUDA.ones(Int, length(notSrcIdx))
        @inbounds gpuNeurons.cycle_timer[notSrcIdx] = gpuNeurons.cycle_timer[notSrcIdx] + CUDA.ones(Int, length(notSrcIdx))
        # TODO: compute the probability of a synapse becoming viable or nonviable
        # TODO:update active synapses matrix α
        #= 
        # * This can be implemented with the adaptive synaptogenesis method
        # * or the Invariant information clustering on the neuron level =#
        # TODO: pass signal through network from signalSrcNeurons
        @inbounds signal[srcIdx] = ((gpuSynapses.α .* gpuSynapses.w) * signal)[srcIdx]
        # * update src neurons
        @inbounds signalSrcNeurons = .!CUDA.isapprox.(signal, 0.0)
        srcIdx = CUDA.findall(el -> el, signalSrcNeurons)
        notSrcIdx = CUDA.findall(el -> !el, signalSrcNeurons)
        @inbounds signal[srcIdx] = tanh.(signal + gpuNeurons.b)[srcIdx]
        
        #= 
        # TODO: Network needs to know why the output neurons are special
        # * this can be done by specifying a structure in the network explicitly that either calculates the error, or mutual information
        # * or it can be done by processing and mutating the signal passing through the output neurons
        # ** mutating for UNSUPERVISED learning can be done by things like mutual information in Invariant information clustering (Ji, X., Henriques, J.F. & Vedaldi, A., 2019. Invariant Information Clustering for Unsupervised Image Classification and Segmentation. arXiv:1807.06653 [cs]. Available at: http://arxiv.org/abs/1807.06653.)
        # ** mutating for SUPERVISED learning can be done by replacing the orginal signal with the error =#
        # * extract outbound signals from output neurons
        @inbounds local outboundSignalSrc::BitArray{1} = Array(signalSrcNeurons .& gpuOutputNeurons)
        if CUDA.reduce(|,outboundSignalSrc)
            @inbounds local _out::Array{Float16,1} = zeros(Float16, outputLength)
            @inbounds _out[outboundSignalSrc[network.outputNeurons]] = Array(σ.(signal))[outboundSignalSrc]
            if x % samplingRate == 0
                @inbounds push!(out, Array(_out))
            end
        end
        # * update weights & biases
        let αidx = findall(el -> el, gpuSynapses.α)
            @inbounds gpuSynapses.w[αidx] = tanh.((gpuSynapses.w + transpose(∇w .* ϵ))[αidx])
        end
        @inbounds gpuNeurons.b[:] = tanh.(gpuNeurons.b + (∇b .* ϵ))
        
        if x % samplingRate == 0
            # track weight changes
            @inbounds push!(∑w, Array(Float16.(∇w)))
            @inbounds push!(∑b, Array(Float16.(∇b)))
        end
    end
    # TODO: propagate until the last input reaches the output
    while false
        
    end
    network.signal = Array(signal)
    network.synapses = replace_storage(Array, gpuSynapses)
    network.neurons = replace_storage(Array, gpuNeurons)
    # GC.gc()
    # CUDA.reclaim()
	return out, size(signal), ∑w, ∑b
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
train_x, train_y = MNIST.traindata(Float16)
train_y = [Flux.onehot(i, collect(0:9)) for i in train_y]
# when to fire the correct output signal?
# when any forward signal reaches any neuron neighbouring any of the correct_output neurons (neigbour meaning connected by a synapse not including itself)
# specify network size
n_size = length(train_x[:,:,1]) + 64 + 128 + 16 + 64 + 64 + 2000 # network size with pytorch
# Initialize network instance
network = Network(n_size, 0.0218, 0.0001)
og_network = Network(n_size, 0.0218, 0.0001)
# Decay Rate of synapse viability
decay_rate = 0.5
# Growth Rate of synapse viability when a signal is fired through the synapse
growth_rate = 2.0 * decay_rate

idx = sample(1:size(train_x)[3], 10, replace = false)
samp = reduce(hcat, [train_x[:,:, i][:] for i in idx])
propagate!(network, samp, 10, 0.001, 10)
Profile.clear()
@profile propagate!(network, samp, 10, 0.001, 10)

open("tmp/prof.txt", "w") do s
    Profile.print(IOContext(s, :displaysize => (24, 500)))
end

err = result[1] .- train_y[idx][1:100:end]
fig = scatter([mean(i.^2) for i in err], ms = [1], msc = [:blue], mc = [:blue])
cd(@__DIR__)
savefig(fig, "mse_1_s1000_MNIST")
saveData(result, "result_1_s1000@10_dMNIST",".", Core, FixedPointNumbers)
GC.gc()
heatmap(result[3][1] .* network.synapses.α, c = :delta)
