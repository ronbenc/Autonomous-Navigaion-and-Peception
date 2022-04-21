using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
end


function PropagateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # predict
    μp = (F * μb) + a    
    Σp = (F * Σb * transpose(F)) + Σw  
    return MvNormal(μp, Σp)
end 



function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    # predict
    mv = PropagateBelief(b,𝒫,a)
    μp = mv.μ
    Σp = mv.Σ
    # update
    k = Σp * inv(Σp + Σv)      
    μb′ = μp + (k *(o - μp))
    I = UniformScaling(1)
    Σb′ = (I - k) * Σp 
    #
    return MvNormal(μb′, Σb′)
end    

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    w = rand(MvNormal([0.0, 0.0], 𝒫.Σw))
    return 𝒫.F*x + a + w
end 

function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1})
    v = rand(MvNormal([0.0, 0.0], 𝒫.Σv))
    return x + v
end   

# ron - a helper function for 2.a
function GenerateRelativeObservation(x::Array{Float64, 1}, x_b::Array{Float64, 1}, r::Float64, rmin::Float64, fixed::Bool)::Array{Float64, 1}
    if fixed == true
        Σv = 0.01^2*[1.0 0.0; 0.0 1.0]
    else
        Σv = 0.01*max(r, rmin)*[1.0 0.0; 0.0 1.0]
    end
    rel_loc = x
    noise = rand(MvNormal([0.0, 0.0], Σv)) # generate white noise with covariance Σv and zero mean
    return rel_loc + noise
end


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1}, fixed::Bool)::Union{NamedTuple, Nothing}
    distances = [norm(x - 𝒫.beacons[i, :]) for i in range(1, length=size(𝒫.beacons, 1))]
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            obs = GenerateRelativeObservation(x, 𝒫.beacons[index, :], distance, 𝒫.rmin, fixed)
            return (obs=obs, index=index) 
        end    
    end 
    return nothing    
end    

function part1()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1

    # set beacons locations 
    beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]

    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                    Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                    Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                    rng = rng , beacons=beacons, d=d, rmin=rmin) 
    ak = [1.0, 0.0]
    xgt0 = [0.5, 0.2]
    T = 10
    # generating the trajectory
    τ = [xgt0]

    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end  
    # generate observation trajectory
    τobs = Array{Float64, 1}[]
    for i in 1:T
        push!(τobs, GenerateObservation(𝒫, τ[i]))
    end  

    # generate beliefs dead reckoning 
    τbp = [b0]

    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end

    #generate posteriors 
    τb = [b0]
    for i in 1:T-1
        push!(τb, PropagateUpdateBelief(τb[end],  𝒫, ak, τobs[i+1]))
    end

    # plots 
    dr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τbp[i].μ, τbp[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    savefig(dr,"dr.pdf")

    tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τb[i].μ, τb[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    savefig(tr,"tr.pdf")
end

function part2()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d = 1.0 
    rmin = 0.1

    # set beacons locations 
    beacons =  [0.0 0.0; 0.0 4.0; 0.0 8.0; 4.0 0.0; 4.0 4.0; 4.0 8.0; 8.0 0.0; 8.0 4.0; 8.0 8.0]

    bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(bplot,"beacons.pdf")

    𝒫 = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                    Σw=0.1^2*[1.0 0.0; 0.0 1.0],
                    Σv=0.01^2*[1.0 0.0; 0.0 1.0], 
                    rng = rng , beacons=beacons, d=d, rmin=rmin)

    ak = [0.1, 0.1]
    xgt0 = [-0.5, -0.2]
    T = 100
    # generating the trajectory
    τ = [xgt0]

    # generate motion trajectory
    for i in 1:T-1
        push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    end 

    # generate observation trajectory
    τobsbeacons = []
    for i in 1:T
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i], false))
    end 

    # generate beliefs dead reckoning 
    τbp = [b0]

    for i in 1:T-1
        push!(τbp, PropagateBelief(τbp[end],  𝒫, ak))
    end


    # generate posteriors 
    τb_not_fixed = [b0]
    for i in 1:T-1
        if isnothing(τobsbeacons[i+1])
            push!(τb_not_fixed, PropagateBelief(τb_not_fixed[end],  𝒫, ak)) 
        else
            push!(τb_not_fixed, PropagateUpdateBelief(τb_not_fixed[end],  𝒫, ak, τobsbeacons[i+1][1])) 
        end
    end
    
    # plots 
    dr2=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τbp[i].μ, τbp[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    # savefig([dr2, bplot], "dr2.pdf") Ron -how to combine two scatters???
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(dr2, "dr2.pdf")
    
    # Ron - what if no observation???
    tr2notfixed=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τb_not_fixed[i].μ, τb_not_fixed[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(tr2notfixed,"tr2notfixed.pdf")


    # clause c.2
    # generate observation trajectory
    τobsbeacons = []
    for i in 1:T
        push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i], true))
    end 

    # generate posteriors 
    τb_fixed = [b0]
    for i in 1:T-1
        if isnothing(τobsbeacons[i+1])
            push!(τb_fixed, PropagateBelief(τb_fixed[end],  𝒫, ak)) 
        else
            push!(τb_fixed, PropagateUpdateBelief(τb_fixed[end],  𝒫, ak, τobsbeacons[i+1][1])) 
        end
    end
    
    # plots 
    
    # Ron - what if no observation???
    tr2fixed=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    for i in 1:T
        covellipse!(τb_fixed[i].μ, τb_fixed[i].Σ, showaxes=true, n_std=3, label="step $i")
    end
    scatter!(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    savefig(tr2fixed,"tr2fixed.pdf")

    # clause c.3
    # graph for not-fixed
    not_fixed_error = []
    fixed_error = []
    for i in 1:T
        push!(not_fixed_error, norm(τ[i]-τb_not_fixed[i].μ)) 
        push!(fixed_error, norm(τ[i]-τb_fixed[i].μ))
    end

    plt = plot(; xlabel="Time", ylabel="Error", grid=:true, legend=:outertopright, legendfont=font(5))
    plot!(not_fixed_error, label= "not fixed covariance error")
    plot!(fixed_error, label= "fixed covariance error")
    savefig(plt, "fixed_vs_not_fixed_error.pdf")

    # clause c.4
    square_root_trace_not_fixed = []
    square_root_trace_fixed = []
    for i in 1:T
        push!(square_root_trace_not_fixed, sqrt(tr(τb_not_fixed[i].Σ))) 
        push!(square_root_trace_fixed, sqrt(tr(τb_fixed[i].Σ)))
    end

    plt = plot(; xlabel="Time", ylabel="square root of trace of est cov", grid=:true, legend=:outertopright, legendfont=font(5))
    plot!(square_root_trace_not_fixed, label= "square root of trace of not fixed covariance")
    plot!(square_root_trace_fixed, label= "square root of trace of fixed covariance")
    savefig(plt, "fixed_vs_not_fixed_sqrt_of_trace.pdf")

end

function main()
    # part1()
    part2()
    
    # use function det(b.Σ) to calculate determinant of the matrix
end 

main()