"""
Threading configuration utilities for COSMO.jl.

Provides automatic thread configuration and validation for optimal performance.
"""

using LinearAlgebra: BLAS

"""
    CPUInfo

Hardware information for threading decisions.
"""
struct CPUInfo
    physical_cores::Int
    logical_cores::Int
    has_hyperthreading::Bool
    recommended_julia_threads::Int
    recommended_blas_threads::Int
end

"""
    detect_cpu_info() -> CPUInfo

Detect CPU topology and recommend thread configuration.

Recommendations:
- Julia threads: Use physical cores (avoid hyperthreading)
- BLAS threads: 1 for multi-threaded Julia, physical cores for single-threaded
- Total threads: Never exceed physical cores to avoid oversubscription

Note: Currently uses Sys.CPU_THREADS as a proxy for physical cores.
For more accurate detection, install Hwloc.jl.
"""
function detect_cpu_info()::CPUInfo
    # Use Sys.CPU_THREADS as proxy for physical cores
    # This is available without additional dependencies
    logical_cores = Sys.CPU_THREADS

    # Assume no hyperthreading for simplicity
    # In practice, physical cores ≈ logical cores / 2 if hyperthreading is enabled
    # But we can't reliably detect this without Hwloc
    physical_cores = logical_cores
    has_hyperthreading = false

    # Recommendations based on COSMO's algorithm characteristics
    # For ADMM with parallel projections:
    # - Prefer Julia threading over BLAS threading
    # - BLAS threading only helps dense operations (KKT solve is sparse)

    recommended_julia_threads = physical_cores
    recommended_blas_threads = 1  # Always 1 for multi-threaded Julia

    return CPUInfo(
        physical_cores,
        logical_cores,
        has_hyperthreading,
        recommended_julia_threads,
        recommended_blas_threads
    )
end

"""
    ThreadConfig

Current threading configuration.
"""
struct ThreadConfig
    julia_threads::Int
    blas_threads::Int
    total_threads::Int
    is_oversubscribed::Bool
    efficiency_estimate::Float64
end

"""
    get_current_thread_config(cpu_info::CPUInfo) -> ThreadConfig

Get current threading configuration and assess efficiency.
"""
function get_current_thread_config(cpu_info::CPUInfo)::ThreadConfig
    julia_threads = Threads.nthreads()
    blas_threads = BLAS.get_num_threads()

    # Worst-case: all Julia threads call BLAS simultaneously
    total_threads = julia_threads * blas_threads

    is_oversubscribed = total_threads > cpu_info.physical_cores

    # Efficiency estimate (heuristic)
    # - Optimal: julia_threads = physical_cores, blas_threads = 1
    # - Penalty for oversubscription
    # - Penalty for underutilization

    if is_oversubscribed
        # Severe penalty for oversubscription
        efficiency = cpu_info.physical_cores / total_threads
    elseif julia_threads == 1
        # Single-threaded Julia: BLAS can use all cores
        efficiency = min(blas_threads / cpu_info.physical_cores, 1.0)
    else
        # Multi-threaded Julia: efficiency based on Julia thread utilization
        julia_efficiency = min(julia_threads / cpu_info.physical_cores, 1.0)
        # Penalty if BLAS > 1 (causes contention)
        blas_penalty = blas_threads == 1 ? 1.0 : 0.5
        efficiency = julia_efficiency * blas_penalty
    end

    return ThreadConfig(
        julia_threads,
        blas_threads,
        total_threads,
        is_oversubscribed,
        efficiency
    )
end

"""
    validate_thread_config(cpu_info::CPUInfo, config::ThreadConfig) -> Vector{String}

Validate threading configuration and return warnings.

Returns empty vector if configuration is optimal.
"""
function validate_thread_config(
    cpu_info::CPUInfo,
    config::ThreadConfig
)::Vector{String}
    warnings = String[]

    # Check oversubscription
    if config.is_oversubscribed
        push!(warnings,
            "CRITICAL: Thread oversubscription detected! " *
            "$(config.julia_threads) Julia threads × $(config.blas_threads) BLAS threads = " *
            "$(config.total_threads) total threads, but only $(cpu_info.physical_cores) " *
            "physical cores available. Expected $(round((1 - config.efficiency_estimate)*100, digits=1))% efficiency loss.")
    end

    # Check BLAS threading with multi-threaded Julia
    if config.julia_threads > 1 && config.blas_threads > 1
        push!(warnings,
            "WARNING: Multi-threaded BLAS ($(config.blas_threads) threads) with " *
            "multi-threaded Julia ($(config.julia_threads) threads) can cause contention. " *
            "Recommended: BLAS.set_num_threads(1)")
    end

    # Check underutilization
    if config.julia_threads == 1 && config.blas_threads < cpu_info.physical_cores
        push!(warnings,
            "INFO: Single-threaded Julia with BLAS=$(config.blas_threads) threads. " *
            "For sequential COSMO, consider BLAS.set_num_threads($(cpu_info.physical_cores)) " *
            "to fully utilize cores.")
    end

    # Check if optimal
    if config.julia_threads == cpu_info.recommended_julia_threads &&
       config.blas_threads == cpu_info.recommended_blas_threads
        push!(warnings, "INFO: Thread configuration is optimal for COSMO.jl")
    end

    return warnings
end

"""
    print_thread_config()

Print current threading configuration and recommendations.

Useful for debugging performance issues.
"""
function print_thread_config()
    println("="^70)
    println("COSMO.jl Threading Configuration")
    println("="^70)
    println()

    cpu_info = detect_cpu_info()
    println("Hardware:")
    println("  Physical cores: $(cpu_info.physical_cores)")
    println("  Logical cores:  $(cpu_info.logical_cores)")
    println("  Hyperthreading: $(cpu_info.has_hyperthreading ? "Yes" : "No")")
    println()

    config = get_current_thread_config(cpu_info)
    println("Current Configuration:")
    println("  Julia threads:  $(config.julia_threads)")
    println("  BLAS threads:   $(config.blas_threads)")
    println("  Total threads:  $(config.total_threads)")
    println("  Oversubscribed: $(config.is_oversubscribed ? "YES ⚠️" : "No")")
    println("  Efficiency:     $(round(config.efficiency_estimate*100, digits=1))%")
    println()

    warnings = validate_thread_config(cpu_info, config)
    if !isempty(warnings)
        println("Messages:")
        for warning in warnings
            println("  • $warning")
        end
        println()
    end

    println("Recommendations:")
    println("  For multi-threaded COSMO (decomposition enabled):")
    println("    export JULIA_NUM_THREADS=$(cpu_info.recommended_julia_threads)")
    println("    BLAS.set_num_threads($(cpu_info.recommended_blas_threads))")
    println()
    println("  For single-threaded COSMO (small problems):")
    println("    export JULIA_NUM_THREADS=1")
    println("    BLAS.set_num_threads($(cpu_info.physical_cores))")
    println("="^70)
end

"""
    auto_configure_blas()

Automatically configure BLAS threading based on Julia thread count.

Rules:
- If Julia has 1 thread: Use all physical cores for BLAS
- If Julia has >1 threads: Use 1 BLAS thread (avoid oversubscription)
"""
function auto_configure_blas()
    cpu_info = detect_cpu_info()
    julia_threads = Threads.nthreads()

    if julia_threads == 1
        # Single-threaded Julia: BLAS can use all cores
        optimal_blas = cpu_info.physical_cores
    else
        # Multi-threaded Julia: Disable BLAS threading
        optimal_blas = 1
    end

    current_blas = BLAS.get_num_threads()
    if current_blas != optimal_blas
        @info "Auto-configuring BLAS threading" julia_threads=julia_threads current_blas=current_blas optimal_blas=optimal_blas
        BLAS.set_num_threads(optimal_blas)
    end

    return optimal_blas
end

# Export public API
export detect_cpu_info, print_thread_config, auto_configure_blas
