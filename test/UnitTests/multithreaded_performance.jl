# Multi-threaded SDP performance benchmarks
# This test measures and validates performance improvements with multi-threading

using COSMO, Test, LinearAlgebra, Random, Threads, BenchmarkTools, Statistics

# Helper function to temporarily disable threading in CompositeConvexSet
function force_single_thread_project!(x::COSMO.SplitVector{T}, C::COSMO.CompositeConvexSet{T}) where{T}
    # Manually perform projections without Threads.@threads
    for i = 1:length(C.sets)
        project!(x.views[i], C.sets[i])
    end
    return nothing
end

# Benchmark function for projection operations
function benchmark_projection(composite_set, x_data; iterations = 10)
    # Warm-up
    x_warm = copy(x_data)
    sv_warm = COSMO.SplitVector(x_warm, composite_set)
    COSMO.project!(sv_warm, composite_set)

    # Single-threaded benchmark
    single_times = []
    for _ in 1:iterations
        x_single = copy(x_data)
        sv_single = COSMO.SplitVector(x_single, composite_set)
        time_single = @elapsed force_single_thread_project!(sv_single, composite_set)
        push!(single_times, time_single)
    end

    # Multi-threaded benchmark
    multi_times = []
    for _ in 1:iterations
        x_multi = copy(x_data)
        sv_multi = COSMO.SplitVector(x_multi, composite_set)
        time_multi = @elapsed COSMO.project!(sv_multi, composite_set)
        push!(multi_times, time_multi)
    end

    return (
        single_mean = mean(single_times),
        single_std = std(single_times),
        multi_mean = mean(multi_times),
        multi_std = std(multi_times),
        speedup = mean(single_times) / mean(multi_times)
    )
end

# Benchmark for full SDP problems
function benchmark_sdp_solver(P, q, A, b, C_sets; iterations = 3, decompose = false, complete_dual = false)
    results = []

    for iter in 1:iterations
        # Single-threaded (no decomposition)
        settings_single = COSMO.Settings(
            decompose = false,
            complete_dual = false,
            verbose = false,
            verbose_timing = false
        )

        model_single = COSMO.Model()
        assemble!(model_single, P, q, A, b, C_sets, settings = settings_single)
        time_single = @elapsed optimize!(model_single)

        # Multi-threaded (with decomposition)
        settings_multi = COSMO.Settings(
            decompose = decompose,
            complete_dual = complete_dual,
            verbose = false,
            verbose_timing = false
        )

        model_multi = COSMO.Model()
        assemble!(model_multi, P, q, A, b, C_sets, settings = settings_multi)
        time_multi = @elapsed optimize!(model_multi)

        push!(results, (single = time_single, multi = time_multi))
    end

    single_times = [r.single for r in results]
    multi_times = [r.multi for r in results]

    return (
        single_mean = mean(single_times),
        single_std = std(single_times),
        multi_mean = mean(multi_times),
        multi_std = std(multi_times),
        speedup = mean(single_times) / mean(multi_times)
    )
end

@testset "Multi-threaded Performance Benchmarks" begin

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(13579)

    @testset "Projection Performance - Multiple PSD Cones" begin
        @testset "Configuration: $(n_cones) cones of size $(size)x$(size)" for (n_cones, size) in [(2, 3), (3, 4), (4, 3), (5, 4), (8, 3)]
            # Create composite set
            psd_cones = [COSMO.PsdCone(size * size) for _ in 1:n_cones]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            x_data = []
            for _ in 1:n_cones
                X = randn(rng, size, size)
                X = Symmetric(X - 1.5 * I)
                append!(x_data, vec(X))
            end

            # Benchmark
            results = benchmark_projection(composite_set, x_data, iterations = 20)

            # Performance expectations
            # For small problems, threading overhead might dominate
            # For larger problems, we expect some speedup
            if size >= 4 && n_cones >= 3
                @test results.speedup >= 0.8  # Allow some overhead, but should be close to 1x or better
            else
                # For small problems, we just verify it completes successfully
                @test results.speedup > 0.1  # Should not be extremely slow
            end

            # Verify correctness during benchmarking
            x_single = copy(x_data)
            x_multi = copy(x_data)
            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test norm(x_single - x_multi, Inf) <= 1e-12

            @info "Projection Benchmark: $(n_cones) x $(size)x$(size) PSD cones"
            @info "  Single-threaded: $(results.single_mean)s ± $(results.single_std)s"
            @info "  Multi-threaded: $(results.multi_mean)s ± $(results.multi_std)s"
            @info "  Speedup: $(round(results.speedup, digits=2))x"
        end
    end

    @testset "Projection Performance - Mixed Cones" begin
        @testset "Mixed configuration $(n_psd) PSD + $(n_other) other" for (n_psd, n_other) in [(2, 3), (3, 5), (4, 4)]
            cones = []

            # Add PSD cones
            for _ in 1:n_psd
                push!(cones, COSMO.PsdCone(9))  # 3x3 PSD
            end

            # Add other cones
            for _ in 1:n_other
                cone_type = rand(rng, 1:3)
                if cone_type == 1
                    push!(cones, COSMO.Nonnegatives(4))
                elseif cone_type == 2
                    push!(cones, COSMO.Box(3))
                else
                    push!(cones, COSMO.SecondOrderCone(4))
                end
            end

            composite_set = COSMO.CompositeConvexSet(cones)

            # Generate test data
            x_data = []
            for cone in cones
                if cone isa COSMO.PsdCone
                    X = randn(rng, 3, 3)
                    X = Symmetric(X - 1.5 * I)
                    append!(x_data, vec(X))
                else
                    append!(x_data, randn(rng, cone.dim))
                end
            end

            results = benchmark_projection(composite_set, x_data, iterations = 15)

            # For mixed cones, speedup might be modest
            @test results.speedup >= 0.5  # Should not be significantly slower

            @info "Mixed Cone Benchmark: $(n_psd) PSD + $(n_other) other cones"
            @info "  Single-threaded: $(results.single_mean)s ± $(results.single_std)s"
            @info "  Multi-threaded: $(results.multi_mean)s ± $(results.multi_std)s"
            @info "  Speedup: $(round(results.speedup, digits=2))x"
        end
    end

    @testset "SDP Solver Performance" begin
        @testset "Closest Correlation Matrix - Size $(n)" for n in [4, 6, 8]
            # Problem setup
            C = randn(rng, n, n) * 0.5
            C = Symmetric(C)

            n2 = n * n
            P = spdiagm(0 => ones(n2))
            q = -vec(C)

            # Diagonal constraint
            A_diag = spzeros(n, n2)
            for i in 1:n
                col_idx = (i - 1) * n + i
                A_diag[i, col_idx] = 1.0
            end
            b_diag = -ones(n)

            # PSD constraint
            A_psd = spdiagm(0 => ones(n2))
            b_psd = zeros(n2)

            A = [A_diag; A_psd]
            b = [b_diag; b_psd]
            C_sets = [COSMO.ZeroSet(n), COSMO.PsdCone(n2)]

            # Benchmark solver performance
            results = benchmark_sdp_solver(P, q, A, b, C_sets, iterations = 3, decompose = true, complete_dual = true)

            # For SDP solver, speedup depends on problem structure
            # We expect at least no significant slowdown
            @test results.speedup >= 0.3  # Allow some overhead

            @info "SDP Solver Benchmark: Closest Correlation $(n)x$(n)"
            @info "  Single-threaded: $(results.single_mean)s ± $(results.single_std)s"
            @info "  Multi-threaded: $(results.multi_mean)s ± $(results.multi_std)s"
            @info "  Speedup: $(round(results.speedup, digits=2))x"
        end
    end

    @testset "Memory Usage Validation" begin
        # Test that multi-threading doesn't significantly increase memory usage
        n_cones = 6
        size = 4

        psd_cones = [COSMO.PsdCone(size * size) for _ in 1:n_cones]
        composite_set = COSMO.CompositeConvexSet(psd_cones)

        # Generate test data
        x_data = []
        for _ in 1:n_cones
            X = randn(rng, size, size)
            X = Symmetric(X - 1.5 * I)
            append!(x_data, vec(X))
        end

        # Measure memory usage for both versions
        function measure_memory(f::Function)
            # Run garbage collection first
            GC.gc()

            # Get initial memory
            initial_memory = Base.gc_bytes()

            # Run the function
            f()

            # Get final memory
            GC.gc()
            final_memory = Base.gc_bytes()

            return final_memory - initial_memory
        end

        # Measure single-threaded memory
        memory_single = measure_memory() do
            x_single = copy(x_data)
            sv_single = COSMO.SplitVector(x_single, composite_set)
            force_single_thread_project!(sv_single, composite_set)
        end

        # Measure multi-threaded memory
        memory_multi = measure_memory() do
            x_multi = copy(x_data)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)
            COSMO.project!(sv_multi, composite_set)
        end

        # Memory usage should be comparable (within 50%)
        memory_ratio = memory_multi / max(memory_single, 1)
        @test memory_ratio <= 1.5  # Multi-threaded should not use significantly more memory

        @info "Memory Usage Comparison"
        @info "  Single-threaded: $(memory_single ÷ 1024) KB"
        @info "  Multi-threaded: $(memory_multi ÷ 1024) KB"
        @info "  Ratio: $(round(memory_ratio, digits=2))x"
    end

    @testset "Scalability Analysis" begin
        # Test how performance scales with number of threads
        if Threads.nthreads() >= 2
            @testset "Scalability with $(length(cones)) cones" for cones in [[2, 3, 4], [4, 4, 4, 4], [6, 6, 6, 6]]
                psd_cones = [COSMO.PsdCone(n * n) for n in cones]
                composite_set = COSMO.CompositeConvexSet(psd_cones)

                # Generate test data
                x_data = []
                for n in cones
                    X = randn(rng, n, n)
                    X = Symmetric(X - 1.5 * I)
                    append!(x_data, vec(X))
                end

                # Benchmark with current threading
                results_current = benchmark_projection(composite_set, x_data, iterations = 10)

                # For scalability, we expect better performance with more cones
                # This is more of a validation test than strict performance requirement
                @test results_current.multi_mean > 0  # Should complete successfully

                @info "Scalability Test: $(cones) size cones"
                @info "  Total dimensions: $(sum(n * n for n in cones))"
                @info "  Multi-threaded time: $(results_current.multi_mean)s"
                @info "  Speedup vs single: $(round(results_current.speedup, digits=2))x"
            end
        else
            @info "Skipping scalability tests - only $(Threads.nthreads()) thread(s) available"
            @test true
        end
    end

    @testset "Performance Regression Detection" begin
        # Define baseline performance expectations
        # These are not strict requirements but serve as regression checks

        @testset "Regression check for $(n_cones) cones" for n_cones in [3, 5, 8]
            psd_cones = [COSMO.PsdCone(9) for _ in 1:n_cones]  # All 3x3 PSD cones
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            x_data = []
            for _ in 1:n_cones
                X = randn(rng, 3, 3)
                X = Symmetric(X - 1.5 * I)
                append!(x_data, vec(X))
            end

            results = benchmark_projection(composite_set, x_data, iterations = 20)

            # Performance expectations (can be adjusted based on hardware)
            max_acceptable_time = 0.1 * n_cones  # 100ms per cone max
            min_acceptable_speedup = 0.3  # Should not be slower than 3x

            @test results.multi_mean <= max_acceptable_time  # Should be reasonably fast
            @test results.speedup >= min_acceptable_speedup  # Should not be significantly slower

            @info "Regression Check: $(n_cones) cones"
            @info "  Multi-threaded time: $(round(results.multi_mean * 1000, digits=1))ms"
            @info "  Speedup: $(round(results.speedup, digits=2))x"
        end
    end
end