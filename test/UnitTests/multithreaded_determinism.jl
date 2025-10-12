# Multi-threaded determinism and thread safety tests
# This test ensures deterministic results regardless of thread scheduling and validates thread safety

using COSMO, Test, LinearAlgebra, Random, Threads, Base.Threads

# Helper function to temporarily disable threading in CompositeConvexSet
function force_single_thread_project!(x::COSMO.SplitVector{T}, C::COSMO.CompositeConvexSet{T}) where{T}
    # Manually perform projections without Threads.@threads
    for i = 1:length(C.sets)
        project!(x.views[i], C.sets[i])
    end
    return nothing
end

# Helper function to compare vectors within tolerance
function compare_vectors(v1::AbstractVector, v2::AbstractVector, tol::Real = 1e-10)
    return norm(v1 - v2, Inf) <= tol
end

# Function to run projection and return result
function run_projection(composite_set, x_data)
    x = copy(x_data)
    sv = COSMO.SplitVector(x, composite_set)
    COSMO.project!(sv, composite_set)
    return copy(x)
end

# Function to solve SDP and return result
function solve_sdp(P, q, A, b, C_sets; decompose = false, complete_dual = false)
    settings = COSMO.Settings(
        decompose = decompose,
        complete_dual = complete_dual,
        verbose = false,
        verbose_timing = false,
        eps_abs = 1e-8,
        eps_rel = 1e-8
    )

    model = COSMO.Model()
    assemble!(model, P, q, A, b, C_sets, settings = settings)
    result = optimize!(model)

    return (
        status = result.status,
        obj_val = result.obj_val,
        x = copy(result.x),
        y = copy(result.y),
        iterations = result.iterations
    )
end

@testset "Multi-threaded Determinism and Thread Safety" begin

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(11111)

    @testset "Deterministic Projection Results" begin
        @testset "PSD Cones - Size $(n)x$(n), $(n_cones) cones" for (n, n_cones) in [(3, 3), (4, 2), (5, 2)]
            # Create composite set
            psd_cones = [COSMO.PsdCone(n * n) for _ in 1:n_cones]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            x_data = []
            for _ in 1:n_cones
                X = randn(rng, n, n)
                X = Symmetric(X - 1.5 * I)
                append!(x_data, vec(X))
            end

            # Run projection multiple times and verify deterministic results
            results = []
            for run in 1:10
                result = run_projection(composite_set, x_data)
                push!(results, result)
            end

            # All results should be identical
            for i in 2:length(results)
                @test compare_vectors(results[1], results[i], 1e-12)
            end

            # Also compare with single-threaded version
            x_single = copy(x_data)
            sv_single = COSMO.SplitVector(x_single, composite_set)
            force_single_thread_project!(sv_single, composite_set)

            @test compare_vectors(x_single, results[1], 1e-12)
        end
    end

    @testset "Deterministic SDP Solver Results" begin
        @testset "Closest Correlation - Size $(n)" for n in [3, 4, 5]
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

            # Run solver multiple times and verify deterministic results
            results = []
            for run in 1:5
                result = solve_sdp(P, q, A, b, C_sets, decompose = true, complete_dual = true)
                push!(results, result)
            end

            # All results should be identical (within numerical tolerance)
            for i in 2:length(results)
                @test results[i].status == results[1].status
                @test abs(results[i].obj_val - results[1].obj_val) <= 1e-8
                @test compare_vectors(results[i].x, results[1].x, 1e-6)
                @test results[i].iterations == results[1].iterations
            end
        end
    end

    @testset "Concurrent Solver Instances" begin
        @testset "Multiple concurrent SDP solves" begin
            n = 4
            C = randn(rng, n, n) * 0.5
            C = Symmetric(C)

            n2 = n * n
            P = spdiagm(0 => ones(n2))
            q = -vec(C)

            A_diag = spzeros(n, n2)
            for i in 1:n
                col_idx = (i - 1) * n + i
                A_diag[i, col_idx] = 1.0
            end
            b_diag = -ones(n)

            A_psd = spdiagm(0 => ones(n2))
            b_psd = zeros(n2)

            A = [A_diag; A_psd]
            b = [b_diag; b_psd]
            C_sets = [COSMO.ZeroSet(n), COSMO.PsdCone(n2)]

            # Run multiple solver instances concurrently
            n_threads = min(Threads.nthreads(), 4)
            results = Vector{Any}(undef, n_threads)

            Threads.@threads for i in 1:n_threads
                results[i] = solve_sdp(P, q, A, b, C_sets, decompose = true, complete_dual = true)
            end

            # All concurrent solves should give similar results
            for i in 2:length(results)
                @test results[i].status == results[1].status
                @test abs(results[i].obj_val - results[1].obj_val) <= 1e-6
                @test compare_vectors(results[i].x, results[1].x, 1e-4)
            end
        end
    end

    @testset "Thread Safety of Data Structures" begin
        @testset "Shared Composite Set - Multiple Threads" begin
            # Create a shared composite set
            psd_cones = [COSMO.PsdCone(9) for _ in 1:4]  # 4 x 3x3 PSD cones
            shared_composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            test_cases = []
            for case in 1:8
                x_case = []
                for _ in 1:4
                    X = randn(rng, 3, 3)
                    X = Symmetric(X - 1.5 * I)
                    append!(x_case, vec(X))
                end
                push!(test_cases, x_case)
            end

            # Run projections on shared data structure from multiple threads
            results = Vector{Any}(undef, length(test_cases))

            Threads.@threads for i in 1:length(test_cases)
                results[i] = run_projection(shared_composite_set, test_cases[i])
            end

            # Compare each result with single-threaded version
            for (i, test_case) in enumerate(test_cases)
                x_single = copy(test_case)
                sv_single = COSMO.SplitVector(x_single, shared_composite_set)
                force_single_thread_project!(sv_single, shared_composite_set)

                @test compare_vectors(x_single, results[i], 1e-12)
            end
        end
    end

    @testset "Randomized Test Order" begin
        @testset "Random execution order" for trial in 1:5
            # Create test configurations
            configs = [
                (3, 2), (4, 2), (3, 3), (5, 2), (4, 3)  # (matrix_size, n_cones)
            ]

            # Shuffle configurations
            shuffled_configs = shuffle(rng, configs)

            for (n, n_cones) in shuffled_configs
                # Create composite set
                psd_cones = [COSMO.PsdCone(n * n) for _ in 1:n_cones]
                composite_set = COSMO.CompositeConvexSet(psd_cones)

                # Generate test data
                x_data = []
                for _ in 1:n_cones
                    X = randn(rng, n, n)
                    X = Symmetric(X - 1.5 * I)
                    append!(x_data, vec(X))
                end

                # Run projection multiple times in random order
                results = []
                for run in 1:3
                    result = run_projection(composite_set, x_data)
                    push!(results, result)
                end

                # Results should be deterministic regardless of execution order
                for i in 2:length(results)
                    @test compare_vectors(results[1], results[i], 1e-12)
                end
            end
        end
    end

    @testset "Stress Test - Rapid Successive Calls" begin
        @testset "Many rapid projection calls" begin
            n = 3
            n_cones = 6
            psd_cones = [COSMO.PsdCone(n * n) for _ in 1:n_cones]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            x_data = []
            for _ in 1:n_cones
                X = randn(rng, n, n)
                X = Symmetric(X - 1.5 * I)
                append!(x_data, vec(X))
            end

            # Make many rapid calls
            n_calls = 50
            results = Vector{Any}(undef, n_calls)

            for i in 1:n_calls
                results[i] = run_projection(composite_set, x_data)
            end

            # All results should be identical
            for i in 2:length(results)
                @test compare_vectors(results[1], results[i], 1e-12)
            end

            # Also compare with single-threaded version
            x_single = copy(x_data)
            sv_single = COSMO.SplitVector(x_single, composite_set)
            force_single_thread_project!(sv_single, composite_set)

            @test compare_vectors(x_single, results[1], 1e-12)
        end
    end

    @testset "Memory Safety" begin
        @testset "No memory corruption with repeated operations" begin
            # Create a relatively large problem
            n = 5
            n_cones = 4
            psd_cones = [COSMO.PsdCone(n * n) for _ in 1:n_cones]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            test_cases = []
            for case in 1:20
                x_case = []
                for _ in 1:n_cones
                    X = randn(rng, n, n)
                    X = Symmetric(X - 1.5 * I)
                    append!(x_case, vec(X))
                end
                push!(test_cases, x_case)
            end

            # Run many operations with different data
            results = Vector{Any}(undef, length(test_cases))
            references = Vector{Any}(undef, length(test_cases))

            for (i, test_case) in enumerate(test_cases)
                # Multi-threaded result
                results[i] = run_projection(composite_set, test_case)

                # Reference single-threaded result
                x_ref = copy(test_case)
                sv_ref = COSMO.SplitVector(x_ref, composite_set)
                force_single_thread_project!(sv_ref, composite_set)
                references[i] = copy(x_ref)
            end

            # All results should match their references
            for i in 1:length(test_cases)
                @test compare_vectors(references[i], results[i], 1e-12)
            end
        end
    end

    @testset "Edge Cases and Corner Situations" begin
        @testset "Empty and singleton composite sets" begin
            # Test with single cone
            psd_cone = COSMO.PsdCone(4)
            composite_single = COSMO.CompositeConvexSet([psd_cone])

            X = randn(rng, 2, 2)
            X = Symmetric(X - 1.5 * I)
            x_data = vec(X)

            # Multiple runs should be deterministic
            result1 = run_projection(composite_single, x_data)
            result2 = run_projection(composite_single, x_data)

            @test compare_vectors(result1, result2, 1e-12)

            # Compare with single-threaded
            x_ref = copy(x_data)
            sv_ref = COSMO.SplitVector(x_ref, composite_single)
            force_single_thread_project!(sv_ref, composite_single)

            @test compare_vectors(x_ref, result1, 1e-12)
        end

        @testset "Very small problems" begin
            # Test 1x1 PSD cones
            psd_cones = [COSMO.PsdCone(1) for _ in 1:5]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            x_data = randn(rng, 5)

            # Multiple runs
            results = []
            for _ in 1:10
                result = run_projection(composite_set, x_data)
                push!(results, result)
            end

            # All should be identical
            for i in 2:length(results)
                @test compare_vectors(results[1], results[i], 1e-12)
            end

            # All entries should be non-negative
            @test all(results[1] .>= 0.0)
        end
    end
end