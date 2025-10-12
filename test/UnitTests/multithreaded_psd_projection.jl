# Multi-threaded PSD cone projection tests
# This test compares the results of single-threaded vs multi-threaded PSD cone projections

using COSMO, Test, LinearAlgebra, Random, Threads

# Helper function to temporarily disable threading in CompositeConvexSet
function force_single_thread_project!(x::COSMO.SplitVector{T}, C::COSMO.CompositeConvexSet{T}) where{T}
    # Manually perform projections without Threads.@threads
    for i = 1:length(C.sets)
        project!(x.views[i], C.sets[i])
    end
    return nothing
end

# Helper function to compare two vectors within tolerance
function compare_vectors(v1::AbstractVector, v2::AbstractVector, tol::Real = 1e-10)
    return norm(v1 - v2, Inf) <= tol
end

# Helper function to compare PSD matrices
function compare_psd_matrices(x1::AbstractVector, x2::AbstractVector, sqrt_dim::Int, tol::Real = 1e-10)
    X1 = reshape(x1, sqrt_dim, sqrt_dim)
    X2 = reshape(x2, sqrt_dim, sqrt_dim)
    return norm(X1 - X2, Inf) <= tol
end

@testset "Multi-threaded PSD Cone Projection" begin

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(12345)

    @testset "Basic PSD Cone Projection" begin
        @testset "Matrix size: $(n)x$(n)" for n in [2, 3, 4, 5, 6, 8, 10]
            dim = n * n

            # Generate random test matrix
            X = randn(rng, n, n)
            X = Symmetric(X - 2.0 * I)  # Make it not necessarily PSD
            x = vec(X)

            # Create PSD cone
            psd_cone = COSMO.PsdCone(dim)

            # Create composite set with single PSD cone
            composite_set = COSMO.CompositeConvexSet([psd_cone])

            # Test with original data
            x_single = copy(x)
            x_multi = copy(x)

            # Create split vectors
            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            # Perform single-threaded projection
            force_single_thread_project!(sv_single, composite_set)

            # Perform multi-threaded projection
            COSMO.project!(sv_multi, composite_set)

            # Compare results
            @test compare_psd_matrices(x_single, x_multi, n, 1e-12)

            # Verify both results are PSD
            X_single = reshape(x_single, n, n)
            X_multi = reshape(x_multi, n, n)
            @test minimum(eigvals(Symmetric(X_single))) >= -1e-10
            @test minimum(eigvals(Symmetric(X_multi))) >= -1e-10
        end
    end

    @testset "PSD Cone Triangle Projection" begin
        @testset "Matrix size: $(n)x$(n)" for n in [2, 3, 4, 5, 6, 8]
            dim = div(n * (n + 1), 2)

            # Generate random test matrix
            X = randn(rng, n, n)
            X = Symmetric(X - 1.5 * I)  # Make it not necessarily PSD

            # Convert to triangle format
            x_tri = zeros(dim)
            k = 0
            for j in 1:n
                for i in 1:j
                    k += 1
                    if i == j
                        x_tri[k] = X[i, j]
                    else
                        x_tri[k] = X[i, j] * sqrt(2.0)
                    end
                end
            end

            # Create PSD triangle cone
            psd_tri_cone = COSMO.PsdConeTriangle(dim)

            # Create composite set
            composite_set = COSMO.CompositeConvexSet([psd_tri_cone])

            # Test data
            x_single = copy(x_tri)
            x_multi = copy(x_tri)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            # Perform projections
            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            # Compare results
            @test compare_vectors(x_single, x_multi, 1e-12)

            # Verify both results are PSD when reconstructed
            X_single = COSMO.matrixify(x_single)
            X_multi = COSMO.matrixify(x_multi)
            @test minimum(eigvals(Symmetric(X_single))) >= -1e-10
            @test minimum(eigvals(Symmetric(X_multi))) >= -1e-10
        end
    end

    @testset "Multiple Random Test Cases" begin
        for test_case in 1:20
            n = rand(rng, [2, 3, 4, 5, 6, 7, 8])
            dim = n * n

            # Generate random matrix with different conditioning
            X = randn(rng, n, n)
            X = Symmetric(X - rand(rng) * 3.0 * I)
            x = vec(X)

            psd_cone = COSMO.PsdCone(dim)
            composite_set = COSMO.CompositeConvexSet([psd_cone])

            x_single = copy(x)
            x_multi = copy(x)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_psd_matrices(x_single, x_multi, n, 1e-12)
        end
    end

    @testset "Edge Cases" begin
        # Test 1x1 case
        @testset "1x1 PSD Cone" begin
            for val in [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0]
                x_single = [val]
                x_multi = [val]

                psd_cone = COSMO.PsdCone(1)
                composite_set = COSMO.CompositeConvexSet([psd_cone])

                sv_single = COSMO.SplitVector(x_single, composite_set)
                sv_multi = COSMO.SplitVector(x_multi, composite_set)

                force_single_thread_project!(sv_single, composite_set)
                COSMO.project!(sv_multi, composite_set)

                @test compare_vectors(x_single, x_multi, 1e-12)
                @test x_single[1] >= 0.0
                @test x_multi[1] >= 0.0
            end
        end

        # Test with already PSD matrices
        @testset "Already PSD Matrices" begin
            n = 4
            X = randn(rng, n, n)
            X = X' * X + I  # Ensure PSD
            x = vec(X)

            psd_cone = COSMO.PsdCone(n * n)
            composite_set = COSMO.CompositeConvexSet([psd_cone])

            x_single = copy(x)
            x_multi = copy(x)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_psd_matrices(x_single, x_multi, n, 1e-12)
        end
    end
end