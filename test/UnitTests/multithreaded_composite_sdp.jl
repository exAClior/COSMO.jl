# Multi-threaded composite SDP constraint tests
# This test compares the results of single-threaded vs multi-threaded projections
# on composite sets containing multiple PSD constraints

using COSMO, Test, LinearAlgebra, Random, Threads, SparseArrays

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

@testset "Multi-threaded Composite SDP Sets" begin

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(54321)

    @testset "Multiple PSD Cones - Different Sizes" begin
        @testset "Sizes: $(sizes)" for sizes in [[2, 3], [2, 2, 2], [3, 4], [2, 3, 4], [2, 2, 3, 3]]
            # Create PSD cones of different sizes
            psd_cones = [COSMO.PsdCone(n * n) for n in sizes]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate random test data for each cone
            x_data = []
            total_dim = 0
            for n in sizes
                X = randn(rng, n, n)
                X = Symmetric(X - 1.5 * I)  # Make it not necessarily PSD
                append!(x_data, vec(X))
                total_dim += n * n
            end

            # Test with original data
            x_single = copy(x_data)
            x_multi = copy(x_data)

            # Create split vectors
            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            # Perform single-threaded projection
            force_single_thread_project!(sv_single, composite_set)

            # Perform multi-threaded projection
            COSMO.project!(sv_multi, composite_set)

            # Compare results
            @test compare_vectors(x_single, x_multi, 1e-12)

            # Verify all projected matrices are PSD
            offset = 0
            for (i, n) in enumerate(sizes)
                dim = n * n
                x_single_part = @view x_single[(offset+1):(offset+dim)]
                x_multi_part = @view x_multi[(offset+1):(offset+dim)]

                X_single = reshape(x_single_part, n, n)
                X_multi = reshape(x_multi_part, n, n)

                @test minimum(eigvals(Symmetric(X_single))) >= -1e-10
                @test minimum(eigvals(Symmetric(X_multi))) >= -1e-10

                offset += dim
            end
        end
    end

    @testset "Mixed PSD and Other Cones" begin
        @testset "PSD + Nonnegatives + Box" begin
            # Create composite set with different cone types
            psd_cone_3x3 = COSMO.PsdCone(9)  # 3x3 PSD
            nonneg_cone = COSMO.Nonnegatives(5)  # 5-dim nonnegative orthant
            box_cone = COSMO.Box(3)  # 3-dim box [-1, 1]
            box_cone.l .= -1.0
            box_cone.u .= 1.0

            composite_set = COSMO.CompositeConvexSet([psd_cone_3x3, nonneg_cone, box_cone])

            # Generate test data
            X_psd = randn(rng, 3, 3)
            X_psd = Symmetric(X_psd - 2.0 * I)
            x_nonneg = randn(rng, 5)
            x_box = 2.0 * randn(rng, 3)

            x_data = vcat(vec(X_psd), x_nonneg, x_box)

            # Test projections
            x_single = copy(x_data)
            x_multi = copy(x_data)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_vectors(x_single, x_multi, 1e-12)

            # Verify constraints
            # PSD part
            X_single_psd = reshape(@view(x_single[1:9]), 3, 3)
            @test minimum(eigvals(Symmetric(X_single_psd))) >= -1e-10

            # Nonnegatives part
            @test all(@view(x_single[10:14]) .>= -1e-10)

            # Box part
            @test all(@view(x_single[15:17]) .<= 1.0 + 1e-10)
            @test all(@view(x_single[15:17]) .>= -1.0 - 1e-10)
        end
    end

    @testset "PSD Triangle Cones in Composite Set" begin
        @testset "Triangle sizes: $(sizes)" for sizes in [[3, 6], [3, 3, 3], [6, 10], [3, 6, 10]]
            # Create PSD triangle cones
            psd_tri_cones = [COSMO.PsdConeTriangle(dim) for dim in sizes]
            composite_set = COSMO.CompositeConvexSet(psd_tri_cones)

            # Generate test data for each cone
            x_data = []
            for (dim_idx, dim) in enumerate(sizes)
                n = Int(round((sqrt(8 * dim + 1) - 1) / 2))  # Convert triangle dim to matrix size
                X = randn(rng, n, n)
                X = Symmetric(X - 1.5 * I)

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
                append!(x_data, x_tri)
            end

            # Test projections
            x_single = copy(x_data)
            x_multi = copy(x_data)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_vectors(x_single, x_multi, 1e-12)

            # Verify all results are PSD when reconstructed
            offset = 0
            for (dim_idx, dim) in enumerate(sizes)
                x_single_part = @view x_single[(offset+1):(offset+dim)]
                X_single = COSMO.matrixify(x_single_part)
                @test minimum(eigvals(Symmetric(X_single))) >= -1e-10
                offset += dim
            end
        end
    end

    @testset "Large Composite Sets" begin
        @testset "Many Small PSD Cones" begin
            # Create 10 small PSD cones
            n_cones = 10
            cone_sizes = fill(2, n_cones)  # All 2x2 PSD cones
            psd_cones = [COSMO.PsdCone(4) for _ in 1:n_cones]
            composite_set = COSMO.CompositeConvexSet(psd_cones)

            # Generate test data
            x_data = []
            for _ in 1:n_cones
                X = randn(rng, 2, 2)
                X = Symmetric(X - 1.0 * I)
                append!(x_data, vec(X))
            end

            # Test projections
            x_single = copy(x_data)
            x_multi = copy(x_data)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_vectors(x_single, x_multi, 1e-12)

            # Verify all projections are PSD
            for i in 1:n_cones
                start_idx = (i-1) * 4 + 1
                end_idx = i * 4
                X_single = reshape(@view(x_single[start_idx:end_idx]), 2, 2)
                @test minimum(eigvals(Symmetric(X_single))) >= -1e-10
            end
        end
    end

    @testset "Random Composite Set Configurations" begin
        for test_case in 1:15
            # Randomly generate composite set configuration
            n_psd = rand(rng, 1:3)
            n_tri = rand(rng, 0:2)
            n_other = rand(rng, 0:2)

            cones = []

            # Add PSD cones
            for _ in 1:n_psd
                n = rand(rng, [2, 3, 4])
                push!(cones, COSMO.PsdCone(n * n))
            end

            # Add PSD triangle cones
            for _ in 1:n_tri
                n = rand(rng, [2, 3, 4])
                dim = div(n * (n + 1), 2)
                push!(cones, COSMO.PsdConeTriangle(dim))
            end

            # Add other cones
            for _ in 1:n_other
                cone_type = rand(rng, 1:3)
                if cone_type == 1
                    push!(cones, COSMO.Nonnegatives(rand(rng, 2:5)))
                elseif cone_type == 2
                    push!(cones, COSMO.Box(rand(rng, 2:4)))
                else
                    push!(cones, COSMO.SecondOrderCone(rand(rng, 3:6)))
                end
            end

            composite_set = COSMO.CompositeConvexSet(cones)

            # Generate test data
            x_data = []
            for cone in cones
                if cone isa COSMO.PsdCone
                    n = cone.sqrt_dim
                    X = randn(rng, n, n)
                    X = Symmetric(X - rand(rng) * 2.0 * I)
                    append!(x_data, vec(X))
                elseif cone isa COSMO.PsdConeTriangle
                    n = cone.sqrt_dim
                    X = randn(rng, n, n)
                    X = Symmetric(X - rand(rng) * 2.0 * I)

                    x_tri = zeros(cone.dim)
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
                    append!(x_data, x_tri)
                else
                    append!(x_data, randn(rng, cone.dim))
                end
            end

            # Test projections
            x_single = copy(x_data)
            x_multi = copy(x_data)

            sv_single = COSMO.SplitVector(x_single, composite_set)
            sv_multi = COSMO.SplitVector(x_multi, composite_set)

            force_single_thread_project!(sv_single, composite_set)
            COSMO.project!(sv_multi, composite_set)

            @test compare_vectors(x_single, x_multi, 1e-12)
        end
    end
end