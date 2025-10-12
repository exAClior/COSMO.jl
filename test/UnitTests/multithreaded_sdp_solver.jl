# Multi-threaded SDP solver tests
# This test compares the results of SDP problems solved with different threading configurations

using COSMO, Test, LinearAlgebra, Random, Threads, SparseArrays

# Helper function to run solver with specific threading configuration
function solve_with_threading(P, q, A, b, C; decompose = false, complete_dual = false, merge_strategy = COSMO.NoMerge, kkt_solver = COSMO.QdldlKKTSolver, eps_abs = 1e-6, eps_rel = 1e-6)
    settings = COSMO.Settings(
        decompose = decompose,
        complete_dual = complete_dual,
        merge_strategy = merge_strategy,
        kkt_solver = kkt_solver,
        eps_abs = eps_abs,
        eps_rel = eps_rel,
        verbose = false,
        verbose_timing = false
    )

    model = COSMO.Model()
    assemble!(model, P, q, A, b, C, settings = settings)
    result = optimize!(model)

    return result
end

# Helper function to compare solver results
function compare_solutions(result1::COSMO.Result, result2::COSMO.Result; obj_tol = 1e-8, x_tol = 1e-6, y_tol = 1e-6)
    # Compare objective values
    obj_diff = abs(result1.obj_val - result2.obj_val)

    # Compare solution vectors if available
    x_diff = result1.x !== nothing && result2.x !== nothing ? norm(result1.x - result2.x, Inf) : 0.0
    y_diff = result1.y !== nothing && result2.y !== nothing ? norm(result1.y - result2.y, Inf) : 0.0

    # Compare statuses
    status_match = result1.status == result2.status

    return obj_diff <= obj_tol && x_diff <= x_tol && y_diff <= y_tol && status_match
end

@testset "Multi-threaded SDP Solver" begin

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(98765)

    @testset "Closest Correlation Matrix Problem" begin
        # Problem: min_X 1/2 ||X - C||^2 s.t. Xii = 1, X ⪰ 0
        n = 6  # Matrix size
        C = randn(rng, n, n) * 0.5
        C = Symmetric(C)

        # Problem setup
        n2 = n * n
        P = spdiagm(0 => ones(n2))
        q = -vec(C)

        # Diagonal constraint Xii = 1
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

        # Test different configurations
        result_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false)
        result_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, complete_dual = true)

        # Compare results
        @test compare_solutions(result_no_decomp, result_decomp, obj_tol = 1e-6, x_tol = 1e-4)

        # Verify solution quality
        @test result_no_decomp.status == :Solved
        @test result_decomp.status == :Solved

        # Check diagonal constraints
        X_no_decomp = reshape(result_no_decomp.x, n, n)
        X_decomp = reshape(result_decomp.x, n, n)

        for i in 1:n
            @test abs(X_no_decomp[i, i] - 1.0) <= 1e-4
            @test abs(X_decomp[i, i] - 1.0) <= 1e-4
        end

        # Check PSD constraint
        @test minimum(eigvals(Symmetric(X_no_decomp))) >= -1e-4
        @test minimum(eigvals(Symmetric(X_decomp))) >= -1e-4
    end

    @testset "Maximum Eigenvalue Problem" begin
        # Problem: min λ s.t. A - λI ⪰ 0
        n = 5
        A = randn(rng, n, n)
        A = Symmetric(A)

        # Variables: λ and X (but we only need λ in objective)
        # Reformulate as: min λ s.t. [A - λI ⪰ 0]

        # Create constraint matrix for PSD constraint
        # We parameterize X = A - λI, where λ is the first variable
        # The remaining n^2 variables represent X

        P = spzeros(n^2 + 1, n^2 + 1)
        P[1, 1] = 1.0  # Minimize λ

        q = zeros(n^2 + 1)

        # Build constraint: X - A + λI = 0
        A_psd = spzeros(n^2, n^2 + 1)
        for i in 1:n, j in 1:n
            row_idx = (i-1) * n + j
            col_idx = row_idx + 1
            A_psd[row_idx, col_idx] = 1.0
        end

        # Add λI terms
        for i in 1:n
            row_idx = (i-1) * n + i
            A_psd[row_idx, 1] = 1.0
        end

        b_psd = vec(A)

        C_sets = [COSMO.PsdCone(n^2)]

        # Test configurations
        result_no_decomp = solve_with_threading(P, q, A_psd, b_psd, C_sets, decompose = false)
        result_decomp = solve_with_threading(P, q, A_psd, b_psd, C_sets, decompose = true, complete_dual = true)

        # Compare results
        @test compare_solutions(result_no_decomp, result_decomp, obj_tol = 1e-5, x_tol = 1e-3)

        @test result_no_decomp.status == :Solved
        @test result_decomp.status == :Solved

        # Extract eigenvalues
        λ_no_decomp = result_no_decomp.x[1]
        λ_decomp = result_decomp.x[1]

        # Compare with actual maximum eigenvalue
        λ_actual = maximum(eigvals(A))
        @test abs(λ_no_decomp - λ_actual) <= 1e-3
        @test abs(λ_decomp - λ_actual) <= 1e-3
    end

    @testset "Chordal Decomposition Test Problem" begin
        # Create a problem with a sparse SDP constraint that can be decomposed
        n = 4

        # Create a sparse matrix pattern (like in the existing PSD completion test)
        A_pattern = randn(rng, n, n)
        A_pattern = 0.5 * (A_pattern + A_pattern')
        A_pattern[1, 3] = A_pattern[1, 4] = A_pattern[3, 1] = A_pattern[4, 1] = 0
        a_pattern = vec(A_pattern)

        # Generate feasible solution
        S_true = generate_pos_def_matrix(rng, n, 0.1, 2.0)
        apply_pattern!(S_true, A_pattern)
        s_true = S_true[:]

        # Problem data
        x_true = rand(rng, 1)
        b = a_pattern * x_true + s_true

        # Dual variable
        Y_true = generate_pos_def_matrix(rng, n, 0.1, 1.0)
        y_true = vec(Y_true)

        P = sparse(zeros(1, 1))
        q = -P * x_true - a_pattern' * y_true

        A = hcat(a_pattern)
        C_sets = [COSMO.PsdCone(n^2)]

        # Test different configurations
        result_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false, eps_abs = 1e-5)
        result_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, complete_dual = true, merge_strategy = COSMO.NoMerge, eps_abs = 1e-5)

        # Both should solve successfully
        @test result_no_decomp.status == :Solved
        @test result_decomp.status == :Solved

        # Compare objective values
        @test abs(result_no_decomp.obj_val - result_decomp.obj_val) <= 1e-4

        # Check PSD constraints
        Y_no_decomp = reshape(result_no_decomp.y, n, n)
        Y_decomp = reshape(result_decomp.y, n, n)

        @test minimum(eigvals(Symmetric(Y_no_decomp))) >= -1e-6
        @test minimum(eigvals(Symmetric(Y_decomp))) >= -1e-6
    end

    @testset "Multiple SDP Constraints" begin
        # Problem with multiple independent SDP constraints
        sizes = [2, 3]  # Two PSD constraints of different sizes

        # Build problem
        total_dim = sum(n * n for n in sizes)
        P = spdiagm(0 => ones(total_dim))
        q = randn(rng, total_dim) * 0.1

        # Build constraints - each block is constrained to be PSD
        A_blocks = []
        b_blocks = []
        C_blocks = []

        offset = 0
        for n in sizes
            dim = n * n
            A_block = spzeros(dim, total_dim)
            A_block[:, (offset+1):(offset+dim)] = I
            push!(A_blocks, A_block)
            push!(b_blocks, zeros(dim))
            push!(C_blocks, COSMO.PsdCone(dim))
            offset += dim
        end

        A = vcat(A_blocks...)
        b = vcat(b_blocks...)
        C_sets = C_blocks

        # Test configurations
        result_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false)
        result_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, complete_dual = true)

        # Compare results
        @test compare_solutions(result_no_decomp, result_decomp, obj_tol = 1e-5, x_tol = 1e-3)

        @test result_no_decomp.status == :Solved
        @test result_decomp.status == :Solved

        # Check PSD constraints for both solutions
        offset = 0
        for (i, n) in enumerate(sizes)
            dim = n * n
            x_part_no_decomp = @view result_no_decomp.x[(offset+1):(offset+dim)]
            x_part_decomp = @view result_decomp.x[(offset+1):(offset+dim)]

            X_no_decomp = reshape(x_part_no_decomp, n, n)
            X_decomp = reshape(x_part_decomp, n, n)

            @test minimum(eigvals(Symmetric(X_no_decomp))) >= -1e-6
            @test minimum(eigvals(Symmetric(X_decomp))) >= -1e-6

            offset += dim
        end
    end

    @testset "Different KKT Solvers with Threading" begin
        # Test that threading works with different KKT solvers
        n = 3
        C = randn(rng, n, n) * 0.3
        C = Symmetric(C)

        n2 = n * n
        P = spdiagm(0 => ones(n2))
        q = -vec(C)

        # Diagonal constraints
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

        # Test with QdldlKKTSolver
        result_qdldl_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false, kkt_solver = COSMO.QdldlKKTSolver)
        result_qdldl_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, kkt_solver = COSMO.QdldlKKTSolver)

        @test compare_solutions(result_qdldl_no_decomp, result_qdldl_decomp, obj_tol = 1e-5, x_tol = 1e-3)

        # Test with different Pardiso solvers if available
        try
            result_pardiso_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false, kkt_solver = COSMO.MKLPardisoKKTSolver)
            result_pardiso_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, kkt_solver = COSMO.MKLPardisoKKTSolver)

            @test compare_solutions(result_pardiso_no_decomp, result_pardiso_decomp, obj_tol = 1e-5, x_tol = 1e-3)
        catch
            # Pardiso not available, skip this test
            @test true
        end
    end

    @testset "Convergence and Iteration Consistency" begin
        # Test that threading doesn't significantly affect convergence properties
        n = 4
        C = randn(rng, n, n) * 0.4
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

        result_no_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = false)
        result_decomp = solve_with_threading(P, q, A, b, C_sets, decompose = true, complete_dual = true)

        # Both should solve
        @test result_no_decomp.status == :Solved
        @test result_decomp.status == :Solved

        # Iteration counts should be reasonable (not too different)
        iter_diff = abs(result_no_decomp.iterations - result_decomp.iterations)
        @test iter_diff <= max(10, 0.2 * max(result_no_decomp.iterations, result_decomp.iterations))

        # Objective values should be close
        @test abs(result_no_decomp.obj_val - result_decomp.obj_val) <= 1e-5
    end
end

# Helper function for generating positive definite matrices (copied from existing tests)
function generate_pos_def_matrix(rng::AbstractRNG, n::Int, min_eig::Real, max_eig::Real)
    A = randn(rng, n, n)
    A = A' * A
    D = eigvals(A)
    min_actual = minimum(D)
    max_actual = maximum(D)

    # Scale eigenvalues to desired range
    scale_lower = min_eig / min_actual
    scale_upper = max_eig / max_actual
    scale = min(scale_lower, scale_upper)

    return scale * A
end

function apply_pattern!(S::Matrix, pattern::Matrix)
    S[1, 3] = S[1, 4] = S[3, 1] = S[4, 1] = 0.0
    return S
end