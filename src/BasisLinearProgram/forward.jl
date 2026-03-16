# Copyright (c) 2020: Akshay Sharma and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# ============================================================================
# GeneralModel forward differentiation
# ============================================================================

function DiffOpt.forward_differentiate!(model::GeneralModel)
    model.diff_time = @elapsed begin
        _build_A!(model)
        _ensure_basis!(model)
        m = length(model.ci_list)

        # 1. Build db from input_cache (1-based row indexing)
        db = zeros(m)
        for (F, S) in keys(model.input_cache.scalar_constraints.dict)
            for (ci, func) in model.input_cache.scalar_constraints[F, S]
                if !isempty(func.terms)
                    error(
                        "BasisLinearProgram: constraint coefficient " *
                        "perturbation (dA) is not supported.",
                    )
                end
                row = model.ci_to_row[ci]
                db[row] = -MOI.constant(func)
            end
        end

        # 2. Solve B * dx_B = db
        dx_B = model.B_lu \ db

        # 3. Map basic structural variables back to MOI VIs
        model.forw_dx = Dict{MOI.VariableIndex,Float64}()
        for (k, col) in enumerate(model.basic_structural)
            model.forw_dx[model.vi_list[col]] = dx_B[k]
        end

        # 4. Dual sensitivity: dλ = B⁻ᵀ dc_B
        model.forw_dy = nothing
        if model.input_cache.objective !== nothing
            dc_B = zeros(m)
            for term in model.input_cache.objective.terms
                k = get(model.col_to_basic, get(model.vi_to_col, term.variable, 0), 0)
                k > 0 && (dc_B[k] = term.coefficient)
            end
            dλ = model.B_lu' \ dc_B
            model.forw_dy = Dict{MOI.ConstraintIndex,Float64}(
                ci => dλ[i] for (i, ci) in enumerate(model.ci_list)
            )
        end
    end
    return
end
