            return reject(error, "invalid Lennard-Jones error buffer");
        if (!opaque || !input || !output)
            return reject(error, "Lennard-Jones requires context, input and output");
        if (input->abi_version != PYRPMD_PES_ABI_VERSION_V2 ||
            output->abi_version != PYRPMD_PES_ABI_VERSION_V2 ||
            input->struct_size != static_cast<int32_t>(sizeof(*input)) ||
            output->struct_size != static_cast<int32_t>(sizeof(*output)))
            return reject(error, "Lennard-Jones ABI version or structure size mismatch",
                          PYRPMD_ABI_MISMATCH_V2);
        auto& context = *reinterpret_cast<LJContext*>(opaque);
        const int64_t batches = input->nstructures, natoms = input->natoms;
        if (batches < 1 || natoms < 1 || natoms > INT64_MAX / 3 ||
            batches > INT64_MAX / (3 * natoms) || batches > INT64_MAX / 9 ||
            input->layout != PYRPMD_LAYOUT_BNC_V2 ||
            input->scalar_type != PYRPMD_SCALAR_FLOAT64_V2)
            return reject(error, "invalid Lennard-Jones dimensions or layout");
        const int64_t dimensions = 3 * natoms, count = batches * dimensions;
        if (static_cast<uint64_t>(count) >
                static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(double) ||
            input->positions_count != count || input->cells_count != 9 * batches ||
            input->pbc_count != 3 * batches || output->energies_count != batches ||
            output->forces_count != count || output->structure_status_count != batches ||
            !input->positions || !input->cells || !input->pbc || !output->energies ||
            !output->forces || !output->structure_status)
            return reject(error, "invalid Lennard-Jones buffers");
        if (output->virials) {
            if (output->virials_count != 9 * batches)
                return reject(error, "invalid Lennard-Jones virial buffer");
        } else if (output->virials_count) {
            return reject(error, "omitted Lennard-Jones virials require zero size");
        }
        if (output->structure_messages) {
            const int64_t stride = output->structure_message_stride;
            if (stride < 1 || batches > INT64_MAX / stride ||
                output->structure_messages_capacity < batches * stride)
                return reject(error, "invalid Lennard-Jones message buffer");
        } else if (output->structure_message_stride || output->structure_messages_capacity) {
            return reject(error, "omitted Lennard-Jones messages require zero sizes");
        }
        context.frames.resize(static_cast<size_t>(batches));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        #pragma omp parallel for schedule(static) if(count >= 4096)
        for (int64_t batch = 0; batch < batches; ++batch) {
            const double* q = input->positions + batch * dimensions;
            const double* cell = input->cells + batch * 9;
            double* force = output->forces + batch * dimensions;
            double* virial = output->virials ? output->virials + batch * 9 : nullptr;
            int32_t status = PYRPMD_STRUCTURE_OK_V2;
            const char* diagnostic = "";
            double energy = 0., inverse[9];
            std::fill(force, force + dimensions, 0.);
            if (virial) std::fill(virial, virial + 9, 0.);
            try {
            for (int64_t index = 0; index < dimensions; ++index)
                if (!std::isfinite(q[index])) status = PYRPMD_STRUCTURE_NONFINITE_V2;
            for (int axis = 0; axis < 3; ++axis)
                if (input->pbc[3 * batch + axis] != 1)
                    status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
            if (status == PYRPMD_STRUCTURE_NONFINITE_V2)
                diagnostic = "Lennard-Jones coordinates contain NaN/Inf";
            else if (status)
                diagnostic = "Lennard-Jones requires three periodic axes";
            else if (!invert_cell(cell, inverse)) {
                status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
                diagnostic = "Lennard-Jones cell is singular, nonfinite or ill conditioned";
            } else if (!valid_cutoff_cell(inverse, context.cutoff)) {
                status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
                diagnostic = "Lennard-Jones cutoff exceeds half the shortest cell face height";
            }
            LJFrameCache& cache = context.frames[static_cast<size_t>(batch)];
            if (!status && rebuild_needed(cache, q, natoms, cell, inverse, context.skin) &&
                !rebuild(cache, q, natoms, cell, inverse, context.list_cutoff2)) {
                status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
                diagnostic = "cell is too skew for a bounded Lennard-Jones minimum-image search";
            }
            if (!status) for (const auto& pair : cache.pairs) {
                const int64_t i = pair.first, j = pair.second;
                double delta[3], image[3];
                for (int cart = 0; cart < 3; ++cart)
                    delta[cart] = q[3 * j + cart] - q[3 * i + cart];
                if (!minimum_image(delta, cell, inverse, image)) {
                    status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
                    diagnostic = "cell is too skew for a bounded Lennard-Jones minimum-image search";
                    break;
                }
                const double distance2 = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
                if (distance2 <= std::numeric_limits<double>::min()) {
                    status = PYRPMD_STRUCTURE_OUT_OF_DOMAIN_V2;
                    diagnostic = "coincident Lennard-Jones atoms";
                    break;
                }
                if (distance2 >= context.cutoff2) continue;
                const double inverse_r2 = 1. / distance2;
                const double sr2 = context.sigma * context.sigma * inverse_r2;
                const double sr6 = sr2 * sr2 * sr2;
                const double sr12 = sr6 * sr6;
                energy += 4. * context.epsilon * (sr12 - sr6) - context.energy_shift;
                const double factor = 24. * context.epsilon * (2. * sr12 - sr6) * inverse_r2;
                for (int cart = 0; cart < 3; ++cart) {
                    const double pair_force = factor * image[cart];
                    force[3 * i + cart] -= pair_force;
                    force[3 * j + cart] += pair_force;
                    if (virial) for (int other = 0; other < 3; ++other)
                        virial[3 * other + cart] += image[other] * pair_force;
                }
            }
            if (!status) {
                if (!std::isfinite(energy)) status = PYRPMD_STRUCTURE_NONFINITE_V2;
                for (int64_t index = 0; index < dimensions; ++index)
                    if (!std::isfinite(force[index])) status = PYRPMD_STRUCTURE_NONFINITE_V2;
                if (virial) for (int index = 0; index < 9; ++index)
                    if (!std::isfinite(virial[index])) status = PYRPMD_STRUCTURE_NONFINITE_V2;
                if (status) diagnostic = "nonfinite Lennard-Jones energy, forces or virial";
            }
            } catch (const std::bad_alloc&) {
                status = PYRPMD_STRUCTURE_FAILED_V2;
                diagnostic = "Lennard-Jones neighbor-list allocation failed";
            } catch (...) {
                status = PYRPMD_STRUCTURE_FAILED_V2;
                diagnostic = "unexpected Lennard-Jones frame failure";
            }
            if (status) {
                energy = nan;
                std::fill(force, force + dimensions, nan);
                if (virial) std::fill(virial, virial + 9, nan);
            }
            output->energies[batch] = energy;
            output->structure_status[batch] = status;
            if (output->structure_messages)
                std::snprintf(output->structure_messages + batch * output->structure_message_stride,
                              static_cast<size_t>(output->structure_message_stride), "%s", diagnostic);
        }
        message(error, "");
        return PYRPMD_OK_V2;
    } catch (const std::bad_alloc&) {
        return reject(error, "Lennard-Jones allocation failed", PYRPMD_INTERNAL_ERROR_V2);
    } catch (...) {
        return reject(error, "unexpected Lennard-Jones native failure", PYRPMD_INTERNAL_ERROR_V2);
    }
}

constexpr const char* lj_capsule_name = "pyrpmd.BuiltinLJ.v2";

inline void destroy_lj(PyObject* capsule) {
    auto* context = static_cast<LJContext*>(PyCapsule_GetPointer(capsule, lj_capsule_name));
    if (context) delete context;
    else PyErr_Clear();
}

inline PyObject* create_lj(PyObject*, PyObject* args) {
    double epsilon, sigma, cutoff, skin;
    if (!PyArg_ParseTuple(args, "dddd", &epsilon, &sigma, &cutoff, &skin)) return nullptr;
    try {
        if (!std::isfinite(epsilon) || !std::isfinite(sigma) ||
            !std::isfinite(cutoff) || !std::isfinite(skin) ||
            epsilon <= 0. || sigma <= 0. || cutoff <= 0. || skin < 0. ||
            !std::isfinite(cutoff + skin))
            throw std::invalid_argument("invalid periodic Lennard-Jones parameters");
        auto context = std::make_unique<LJContext>(epsilon, sigma, cutoff, skin);
        if (!std::isfinite(context->cutoff2) || !std::isfinite(context->list_cutoff2) ||
            !std::isfinite(context->energy_shift))
            throw std::invalid_argument("periodic Lennard-Jones parameters overflow");
        PyObject* capsule = PyCapsule_New(context.get(), lj_capsule_name, destroy_lj);
        if (!capsule) return nullptr;
        LJContext* pointer = context.release();
        PyObject* result = Py_BuildValue("OKK", capsule,
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&pyrpmd_builtin_lj_evaluate_v2)),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(pointer)));
        Py_DECREF(capsule);
        return result;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& exception) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }
}

} // namespace builtin_pes
