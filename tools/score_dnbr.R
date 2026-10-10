# Score people with DNBr kernels and the published single-sample DNB index.

read_matrix <- function(path) {
    # PSEUDOCODE: preserve feature/sample identities and reject ambiguous numeric inputs.
    data <- read.csv(path, row.names = 1, check.names = FALSE)
    if (!all(vapply(data, is.numeric, logical(1)))) stop("Non-numeric matrix.")
    data <- as.matrix(data)
    if (anyDuplicated(rownames(data)) || anyDuplicated(colnames(data)) ||
        any(!is.finite(data))) stop("Nonfinite or duplicate matrix entries.")
    data
}

single_components <- function(target, reference, module, epsilon) {
    # PSEUDOCODE: use upstream deviation/network kernels -> Eq. 6 averages -> guarded upstream CI.
    if (!identical(names(target), rownames(reference)) || !all(module %in% names(target)) ||
        anyDuplicated(module) || length(module) < 2 || length(module) >= nrow(reference))
        stop("Invalid fixed module or feature order.")
    if (ncol(reference) < 9 || any(!is.finite(reference)) || any(!is.finite(target)) ||
        any(apply(reference, 1, sd) <= 1e-12)) stop("Invalid reference or target.")
    base <- DNBr::NetworkCreate(reference)$r
    combined <- DNBr::NetworkCreate(cbind(reference, target))$r
    delta <- combined - base
    diag(delta) <- 0
    if (any(!is.finite(delta))) stop("Undefined network perturbation.")
    sed <- getFromNamespace("sED", "DNBr")
    amplitude <- mean(vapply(module, function(name)
        sed(target[name], reference[name, ], scale = FALSE), numeric(1)))
    internal <- mean(abs(delta[module, module, drop = FALSE]))
    external <- mean(abs(delta[module, setdiff(rownames(reference), module), drop = FALSE]))
    valid <- is.finite(external) && external > epsilon
    value <- if (valid) getFromNamespace("CI", "DNBr")(
        n = 1, Sd = amplitude, pcc_in = internal, pcc_out = external) else NA_real_
    valid <- valid && is.finite(value)
    list(sED_in = amplitude, sPCC_in = internal, sPCC_out = external,
         score = if (valid) value else NA_real_, valid = valid,
         reason = if (valid) "ok" else "near_zero_or_nonfinite_external_perturbation")
}

score_dnbr <- function(input_dir, output_dir, library_path) {
    # PSEUDOCODE: verify sealed inputs -> score each person independently -> save scores and every component.
    if (file.exists(output_dir)) stop("Preserve previous outputs; choose a new directory.")
    .libPaths(c(normalizePath(library_path, mustWork = TRUE), .libPaths()))
    input <- jsonlite::fromJSON(file.path(input_dir, "manifest.json"), simplifyVector = FALSE)
    for (name in names(input$files)) {
        if (basename(name) != name || grepl("[/\\\\]", name) ||
            digest::digest(file = file.path(input_dir, name), algo = "sha256") != input$files[[name]])
            stop("Scoring input hash mismatch: ", name)
    }
    if (utils::packageDescription("DNBr")$RemoteSha != input$upstream_revision)
        stop("Unexpected upstream commit.")
    if (input$pair_convention != "paper_k_squared" || input$sdnb_size_multiplier != 1 ||
        !is.numeric(input$epsilon) || input$epsilon <= 0) stop("Invalid sDNB settings.")
    reference <- read_matrix(file.path(input_dir, "reference.csv"))
    targets <- read_matrix(file.path(input_dir, "targets.csv"))
    if (!identical(rownames(reference), rownames(targets)) ||
        length(intersect(colnames(reference), colnames(targets)))) stop("Reference/target alignment error.")
    if (ncol(reference) < 9 || any(apply(reference, 1, sd) <= 1e-12)) stop("Invalid reference.")
    modules <- input$modules
    scores <- list(); components <- list()
    for (record in colnames(targets)) {
        parts <- lapply(seq_along(modules), function(i) {
            part <- single_components(targets[, record], reference, unlist(modules[[i]]), input$epsilon)
            as.data.frame(c(list(record_id = record, module_id = i), part))
        })
        valid <- length(parts) > 0 && all(vapply(parts, function(p) p$valid, logical(1)))
        scores[[record]] <- data.frame(record_id = record,
            score = if (valid) max(vapply(parts, function(p) p$score, numeric(1))) else NA_real_,
            status = if (valid) "ok" else if (!length(parts)) "no_development_module" else "invalid_module")
        components <- c(components, parts)
    }
    dir.create(output_dir, recursive = TRUE)
    write.csv(do.call(rbind, scores), file.path(output_dir, "scores.csv"), row.names = FALSE, na = "")
    write.csv(if (length(components)) do.call(rbind, components) else data.frame(),
              file.path(output_dir, "components.csv"), row.names = FALSE, na = "")
    writeLines(capture.output(sessionInfo()), file.path(output_dir, "session.txt"))
    jsonlite::write_json(list(input_sha256 = digest::digest(file = file.path(input_dir, "manifest.json"), algo = "sha256"),
        upstream_revision = input$upstream_revision, records = ncol(targets), modules = length(modules),
        valid = sum(vapply(scores, function(x) x$status == "ok", logical(1))),
        method = "DNBr development modules + Liu2017 fixed-module sDNB adapter"),
        file.path(output_dir, "receipt.json"), pretty = TRUE, auto_unbox = TRUE)
}

if (sys.nframe() == 0) {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) != 3) stop("Usage: Rscript tools/score_dnbr.R <input> <new-output> <R-library>")
    score_dnbr(args[1], args[2], args[3])
}
