# Discover each current sample's candidate modules, using unchanged DNBr scoring kernels.

personal_branches <- function(delta) {
    # PSEUDOCODE: form a single-linkage tree -> retain every branch except singleton leaves and the full universe.
    distance <- 2 - abs(delta)
    diag(distance) <- 0
    tree <- hclust(as.dist(distance), method = "single")
    branches <- list()
    for (i in seq_len(nrow(tree$merge))) {
        branch <- unlist(lapply(tree$merge[i, ], function(index)
            if (index < 0) rownames(delta)[-index] else branches[[index]]))
        branches[[i]] <- sort(branch)
    }
    branches[-length(branches)]
}

score_personal <- function(input_dir, output_dir, library_path) {
    # PSEUDOCODE: verify identities -> discover using current vector only -> save every candidate and its winning score.
    script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
    source(file.path(dirname(script), "score_dnbr.R"))
    if (file.exists(output_dir)) stop("Preserve previous results.")
    .libPaths(c(normalizePath(library_path, mustWork = TRUE), .libPaths()))
    input <- jsonlite::fromJSON(file.path(input_dir, "manifest.json"), simplifyVector = FALSE)
    for (name in names(input$files)) {
        if (basename(name) != name || grepl("[/\\\\]", name) ||
            digest::digest(file = file.path(input_dir, name), algo = "sha256") != input$files[[name]])
            stop("Input identity mismatch.")
    }
    if (utils::packageDescription("DNBr")$RemoteSha != input$upstream_revision ||
        input$pair_convention != "paper_k_squared" || input$sdnb_size_multiplier != 1 ||
        !is.numeric(input$epsilon) || input$epsilon <= 0) stop("Unexpected kernel settings.")
    reference <- read_matrix(file.path(input_dir, "reference.csv"))
    targets <- read_matrix(file.path(input_dir, "targets.csv"))
    if (!identical(rownames(reference), rownames(targets)) || nrow(reference) < 3 || ncol(reference) < 9 ||
        any(grepl(",", rownames(reference))) || any(apply(reference, 1, sd) <= 1e-12) ||
        length(intersect(colnames(reference), colnames(targets)))) stop("Invalid reference or target identities.")
    ordering <- order(rownames(reference))
    reference <- reference[ordering, , drop = FALSE]; targets <- targets[ordering, , drop = FALSE]
    base <- DNBr::NetworkCreate(reference)$r
    components <- list(); scores <- list()
    for (record in colnames(targets)) {
        target <- targets[, record]
        delta <- DNBr::NetworkCreate(cbind(reference, target))$r - base
        diag(delta) <- 0
        if (any(!is.finite(delta))) stop("Undefined network perturbation.")
        modules <- personal_branches(delta)
        parts <- lapply(seq_along(modules), function(i) {
            part <- single_components(target, reference, modules[[i]], input$epsilon)
            as.data.frame(c(list(record_id = record, module_id = i,
                                genes = paste(modules[[i]], collapse = ",")), part))
        })
        valid <- length(parts) > 0 && all(vapply(parts, function(x) x$valid, logical(1)))
        best <- if (valid) which.max(vapply(parts, function(x) x$score, numeric(1))) else NA_integer_
        scores[[record]] <- data.frame(record_id = record,
            score = if (valid) parts[[best]]$score else NA_real_,
            status = if (valid) "ok" else "invalid_module", modules = length(parts),
            winning_genes = if (valid) parts[[best]]$genes else "")
        components <- c(components, parts)
    }
    dir.create(output_dir, recursive = TRUE)
    write.csv(do.call(rbind, scores), file.path(output_dir, "scores.csv"), row.names = FALSE, na = "")
    write.csv(do.call(rbind, components), file.path(output_dir, "components.csv"), row.names = FALSE, na = "")
    writeLines(capture.output(sessionInfo()), file.path(output_dir, "session.txt"))
    jsonlite::write_json(list(input_sha256 = digest::digest(file = file.path(input_dir, "manifest.json"), algo = "sha256"),
        upstream_revision = input$upstream_revision, records = ncol(targets),
        method = "Exploratory continuous per-target tree branches; not significance-certified DNB"),
        file.path(output_dir, "receipt.json"), auto_unbox = TRUE, pretty = TRUE)
}

if (sys.nframe() == 0) {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) != 3) stop("Usage: Rscript score_dnbr_personal.R <input> <new-output> <R-library>")
    score_personal(args[1], args[2], args[3])
}
