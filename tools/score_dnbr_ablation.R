# Audit the network contribution; keep the original DNBr experiment untouched.

score_ablation <- function(input_dir, output_dir, library_path) {
    # PSEUDOCODE: reproduce the original score, then replace only its differential-correlation scale.
    script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
    source(file.path(dirname(script), "score_dnbr.R"))
    score_dnbr(input_dir, output_dir, library_path)
    input <- jsonlite::fromJSON(file.path(input_dir, "manifest.json"), simplifyVector = FALSE)
    reference <- read_matrix(file.path(input_dir, "reference.csv"))
    targets <- read_matrix(file.path(input_dir, "targets.csv"))
    base <- DNBr::NetworkCreate(reference)$r
    denominator <- 1 - base^2
    diag(denominator) <- 1
    if (any(denominator <= input$epsilon)) stop("Near-perfect reference correlation; Z is undefined.")
    parts <- read.csv(file.path(output_dir, "components.csv"), check.names = FALSE)
    parts$z_in <- NA_real_; parts$z_out <- NA_real_; parts$z_score <- NA_real_
    for (record in colnames(targets)) {
        delta <- DNBr::NetworkCreate(cbind(reference, targets[, record]))$r - base
        z <- delta * (ncol(reference) - 1) / denominator
        diag(z) <- 0
        for (i in seq_along(input$modules)) {
            module <- unlist(input$modules[[i]])
            inside <- mean(abs(z[module, module, drop = FALSE]))
            outside <- mean(abs(z[module, setdiff(rownames(reference), module), drop = FALSE]))
            row <- which(parts$record_id == record & parts$module_id == i)
            if (length(row) != 1) stop("Missing or repeated component.")
            parts$z_in[row] <- inside; parts$z_out[row] <- outside
            if (is.finite(outside) && outside > input$epsilon)
                parts$z_score[row] <- parts$sED_in[row] * inside / outside
        }
    }
    write.csv(parts, file.path(output_dir, "ablation-components.csv"), row.names = FALSE, na = "")
    # PSEUDOCODE: audit eligibility for the author's landscape reference graph, without relaxing its gate.
    p <- nrow(reference); df <- ncol(reference) - 2
    probability <- 2 * pt(-abs(base * sqrt(df / denominator)), df)
    adjacent <- probability < 0.01 / p^2
    diag(adjacent) <- FALSE
    jsonlite::write_json(list(features = p, reference_people = ncol(reference),
        correlation_edges = sum(adjacent) / 2, centers_with_three_neighbors = sum(rowSums(adjacent) >= 3),
        cutoff = 0.01 / p^2, interpretation = "Reference-graph eligibility only; not l-DNB warning accuracy.",
        author_revision = "01cddcf7b124f76da4fcefca1ec0049a319a4755"),
        file.path(output_dir, "landscape-eligibility.json"), pretty = TRUE, auto_unbox = TRUE)
}

if (sys.nframe() == 0) {
    args <- commandArgs(trailingOnly = TRUE)
    if (length(args) != 3) stop("Usage: Rscript score_dnbr_ablation.R <input> <new-output> <R-library>")
    score_ablation(args[1], args[2], args[3])
}
