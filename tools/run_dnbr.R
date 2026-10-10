# Call the pinned upstream DNBr package on reference-scaled development data.

read_json <- function(path) {
    # PSEUDOCODE: read a JSON object without collapsing its named structures.
    jsonlite::fromJSON(path, simplifyVector = FALSE)
}

install_dnbr <- function(settings_path, library_path) {
    # PSEUDOCODE: install into an explicit project library and pin the upstream commit.
    dir.create(library_path, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(normalizePath(library_path), .libPaths()))
    options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 600)
    required <- c("jsonlite", "digest", "remotes")
    missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing)) utils::install.packages(missing, lib = library_path)
    settings <- read_json(settings_path)
    remotes::install_github(paste0(settings$repository, "@", settings$revision),
                           lib = library_path, dependencies = NA, upgrade = "never",
                           build_vignettes = FALSE)
}

extract_rows <- function(object, stages, slot_name) {
    # PSEUDOCODE: extract upstream results by explicit stage, preserving empty stages.
    tables <- lapply(stages, function(stage) {
        result <- methods::slot(object[[stage]], slot_name)
        if (!length(result@resource)) return(NULL)
        rows <- DNBr::resultAllExtract(object, group = stage, slot = slot_name, mess = FALSE)
        rows$stage <- stage
        rows
    })
    tables <- Filter(Negate(is.null), tables)
    if (!length(tables)) return(data.frame())
    do.call(rbind, tables)
}

analyse_dnbr <- function(input_dir, output_dir, library_path) {
    # PSEUDOCODE: validate provenance and complete inputs -> call upstream discovery/ranking -> save descriptive results.
    if (file.exists(output_dir)) stop("Output already exists; preserve previous experiments.")
    .libPaths(c(normalizePath(library_path, mustWork = TRUE), .libPaths()))
    for (package in c("jsonlite", "digest", "DNBr")) {
        if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
    }
    manifest <- read_json(file.path(input_dir, "manifest.json"))
    for (name in names(manifest$files)) {
        if (basename(name) != name || grepl("[/\\\\]", name)) stop("Invalid input file name.")
        if (digest::digest(file = file.path(input_dir, name), algo = "sha256") != manifest$files[[name]])
            stop("Input hash mismatch: ", name)
    }
    settings <- read_json(file.path(input_dir, "settings.json"))
    description <- utils::packageDescription("DNBr")
    if (settings$repository != "Kaiyu-W/DNBr" || is.null(description$RemoteSha) ||
        description$RemoteSha != settings$revision) stop("DNBr commit is not the pinned source.")
    if (manifest$purpose != "development_analysis_not_validated_warning_model" ||
        settings$purpose != manifest$purpose) stop("Unexpected analysis purpose.")
    stages <- unlist(manifest$stages, use.names = FALSE)
    if (!identical(stages, c("stable", "pre_event"))) stop("Unexpected development stages.")
    data <- utils::read.csv(file.path(input_dir, "matrix.csv"), row.names = 1,
                            check.names = FALSE, fileEncoding = "UTF-8")
    samples <- utils::read.csv(file.path(input_dir, "samples.csv"), check.names = FALSE,
                               colClasses = "character", fileEncoding = "UTF-8")
    if (!all(vapply(data, is.numeric, logical(1)))) stop("All feature values must be numeric.")
    data <- as.matrix(data)
    if (!identical(rownames(data), unlist(manifest$features, use.names = FALSE)) ||
        !identical(colnames(data), samples$record_id) || anyDuplicated(samples$record_id) ||
        anyDuplicated(rownames(data)) || !all(is.finite(data))) stop("Invalid matrix alignment or values.")
    if (!all(samples$split == "train") || !setequal(unique(samples$stage), stages) ||
        anyDuplicated(samples[c("participant_id", "stage")])) stop("Invalid development cohort.")
    people <- lapply(stages, function(stage) sort(samples$participant_id[samples$stage == stage]))
    if (!identical(people[[1]], people[[2]]) || length(people[[1]]) < 9)
        stop("Discovery requires the same independent people at both stages.")
    if (ncol(data) != manifest$observations || length(people[[1]]) != manifest$people)
        stop("Cohort size differs from the input manifest.")
    for (stage in stages) {
        sd <- apply(data[, samples$stage == stage, drop = FALSE], 1, stats::sd)
        if (any(!is.finite(sd) | sd <= 1e-12)) stop("Undefined/constant feature in stage: ", stage)
    }
    width <- nrow(data)
    lower <- settings$module_min_inclusive
    upper <- if (is.null(settings$module_max_inclusive)) width - 1 else settings$module_max_inclusive
    if (length(lower) != 1 || length(upper) != 1 || lower != as.integer(lower) ||
        upper != as.integer(upper) || lower < 2 || upper >= width || lower > upper)
        stop("Modules need at least two internal and one external feature.")
    if (settings$high_cutoff != -1 || !isTRUE(settings$force_allgene))
        stop("The rhythm panel must remain fixed across stages.")
    if (!is.logical(settings$size_effect) || length(settings$size_effect) != 1 ||
        settings$ntop < 1 || settings$ntop != as.integer(settings$ntop)) stop("Invalid ranking settings.")
    if (settings$cutree_method != "h" || !is.numeric(settings$cutree_cutoff) ||
        length(settings$cutree_cutoff) != 1 || !is.finite(settings$cutree_cutoff) ||
        settings$cutree_cutoff <= 0 || settings$cutree_cutoff > 1) stop("Invalid clustering height.")
    meta <- data.frame(stage = samples$stage, row.names = samples$record_id)
    dir.create(output_dir, recursive = TRUE)
    saveRDS(list(manifest = manifest, settings = settings, samples = samples),
            file.path(output_dir, "inputs.rds"))
    writeLines(capture.output(sessionInfo()), file.path(output_dir, "session.txt"))
    # DNBr compares strict module bounds; convert our inclusive bounds explicitly.
    raw <- DNBr::DNBcompute(data, meta, meta_levels = stages, high_cutoff = -1,
                           cutree_method = settings$cutree_method, cutree_cutoff = settings$cutree_cutoff,
                           minModule = lower - 1, maxModule = upper + 1,
                           size_effect = settings$size_effect, quiet = TRUE)
    saveRDS(raw, file.path(output_dir, "discovery.rds"))
    candidates <- extract_rows(raw, stages, "pre_result")
    utils::write.csv(candidates, file.path(output_dir, "candidates.csv"), row.names = FALSE)
    scores <- data.frame()
    if (nrow(candidates)) {
        filtered <- DNBr::DNBfilter(raw, ntop = settings$ntop, force_allgene = TRUE,
                                   size_effect = settings$size_effect, quiet = TRUE)
        saveRDS(filtered, file.path(output_dir, "ranked.rds"))
        scores <- extract_rows(filtered, stages, "result")
    }
    if (nrow(scores)) {
        module_size <- lengths(strsplit(as.character(scores$genes), ",", fixed = TRUE))
        scores$usable <- scores$bestMODULE & module_size >= lower & module_size <= upper &
                         is.finite(scores$SCORE) & is.finite(scores$SD) & scores$SD > 0 &
                         is.finite(scores$PCC_IN) & is.finite(scores$PCC_OUT) & scores$PCC_OUT > 0
        scores$warning <- NA_integer_
        # Keep zero-denominator and ineligible upstream rows visible, never treat them as no-warning.
        scores$status <- ifelse(scores$usable, "candidate_requires_calibration", "unscorable_or_ineligible")
        if (any(scores$usable)) {
            figure <- ggplot2::ggplot(scores[scores$usable, ],
                      ggplot2::aes(x = factor(stage, levels = stages), y = SCORE,
                                   group = resource, colour = resource)) +
                      ggplot2::geom_line() + ggplot2::geom_point() +
                      ggplot2::labs(x = "Development stage", y = "DNBr score", colour = "Module") +
                      ggplot2::theme_minimal()
            ggplot2::ggsave(file.path(output_dir, "development-scores.pdf"), figure, width = 8, height = 5)
        }
    }
    utils::write.csv(scores, file.path(output_dir, "scores.csv"), row.names = FALSE)
    result <- list(status = if (nrow(scores) && any(scores$usable)) "development_candidates_only" else "no_usable_modules",
                   upstream_revision = settings$revision, settings = settings,
                   input_id = manifest$id, domain = manifest$domain,
                   warning_policy_calibrated = FALSE, warning_accuracy = NULL,
                   note = "Group-stage scores; not individual forecasts or replication of an upstream paper.")
    jsonlite::write_json(result, file.path(output_dir, "result.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
    files <- list.files(output_dir, full.names = TRUE)
    hashes <- vapply(files, function(path) digest::digest(file = path, algo = "sha256"), character(1))
    jsonlite::write_json(as.list(setNames(hashes, basename(files))),
                        file.path(output_dir, "hashes.json"), auto_unbox = TRUE, pretty = TRUE)
    invisible(result)
}

main <- function(args) {
    # PSEUDOCODE: separate dependency installation from analysis; never start GPU work here.
    if (length(args) == 3 && args[1] == "install") {
        install_dnbr(args[2], args[3])
    } else if (length(args) == 4 && args[1] == "analyze") {
        # Record a failed fresh run; never overwrite an existing experiment's status.
        was_present <- file.exists(args[3])
        tryCatch(analyse_dnbr(args[2], args[3], args[4]), error = function(error) {
            if (!was_present && dir.exists(args[3]))
                writeLines(conditionMessage(error), file.path(args[3], "failure.txt"))
            stop(error)
        })
    } else {
        stop(paste("Usage: Rscript tools/run_dnbr.R install configs/dnbr.json <R-library>",
                   "or: Rscript tools/run_dnbr.R analyze <export-dir> <new-output-dir> <R-library>", sep = "\n"))
    }
}

if (sys.nframe() == 0) main(commandArgs(trailingOnly = TRUE))
