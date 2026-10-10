# Independent arithmetic and degeneracy checks for the R single-sample adapter.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(normalizePath(args[1]), .libPaths()))
source(args[2])
set.seed(811)
reference <- matrix(rnorm(60 * 5), nrow = 5, dimnames = list(letters[1:5], paste0("ref", 1:60)))
target <- setNames(c(2, -3, 1, .2, -.1), letters[1:5])
module <- c("a", "b", "d")
part <- single_components(target, reference, module, 1e-8)
delta <- cor(t(cbind(reference, target))) - cor(t(reference))
diag(delta) <- 0
sed <- mean(abs(target[module] - rowMeans(reference)[module]))
pccin <- sum(abs(delta[module, module])) / length(module)^2
pccout <- mean(abs(delta[module, setdiff(rownames(reference), module)]))
stopifnot(part$valid, abs(part$score - sed * pccin / pccout) < 1e-12,
          abs(part$sPCC_in - pccin) < 1e-12)
# A sample at the exact reference mean cannot manufacture risk via an undefined ratio.
independent <- diag(5)
independent <- cbind(independent, -independent)
rownames(independent) <- letters[1:5]
zero <- single_components(setNames(rep(0, 5), letters[1:5]), independent, module, 1e-8)
stopifnot(!zero$valid, is.na(zero$score))
bad <- reference; bad[1, ] <- 1
stopifnot(inherits(try(single_components(target, bad, module, 1e-8), silent = TRUE), "try-error"))
stopifnot(inherits(try(single_components(target, reference, letters[1:5], 1e-8), silent = TRUE), "try-error"))
# Scoring one target does not change the reference used for the next target.
before <- reference
again <- single_components(target, reference, module, 1e-8)
stopifnot(identical(reference, before), identical(part, again))
cat("R arithmetic, zero-denominator, constant-reference, module-bound and isolation checks passed.\n")
