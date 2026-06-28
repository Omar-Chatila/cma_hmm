#!/usr/bin/env Rscript
# Reference moveHMM fit using projected coordinates, step length, and turn angle.
#
# Usage:
#   Rscript movehmm_reference.R input_points.csv output_states.csv [number_of_states]
#
# input_points.csv must contain: individual_local_identifier, utm_x, utm_y
# and either timestamp or timestamp_utc. The output preserves the input rows
# (sorted by animal and time) and adds movehmm_step, movehmm_angle, and a
# zero-based state column.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || length(args) > 3) {
  stop("Usage: Rscript movehmm_reference.R input_points.csv output_states.csv [number_of_states]")
}
if (!requireNamespace("moveHMM", quietly = TRUE)) {
  stop("moveHMM is required. Install it once with: install.packages('moveHMM')")
}

input_path <- args[[1]]
output_path <- args[[2]]
model_path <- if (grepl("\\.csv$", output_path, ignore.case = TRUE)) {
  sub("\\.csv$", ".rds", output_path, ignore.case = TRUE)
} else {
  paste0(output_path, ".rds")
}
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
nb_states <- if (length(args) == 3) as.integer(args[[3]]) else 2L
if (is.na(nb_states) || nb_states < 2) stop("number_of_states must be an integer of at least 2.")

points <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
id_col <- "individual_local_identifier"
time_col <- if ("timestamp" %in% names(points)) "timestamp" else "timestamp_utc"
required <- c(id_col, time_col, "utm_x", "utm_y")
missing <- setdiff(required, names(points))
if (length(missing)) stop("Missing required columns: ", paste(missing, collapse = ", "))

points$.source_row <- seq_len(nrow(points))
points$.time <- as.POSIXct(points[[time_col]], tz = "UTC")
if (anyNA(points$.time)) stop("Could not parse every timestamp in column '", time_col, "'.")
points$utm_x <- as.numeric(points$utm_x)
points$utm_y <- as.numeric(points$utm_y)
points <- points[complete.cases(points[, c(id_col, ".time", "utm_x", "utm_y")]), ]
points <- points[order(points[[id_col]], points$.time), ]
if (nrow(points) < nb_states * 3) stop("Too few valid locations for the requested number of states.")

# prepData calculates UTM step length and turning angle. Keep this data frame
# minimal: timestamp is used for ordering above, not as a model covariate.
track <- data.frame(
  ID = points[[id_col]],
  x = points$utm_x,
  y = points$utm_y
)
movement <- moveHMM::prepData(track, type = "UTM", coordNames = c("x", "y"))

# Robust, data-scaled starting values. State 1 starts shorter/slower and the
# last state longer/faster; the optimiser estimates the final parameters.
positive_steps <- movement$step[is.finite(movement$step) & movement$step > 0]
if (length(positive_steps) < nb_states) stop("Too few positive step lengths to fit a gamma HMM.")
mu0 <- as.numeric(quantile(positive_steps, probs = seq(0.2, 0.8, length.out = nb_states), names = FALSE))
sigma0 <- rep(max(stats::sd(positive_steps), median(positive_steps) * 0.1, 1e-6), nb_states)
step_par0 <- c(mu0, sigma0)
if (any(movement$step == 0, na.rm = TRUE)) step_par0 <- c(step_par0, rep(0.01, nb_states))

# A conventional movement model: all states are centred on forward movement,
# while kappa distinguishes tortuous (low) from directed (high) movement.
angle_mean <- rep(0, nb_states)
angle_par0 <- seq(0.2, 2, length.out = nb_states)
model <- moveHMM::fitHMM(
  data = movement,
  nbStates = nb_states,
  stepPar0 = step_par0,
  anglePar0 = angle_par0,
  angleMean = angle_mean,
  stepDist = "gamma",
  angleDist = "vm",
  stationary = TRUE,
  verbose = 0
)

points$movehmm_step <- movement$step
points$movehmm_angle <- movement$angle
points$state <- as.integer(moveHMM::viterbi(model)) - 1L
points$.time <- NULL
write.csv(points, output_path, row.names = FALSE)
saveRDS(model, model_path)

cat("Wrote zero-based states to", output_path, "and model to", model_path, "\n")
