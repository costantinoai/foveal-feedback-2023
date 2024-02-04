#-------------------------------------------------------------------------
# Script for fMRI Data Analysis using Linear Mixed-Effects Models
#
# Purpose:
# This script performs the following tasks:
# 1. Loads and preprocesses fMRI data from a provided CSV.
# 2. Creates a series of linear mixed-effects models for specified predictors.
# 3. Compares models with and without interaction terms using parallel bootstrapping.
# 4. Extracts and prints model equations in LaTeX format.
#
# Background:
# Linear mixed-effects models are employed to handle both fixed and random effects in fMRI data.
# This allows for the incorporation of subject-specific random effects which are essential 
# given the high inter-subject variability in fMRI responses.
# The primary aim is to evaluate the contribution of various predictors (e.g., y_per, y_opp, etc.)
# to the response variable (y_fov) while considering the interaction between the predictor and y_p.
#
# Equations:
# General form of the model equation used:
# y_fov = y_p + predictor + y_ppi_predictor + random_effects + run + tx + ... + drift_n + 1
# Where:
# - y_fov is the response variable.
# - predictor is the specific predictor under investigation.
# - y_ppi_predictor is the interaction term between y_p and the predictor.
# - random_effects include subject-specific random intercepts and slopes.
# - run, tx, ..., drift_n are control variables.
#
# The model equation for a given predictor, say 'y_per', is represented as:
# [y_fov] = β₀ + β₁[y_p] + β₂[y_per] + β₃[y_ppi_y_per] + random_effects + β₄[run] + ... + βₙ[drift_n]
# Where:
# β₀, β₁, ... are the fixed effect coefficients.
# random_effects include subject-specific random intercepts and slopes for y_p and y_per.
# [run], ..., [drift_n] are control variables with their respective coefficients.
# This equation is created for each predictor and the significance of the coefficients is evaluated.

# When comparing two models, the key interest is in the p-value which indicates whether 
# the interaction term (y_ppi_predictor) significantly improves the model fit.
# A lower p-value (typically < 0.05) suggests that the interaction term is significant.

# The 'extract_eq' function will produce the equation in LaTeX format which can be directly used 
# for publication in the academic paper. Ensure to consult the generated equations for accuracy 
# before inclusion in the paper.
#
# Dataset Structure:
# Expected columns in the dataset include 'run', 'TR', 'y_fov', 'y_p', each predictor, and control variables.
# - 'run' is expected to be a factor indicating the specific fMRI run.
# - 'TR' indicates the repetition time.
#
# Note:
# When using this script as supplementary material, ensure that the provided dataset follows the described structure.
#
# Dependencies:
# - data.table
# - ggplot2
# - parallel
#
# Authors: Matthew Crossley
# Edited: Andrea Costantino
# Date: 03/10/2023
# -----------------------------------------------------------

rm(list=ls())
library(data.table)
library(ggplot2)
library(parallel)

set.seed(42)
options("width" = 100)

X <- fread('../res/PPI/ppi_results.csv')

X_per <- X[, .(sub, run, TR, y_fov, y_p, y_per, y_ppi_per, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_opp <- X[, .(sub, run, TR, y_fov, y_p, y_opp, y_ppi_opp, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_loc <- X[, .(sub, run, TR, y_fov, y_p, y_loc, y_ppi_loc, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_ffa <- X[, .(sub, run, TR, y_fov, y_p, y_ffa, y_ppi_ffa, tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]
X_a1 <-  X[, .(sub, run, TR, y_fov, y_p, y_a1, y_ppi_a1  , tx, ty, tz, rx, ry, rz, drift_1, drift_2, drift_3)]

setnames(X_per, c("y_per", "y_ppi_per"), c("y_s", "y_ppi"))
setnames(X_opp, c("y_opp", "y_ppi_opp"), c("y_s", "y_ppi"))
setnames(X_loc, c("y_loc", "y_ppi_loc"), c("y_s", "y_ppi"))
setnames(X_ffa, c("y_ffa", "y_ppi_ffa"), c("y_s", "y_ppi"))
setnames(X_a1,  c("y_a1", "y_ppi_a1"),   c("y_s", "y_ppi"))

X_per[, predictor := "per"]
X_opp[, predictor := "opp"]
X_loc[, predictor := "loc"]
X_ffa[, predictor := "ffa"]
X_a1[, predictor := "a1"]

X = rbindlist(list(X_per, X_opp, X_loc, X_ffa, X_a1))

d <- list()
predictors <- c("per", "opp", "loc", "ffa", "a1")

for(i in 1:length(predictors)) {

    print(predictors[i])

    XX <- X[predictor==predictors[i], lapply(.SD, mean), .(predictor, TR)]

    fm <- lm(y_fov ~ y_p + y_s + y_ppi 
                       + tx + ty + tz 
                       + rx + ry + rz 
                       + drift_1 + drift_2 + drift_3 
                       + run,
                       data=XX
                       )

    d[[i]] = data.table(Predictor=predictors[i],
                        Estimate=coef(fm)["y_ppi"],
                        lower=confint(fm)[4, 1],
                        upper=confint(fm)[4, 2])

    print(summary(fm))
}

d <- rbindlist(d)

g <- ggplot(data=d, aes(x=Predictor, y=Estimate)) +
    geom_pointrange(aes(ymin=lower, ymax=upper))
ggsave("../res/PPI/fig_ppi.png", g, width=6, height=3.5)


print(d)
