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
# - lme4
# - lmerTest
# - pbkrtest
# - parallel
#
# Authors: Matthew Crossley
# Edited: Andrea Costantino
# Date: 03/10/2023
# -----------------------------------------------------------
# 1. Initialization: Load necessary libraries and set global options

rm(list=ls())
library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)
library(pbkrtest)
library(parallel)
ddf <- c('Satterthwaite') # Set up method for degrees of freedom calculation
set.seed(42)

# -----------------------------------------------------------
# 2. Data Loading and Pre-processing

X <- fread('/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/foveal-feedback-2023/res/PPI/ppi_results.csv') # Load the fMRI dataset
# X[, V1 := NULL] # Remove the V1 column
X[, run := as.factor(run)] # Convert 'run' column to factor type
XX <- X[, lapply(.SD, mean), .(run, TR)] # Aggregate data by 'run' and 'TR'

# -----------------------------------------------------------
# 3. Diagnostic Plots (commented out for demonstration)

# Uncomment this block to generate and visualize diagnostic plots
# ggplot(data=X, aes(x=y_fov, y=y_ppi_loc, colour=as.factor(run))) +
# ggplot(data=X, aes(x=y_fov, y=y_ppi_ffa)) +
#   geom_point(alpha=0.2) +
#   geom_smooth(method='lm') +
#   facet_wrap(~sub, ncol=5)

# -----------------------------------------------------------
# 4. Mixed Linear Models

# List of predictors
# predictors <- c("y_per", "y_opp", "y_loc", "y_ffa", "y_a1")
predictors <- c("y_per", "y_opp", "y_loc", "y_ffa")

# Function to create a model for a given predictor
create_model <- function(predictor, data) {
  # Build the formula string for the model based on the predictor
  formula_string <- paste0(
    "y_fov ~ y_p + ", predictor, " + y_ppi_", sub("y_", "", predictor),
    " + (0 + y_p|sub) + (0 + ", predictor, "|sub) + (0 + y_ppi_", sub("y_", "", predictor), "|sub)",
    " + (1|sub) + run + tx + ty + tz + rx + ry + rz + drift_1 + drift_2 + drift_3 + 1"
  )
  
  # Convert the formula string to a formula object
  formula_obj <- as.formula(formula_string)
  
  # Create the linear mixed-effects model with the constructed formula
  model <- lmer(formula_obj, data = data)
  
  return(model)
}

# Dictionary to store models
model_dict <- list()

# Initialize an empty data frame to store results
results_df <- data.frame(Predictor = character(),
                         Beta = numeric(),
                         CI_Lower = numeric(),
                         CI_Upper = numeric(),
                         stringsAsFactors = FALSE)

# Loop through predictors, create models, and extract summaries
for(predictor in predictors){
  model <- create_model(predictor, X)
  model_dict[[predictor]] <- model  # Store the model for future use
  
  cat(paste0("Summary for ", predictor, ":\n"))
  model_summary <- summary(model)
  print(model_summary)
  
  # # Compute confidence intervals
  # ci <- confint(model, method = "profile")
  # 
  # # Extract and store coefficients and confidence intervals
  # beta_coeffs <- fixef(model)
  # for(term in names(beta_coeffs)) {
  #   beta <- beta_coeffs[term]
  #   ci_lower <- ci[term, 1]
  #   ci_upper <- ci[term, 2]
  #   
  #   # Append to the dataframe
  #   results_df <- rbind(results_df, data.frame(
  #     Predictor = term,
  #     Beta = beta,
  #     CI_Lower = ci_lower,
  #     CI_Upper = ci_upper
  #   ))
  # }
}

# -----------------------------------------------------------
# 5. Model Comparison - For each predictor, we'll compare two models to determine the better fit.

# Function to create models for a given predictor
create_models <- function(predictor, data){
  # Corrected interaction term construction
  interaction_term <- paste0("y_ppi_", sub("y_", "", predictor))
  
  # Model with interaction term
  model_1_formula <- as.formula(paste0("y_fov ~ y_p + ", predictor, " + ", interaction_term, 
                                       " + (0 + y_p|sub) + (0 + ", predictor, "|sub) + (0 + ", interaction_term, "|sub) + (1|sub) + run + tx + ty + tz + rx + ry + rz + drift_1 + drift_2 + drift_3 + 1"))
  model_1 <- lmer(model_1_formula, data = data)
  
  # Model without interaction term
  model_2_formula <- as.formula(paste0("y_fov ~ y_p + ", predictor, 
                                       " + (0 + y_p|sub) + (0 + ", predictor, "|sub) + (0 + ", interaction_term, "|sub) + (1|sub) + run + tx + ty + tz + rx + ry + rz + drift_1 + drift_2 + drift_3 + 1"))
  model_2 <- lmer(model_2_formula, data = data)
  
  return(list(model_1, model_2))
}

# Function to perform model comparison
compare_models <- function(model_1, model_2, cl){
  return(pbkrtest::PBmodcomp(model_1, model_2, seed=0, nsim=5000, cl=cl))
}

# Detect number of CPU cores and create a cluster for parallel processing
nc <- detectCores() 
cl <- makeCluster(rep("localhost", nc))

# Lists to store models and results
model_list <- list()
results <- list()

# Loop through predictors and conduct model comparisons
for(predictor in predictors){
  models <- create_models(predictor, X)
  model_list[[predictor]] <- models[[1]]  # Store the model with interaction for summaries
  results[[predictor]] <- compare_models(models[[1]], models[[2]], cl)
  
  # Display summaries and equations
  cat(paste0("Summary for ", predictor, " (with interaction):\n"))
  print(summary(model_list[[predictor]], ddf=ddf))
  cat(paste0("Model comparison for ", predictor, ":\n"))
  print(summary(results[[predictor]]))
  
}

write.csv(results_df, "./res/PPI/R_results.csv", row.names = FALSE)


