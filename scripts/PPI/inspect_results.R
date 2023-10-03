## Script Overview:
#
# This script loads and processes fMRI data from a CSV file and then 
# conducts mixed-effects linear models to investigate the effect of various 
# predictors on the dependent variable. 
# It performs the following tasks:
# 
# 1. **Data Loading and Pre-processing**: Loads data from a CSV, prepares the 
#    dataset by aggregating and transforming necessary variables.
# 
# 2. **Diagnostic Plots (commented out)**: Generates diagnostic plots to visualize 
#    relationships between predictors and the dependent variable. 
#
# 3. **Mixed Linear Models**: Fits several mixed-effects linear models to the 
#    data and provides detailed summaries.
# 
# 4. **Model Comparison**: Compares pairs of models to ascertain which provides 
#    a better fit to the data.
#
# 5. **Report Summary**: Displays summary statistics of all fitted models.
#
# ## Dependencies:
# 
# - data.table
# - ggplot2
# - lme4
# - lmerTest
# - pbkrtest
# - parallel

library(data.table)
library(ggplot2)
library(lme4)
library(lmerTest)
library(pbkrtest)
library(parallel)

## http://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#ddf
## https://stats.stackexchange.com/questions/20836/algorithms-for-automatic-model-selection

## NOTE: prepare data
X <- fread('X.csv')
X[, V1 := NULL]
X[, run := as.factor(run)]
XX <- X[, lapply(.SD, mean), .(run, TR)]

## NOTE: diagnostic plots
## ggplot(data=X, aes(x=y_fov, y=y_ppi_loc, colour=as.factor(run))) +
## ggplot(data=X, aes(x=y_fov, y=y_ppi_ffa)) +
##   geom_point(alpha=0.2) +
##   geom_smooth(method='lm') +
##   facet_wrap(~sub, ncol=5)

## NOTE: mixed
ddf <- c('Satterthwaite')
## ddf <- c('Kenward-Roger')

fm_per_norm <- lmer(y_fov ~
                      y_p +
                      y_per_norm +
                      y_ppi_per_norm +
                      (0 + y_p|sub) +
                      (0 + y_per_norm|sub) +
                      (0 + y_ppi_per_norm|sub) +
                      (1|sub) +
                      run +
                      tx + ty + tz +
                      rx + ry + rz +
                      drift_1 + drift_2 + drift_3 +
                      1,
                    data=X)
summary(fm_per_norm, ddf=ddf)

fm_per_inv <- lmer(y_fov ~
                      y_p +
                      y_per_inv +
                      y_ppi_per_inv +
                      (0 + y_p|sub) +
                      (0 + y_per_inv|sub) +
                      (0 + y_ppi_per_inv|sub) +
                      (1|sub) +
                      run +
                      tx + ty + tz +
                      rx + ry + rz +
                      drift_1 + drift_2 + drift_3 +
                      1,
                    data=X)
summary(fm_per_inv, ddf=ddf)

fm_loc <- lmer(y_fov ~
                      y_p +
                      y_loc +
                      y_ppi_loc +
                      (0 + y_p|sub) +
                      (0 + y_loc|sub) +
                      (0 + y_ppi_loc|sub) +
                      (1|sub) +
                      run +
                      tx + ty + tz +
                      rx + ry + rz +
                      drift_1 + drift_2 + drift_3 +
                      1,
                    data=X)
summary(fm_loc, ddf=ddf)

fm_ffa <- lmer(y_fov ~
                      y_p +
                      y_ffa +
                      y_ppi_ffa +
                      (0 + y_p|sub) +
                      (0 + y_ffa|sub) +
                      (0 + y_ppi_ffa|sub) +
                      (1|sub) +
                      run +
                      tx + ty + tz +
                      rx + ry + rz +
                      drift_1 + drift_2 + drift_3 +
                      1,
                    data=X)
summary(fm_ffa, ddf=ddf)

fm_a1 <- lmer(y_fov ~
                      y_p +
                      y_a1 +
                      y_ppi_a1 +
                      (0 + y_p|sub) +
                      (0 + y_a1|sub) +
                      (0 + y_ppi_a1|sub) +
                      (1|sub) +
                      run +
                      tx + ty + tz +
                      rx + ry + rz +
                      drift_1 + drift_2 + drift_3 +
                      1,
                    data=X)
summary(fm_a1, ddf=ddf)

## NOTE: model comparison approach
fm_per_norm_1 <- lmer(y_fov ~
                        y_p +
                        y_per_norm +
                        y_ppi_per_norm +
                        (0 + y_p|sub) +
                        (0 + y_per_norm|sub) +
                        (0 + y_ppi_per_norm|sub) +
                        (1|sub) +
                        run +
                        tx + ty + tz +
                        rx + ry + rz +
                        drift_1 + drift_2 + drift_3 +
                        1,
                      data=X)

fm_per_norm_2 <- lmer(y_fov ~
                        y_p +
                        y_per_norm +
                        (0 + y_p|sub) +
                        (0 + y_per_norm|sub) +
                        (0 + y_ppi_per_norm|sub) +
                        (1|sub) +
                        run +
                        tx + ty + tz +
                        rx + ry + rz +
                        drift_1 + drift_2 + drift_3 +
                        1,
                      data=X)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))
pb_per_norm <- pbkrtest::PBmodcomp(fm_per_norm_1, fm_per_norm_2, seed=0, nsim=5000, cl=cl)

fm_per_inv_1 <- lmer(y_fov ~
                       y_p +
                       y_per_inv +
                       y_ppi_per_inv +
                       (0 + y_p|sub) +
                       (0 + y_per_inv|sub) +
                       (0 + y_ppi_per_inv|sub) +
                       (1|sub) +
                       run +
                       tx + ty + tz +
                       rx + ry + rz +
                       drift_1 + drift_2 + drift_3 +
                       1,
                     data=X)

fm_per_inv_2 <- lmer(y_fov ~
                       y_p +
                       y_per_inv +
                       (0 + y_p|sub) +
                       (0 + y_per_inv|sub) +
                       (0 + y_ppi_per_inv|sub) +
                       (1|sub) +
                       run +
                       tx + ty + tz +
                       rx + ry + rz +
                       drift_1 + drift_2 + drift_3 +
                       1,
                     data=X)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))
pb_per_inv <- pbkrtest::PBmodcomp(fm_per_inv_1, fm_per_inv_2, seed=0, nsim=5000, cl=cl)

fm_loc_1 <- lmer(y_fov ~
                   y_p +
                   y_loc +
                   y_ppi_loc +
                   (0 + y_p|sub) +
                   (0 + y_loc|sub) +
                   (0 + y_ppi_loc|sub) +
                   (1|sub) +
                   run +
                   tx + ty + tz +
                   rx + ry + rz +
                   drift_1 + drift_2 + drift_3 +
                   1,
                 data=X)

fm_loc_2 <- lmer(y_fov ~
                   y_p +
                   y_loc +
                   (0 + y_p|sub) +
                   (0 + y_loc|sub) +
                   (0 + y_ppi_loc|sub) +
                   (1|sub) +
                   run +
                   tx + ty + tz +
                   rx + ry + rz +
                   drift_1 + drift_2 + drift_3 +
                   1,
                 data=X)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))
pb_loc <- pbkrtest::PBmodcomp(fm_loc_1, fm_loc_2, seed=0, nsim=5000, cl=cl)

fm_ffa_1 <- lmer(y_fov ~
                   y_p +
                   y_ffa +
                   y_ppi_ffa +
                   (0 + y_p|sub) +
                   (0 + y_ffa|sub) +
                   (0 + y_ppi_ffa|sub) +
                   (1|sub) +
                   run +
                   tx + ty + tz +
                   rx + ry + rz +
                   drift_1 + drift_2 + drift_3 +
                   1,
                 data=X)

fm_ffa_2 <- lmer(y_fov ~
                   y_p +
                   y_ffa +
                   (0 + y_p|sub) +
                   (0 + y_ffa|sub) +
                   (0 + y_ppi_ffa|sub) +
                   (1|sub) +
                   run +
                   tx + ty + tz +
                   rx + ry + rz +
                   drift_1 + drift_2 + drift_3 +
                   1,
                 data=X)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))
pb_ffa <- pbkrtest::PBmodcomp(fm_ffa_1, fm_ffa_2, seed=0, nsim=5000, cl=cl)

fm_a1_1 <- lmer(y_fov ~
                  y_p +
                  y_a1 +
                  y_ppi_a1 +
                  (0 + y_p|sub) +
                  (0 + y_a1|sub) +
                  (0 + y_ppi_a1|sub) +
                  (1|sub) +
                  run +
                  tx + ty + tz +
                  rx + ry + rz +
                  drift_1 + drift_2 + drift_3 +
                  1,
                data=X)

fm_a1_2 <- lmer(y_fov ~
                  y_p +
                  y_a1 +
                  (0 + y_p|sub) +
                  (0 + y_a1|sub) +
                  (0 + y_ppi_a1|sub) +
                  (1|sub) +
                  run +
                  tx + ty + tz +
                  rx + ry + rz +
                  drift_1 + drift_2 + drift_3 +
                  1,
                data=X)

nc <- detectCores()
cl <- makeCluster(rep("localhost", nc))
pb_a1 <- pbkrtest::PBmodcomp(fm_a1_1, fm_a1_2, seed=0, nsim=5000, cl=cl)

# NOTE: report summary
summary(fm_per_norm, ddf=ddf)
summary(fm_per_inv, ddf=ddf)
summary(fm_loc, ddf=ddf)
summary(fm_ffa, ddf=ddf)
summary(fm_a1, ddf=ddf)

summary(pb_per_norm)
summary(pb_per_inv)
summary(pb_loc)
summary(pb_ffa)
summary(pb_a1)

## NOTE: get model equations
## library(equatiomatic)
## extract_eq(fm)
## extract_eq(get_model(s))
