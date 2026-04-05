suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(CCA)
  library(CCP)
})

# Config
LONG_CSV <- "./data/Communication/config/EnrichScoreMatrix_long.csv"
OUT_DIR  <- "./data/Communication/config"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TopVariableNum <- 5
NBOOT <- 10000
MIN_COMMON_SAMPLES <- 3
SD_EPS <- 1e-8
N_CANON_SAVE <- 2   # 保存前两个典型相关系数

# 1) Read long table
all_df <- read.csv(LONG_CSV, stringsAsFactors = FALSE)
all_df <- all_df %>%
  transmute(
    Sample   = as.character(Sample),
    CN       = as.character(CN),
    CellType = as.character(CellType),
    Score    = as.numeric(Score)
  )

# 2) Build Sample × CellType matrix for each CN
CN_list <- sort(unique(all_df$CN))
ScoreList <- lapply(CN_list, function(cn) {
  mat <- all_df %>%
    dplyr::filter(CN == cn) %>%
    dplyr::select(Sample, CellType, Score) %>%
    tidyr::pivot_wider(names_from = CellType, values_from = Score) %>%
    as.data.frame()
  
  rownames(mat) <- mat$Sample
  mat <- mat[, colnames(mat) != "Sample", drop = FALSE]
  mat[] <- lapply(mat, as.numeric)
  mat
})
names(ScoreList) <- CN_list

# 3) Select top enriched features for each CN
select_top_features <- function(mat, top_n) {
  if (is.null(mat) || ncol(mat) == 0) return(mat)
  
  mean_vec <- apply(mat, 2, function(z) mean(z, na.rm = TRUE))
  mean_vec[is.na(mean_vec)] <- -Inf
  ord <- order(mean_vec, decreasing = TRUE)
  keep_n <- min(top_n, length(ord))
  
  mat[, ord[seq_len(keep_n)], drop = FALSE]
}

ScoreListTop <- lapply(ScoreList, select_top_features, top_n = TopVariableNum)

# 4) Data cleaning
clean_data <- function(X, Y) {
  common <- intersect(rownames(X), rownames(Y))
  if (length(common) < MIN_COMMON_SAMPLES) return(NULL)
  
  X <- X[common, , drop = FALSE]
  Y <- Y[common, , drop = FALSE]
  
  X <- X[, colSums(!is.na(X)) > 0, drop = FALSE]
  Y <- Y[, colSums(!is.na(Y)) > 0, drop = FALSE]
  
  X <- X[, sapply(X, function(z) sd(z, na.rm = TRUE) > SD_EPS), drop = FALSE]
  Y <- Y[, sapply(Y, function(z) sd(z, na.rm = TRUE) > SD_EPS), drop = FALSE]
  
  keep_rows <- complete.cases(X) & complete.cases(Y)
  X <- X[keep_rows, , drop = FALSE]
  Y <- Y[keep_rows, , drop = FALSE]
  
  if (nrow(X) < MIN_COMMON_SAMPLES) return(NULL)
  if (ncol(X) == 0 || ncol(Y) == 0) return(NULL)
  
  list(
    X = as.matrix(X),
    Y = as.matrix(Y)
  )
}

# 5) Run CCA for all CN pairs
CN_pairs <- combn(CN_list, 2, simplify = FALSE)
rho_results <- list()
cc_cache <- list()

for (pair in CN_pairs) {
  cnA <- pair[1]
  cnB <- pair[2]
  
  cleaned <- clean_data(ScoreListTop[[cnA]], ScoreListTop[[cnB]])
  if (is.null(cleaned)) next
  
  cc_res <- CCA::cc(cleaned$X, cleaned$Y)
  n_can  <- min(length(cc_res$cor), N_CANON_SAVE)
  
  # 顺序检验：第1个典型相关对应 rhostart=1，第2个对应 rhostart=2
  pvals <- rep(NA_real_, n_can)
  for (k in seq_len(n_can)) {
    perm_k <- CCP::p.perm(cleaned$X, cleaned$Y, nboot = NBOOT, rhostart = k)
    pvals[k] <- perm_k$p.value[1]
  }
  
  key <- paste(cnA, cnB, sep = "_vs_")
  
  one_row <- data.frame(
    CN_A = cnA,
    CN_B = cnB,
    n_samples = nrow(cleaned$X),
    stringsAsFactors = FALSE
  )
  
  # 保存 rho1, rho2, pval1, pval2
  for (k in seq_len(N_CANON_SAVE)) {
    one_row[[paste0("rho", k)]]  <- if (k <= length(cc_res$cor)) cc_res$cor[k] else NA_real_
    one_row[[paste0("pval", k)]] <- if (k <= length(pvals)) pvals[k] else NA_real_
  }
  
  rho_results[[key]] <- one_row
  cc_cache[[key]] <- cc_res
}

df_rho <- bind_rows(rho_results) %>% arrange(desc(abs(rho1)))

write.csv(df_rho, file.path(OUT_DIR, "CCA_config.csv"), row.names = FALSE)

# 6) Write variable coordinates for first and second canonical variates
for (i in seq_len(nrow(df_rho))) {
  cnA <- as.character(df_rho$CN_A[i])
  cnB <- as.character(df_rho$CN_B[i])
  key <- paste(cnA, cnB, sep = "_vs_")
  cc_res <- cc_cache[[key]]
  
  max_k <- min(N_CANON_SAVE,
               ncol(cc_res$scores$corr.X.xscores),
               ncol(cc_res$scores$corr.Y.yscores))
  
  x_ct <- rownames(cc_res$scores$corr.X.xscores)
  y_ct <- rownames(cc_res$scores$corr.Y.yscores)
  all_ct <- union(x_ct, y_ct)
  
  df_coord <- data.frame(CellType = all_ct, stringsAsFactors = FALSE)
  
  for (k in seq_len(max_k)) {
    xcorr <- cc_res$scores$corr.X.xscores[, k]
    ycorr <- cc_res$scores$corr.Y.yscores[, k]
    
    df_coord[[paste0("CN_A_coord_can", k)]] <- unname(xcorr[all_ct])
    df_coord[[paste0("CN_B_coord_can", k)]] <- unname(ycorr[all_ct])
    df_coord[[paste0("rho", k)]]  <- df_rho[[paste0("rho", k)]][i]
    df_coord[[paste0("pval", k)]] <- df_rho[[paste0("pval", k)]][i]
  }
  
  write.csv(
    df_coord,
    file.path(OUT_DIR, paste0("CCA_coordinates_CN", cnA, "_vs_CN", cnB, ".csv")),
    row.names = FALSE
  )
}