suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(CCA)
  library(CCP)
})

# =========================
# Config
# =========================
LONG_CSV <- "./data/Communication/config/EnrichScoreMatrix_long.csv"
OUT_DIR  <- "./data/Communication/config"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

TopVariableNum <- 5
NBOOT <- 10000
MIN_COMMON_SAMPLES <- 3
SD_EPS <- 1e-8
N_CANON_SAVE <- 2

# =========================
# Helpers
# =========================
infer_condition <- function(sample_name) {
  s <- trimws(tolower(as.character(sample_name)))
  
  # 1) patient02_0_210115 -> condition0
  parts <- unlist(strsplit(s, "_"))
  if (length(parts) >= 3) {
    mid <- parts[2]
    if (grepl("^[0-9]+$", mid)) {
      return(paste0("condition", mid))
    }
  }
  
  # 2) t0 / t1 / t2
  m <- regexpr("(^|[_\\-])(t[0-9]+)([_\\-]|$)", s, perl = TRUE)
  if (m[1] != -1) {
    val <- regmatches(s, m)
    val <- gsub("^[_\\-]|[_\\-]$", "", val)
    return(val)
  }
  
  # 3) condition0 / condition1 / condition2
  m <- regexpr("(^|[_\\-])(condition[0-9]+)([_\\-]|$)", s, perl = TRUE)
  if (m[1] != -1) {
    val <- regmatches(s, m)
    val <- gsub("^[_\\-]|[_\\-]$", "", val)
    return(val)
  }
  
  return("Unknown")
}

select_top_features <- function(mat, top_n) {
  if (is.null(mat) || ncol(mat) == 0) return(mat)
  
  mean_vec <- apply(mat, 2, function(z) mean(z, na.rm = TRUE))
  mean_vec[is.na(mean_vec)] <- -Inf
  ord <- order(mean_vec, decreasing = TRUE)
  keep_n <- min(top_n, length(ord))
  
  mat[, ord[seq_len(keep_n)], drop = FALSE]
}

clean_data <- function(X, Y) {
  if (is.null(X) || is.null(Y)) return(NULL)
  if (ncol(X) == 0 || ncol(Y) == 0) return(NULL)
  
  common <- intersect(rownames(X), rownames(Y))
  if (length(common) < MIN_COMMON_SAMPLES) return(NULL)
  
  X <- X[common, , drop = FALSE]
  Y <- Y[common, , drop = FALSE]
  
  X <- X[, colSums(!is.na(X)) > 0, drop = FALSE]
  Y <- Y[, colSums(!is.na(Y)) > 0, drop = FALSE]
  
  X <- X[, sapply(X, function(z) sd(z, na.rm = TRUE) > SD_EPS), drop = FALSE]
  Y <- Y[, sapply(Y, function(z) sd(z, na.rm = TRUE) > SD_EPS), drop = FALSE]
  
  if (ncol(X) == 0 || ncol(Y) == 0) return(NULL)
  
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

build_score_list <- function(df_sub, cn_list) {
  out <- list()
  
  for (cn in cn_list) {
    mat <- df_sub %>%
      dplyr::filter(CN == cn) %>%
      dplyr::select(Sample, CellType, Score) %>%
      tidyr::pivot_wider(names_from = CellType, values_from = Score) %>%
      as.data.frame()
    
    if (nrow(mat) == 0) {
      out[[cn]] <- NULL
      next
    }
    
    rownames(mat) <- mat$Sample
    mat <- mat[, colnames(mat) != "Sample", drop = FALSE]
    mat[] <- lapply(mat, as.numeric)
    out[[cn]] <- mat
  }
  
  names(out) <- cn_list
  out
}

# =========================
# Read data
# =========================
all_df <- read.csv(LONG_CSV, stringsAsFactors = FALSE)

all_df <- all_df %>%
  transmute(
    Sample   = as.character(Sample),
    CN       = as.character(CN),
    CellType = as.character(CellType),
    Score    = as.numeric(Score)
  )

all_df$Condition <- vapply(all_df$Sample, infer_condition, character(1))

if (any(all_df$Condition == "Unknown")) {
  warning("Some samples have Unknown condition and will be excluded.")
  all_df <- all_df %>% filter(Condition != "Unknown")
}

# 统一 condition 顺序
cond_levels <- c("condition0", "condition1", "condition2", "t0", "t1", "t2")
cond_present <- unique(all_df$Condition)
cond_present <- c(intersect(cond_levels, cond_present),
                  setdiff(sort(cond_present), cond_levels))

all_df$Condition <- factor(all_df$Condition, levels = cond_present)
all_df <- all_df %>% arrange(Condition, Sample, CN, CellType)

CN_list <- sort(unique(all_df$CN))
Condition_list <- levels(all_df$Condition)

# 保存带 condition 的长表，便于后面排查
write.csv(all_df, file.path(OUT_DIR, "EnrichScoreMatrix_long_with_condition.csv"), row.names = FALSE)

# =========================
# Run condition-specific CCA
# =========================
all_results <- list()

for (cond in Condition_list) {
  message("[Running CCA] ", cond)
  
  df_cond <- all_df %>% filter(Condition == cond)
  
  ScoreList <- build_score_list(df_cond, CN_list)
  ScoreListTop <- lapply(ScoreList, select_top_features, top_n = TopVariableNum)
  
  CN_pairs <- combn(CN_list, 2, simplify = FALSE)
  cond_results <- list()
  cc_cache <- list()
  
  for (pair in CN_pairs) {
    cnA <- pair[1]
    cnB <- pair[2]
    
    cleaned <- clean_data(ScoreListTop[[cnA]], ScoreListTop[[cnB]])
    if (is.null(cleaned)) next
    
    cc_res <- CCA::cc(cleaned$X, cleaned$Y)
    n_can  <- min(length(cc_res$cor), N_CANON_SAVE)
    
    pvals <- rep(NA_real_, n_can)
    for (k in seq_len(n_can)) {
      perm_k <- CCP::p.perm(cleaned$X, cleaned$Y, nboot = NBOOT, rhostart = k)
      pvals[k] <- perm_k$p.value[1]
    }
    
    key <- paste(cond, cnA, cnB, sep = "__")
    
    one_row <- data.frame(
      Condition = as.character(cond),
      CN_A = cnA,
      CN_B = cnB,
      n_samples = nrow(cleaned$X),
      stringsAsFactors = FALSE
    )
    
    for (k in seq_len(N_CANON_SAVE)) {
      one_row[[paste0("rho", k)]]  <- if (k <= length(cc_res$cor)) cc_res$cor[k] else NA_real_
      one_row[[paste0("pval", k)]] <- if (k <= length(pvals)) pvals[k] else NA_real_
    }
    
    cond_results[[key]] <- one_row
    cc_cache[[key]] <- cc_res
  }
  
  df_cond_rho <- bind_rows(cond_results) %>%
    arrange(desc(abs(rho1)))
  
  # 每个 condition 单独保存
  write.csv(
    df_cond_rho,
    file.path(OUT_DIR, paste0("CCA_config_", as.character(cond), ".csv")),
    row.names = FALSE
  )
  
  # 保存 coordinates
  if (nrow(df_cond_rho) > 0) {
    for (i in seq_len(nrow(df_cond_rho))) {
      cnA <- as.character(df_cond_rho$CN_A[i])
      cnB <- as.character(df_cond_rho$CN_B[i])
      key <- paste(as.character(cond), cnA, cnB, sep = "__")
      cc_res <- cc_cache[[key]]
      
      max_k <- min(
        N_CANON_SAVE,
        ncol(cc_res$scores$corr.X.xscores),
        ncol(cc_res$scores$corr.Y.yscores)
      )
      
      x_ct <- rownames(cc_res$scores$corr.X.xscores)
      y_ct <- rownames(cc_res$scores$corr.Y.yscores)
      all_ct <- union(x_ct, y_ct)
      
      df_coord <- data.frame(CellType = all_ct, stringsAsFactors = FALSE)
      
      for (k in seq_len(max_k)) {
        xcorr <- cc_res$scores$corr.X.xscores[, k]
        ycorr <- cc_res$scores$corr.Y.yscores[, k]
        
        df_coord[[paste0("CN_A_coord_can", k)]] <- unname(xcorr[all_ct])
        df_coord[[paste0("CN_B_coord_can", k)]] <- unname(ycorr[all_ct])
        df_coord[[paste0("rho", k)]]  <- df_cond_rho[[paste0("rho", k)]][i]
        df_coord[[paste0("pval", k)]] <- df_cond_rho[[paste0("pval", k)]][i]
      }
      
      write.csv(
        df_coord,
        file.path(
          OUT_DIR,
          paste0("CCA_coordinates_", as.character(cond), "_CN", cnA, "_vs_CN", cnB, ".csv")
        ),
        row.names = FALSE
      )
    }
  }
  
  all_results[[as.character(cond)]] <- df_cond_rho
}

df_all <- bind_rows(all_results) %>%
  arrange(Condition, desc(abs(rho1)))

write.csv(df_all, file.path(OUT_DIR, "CCA_config_by_condition.csv"), row.names = FALSE)

message("[OK] Saved: ", file.path(OUT_DIR, "CCA_config_by_condition.csv"))