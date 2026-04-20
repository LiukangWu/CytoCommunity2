suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
  library(ggtext)
  library(showtext)
  library(sysfonts)
})

font_add("Arial", "C:/Windows/Fonts/arial.ttf")
showtext_auto()
# =========================
# Config
# =========================
IN_DIR  <- "./data/Communication/config"
OUT_DIR <- "./plot/Communication/SpearmanBetweenCNs_plot/CCA_plots"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# 读取所有系数文件
coef_files <- list.files(
  IN_DIR,
  pattern = "^CCA_coordinates_CN.+_vs_CN.+\\.csv$",
  full.names = TRUE
)

if (length(coef_files) == 0) {
  stop("No CCA coefficient files found in: ", IN_DIR)
}

# =========================
# Helper: make one plot
# =========================
plot_one_cca <- function(coef_file, out_dir) {
  fname <- basename(coef_file)
  
  cnA <- sub("^CCA_coordinates_CN(.+)_vs_CN(.+)\\.csv$", "\\1", fname)
  cnB <- sub("^CCA_coordinates_CN(.+)_vs_CN(.+)\\.csv$", "\\2", fname)
  
  df <- read.csv(coef_file, stringsAsFactors = FALSE, check.names = FALSE)
  
  # rho1 / rho2
  rho1 <- if ("rho1" %in% colnames(df)) unique(na.omit(df$rho1))[1] else NA_real_
  rho2 <- if ("rho2" %in% colnames(df)) unique(na.omit(df$rho2))[1] else NA_real_
  
  df_A <- df %>%
  transmute(
    CellType = CellType,
    x = CN_A_coord_can1,
    y = CN_A_coord_can2,
    Group = paste0("CN-", cnA)
  ) %>%
  filter(!is.na(x), !is.na(y))

  df_B <- df %>%
    transmute(
      CellType = CellType,
      x = CN_B_coord_can1,
      y = CN_B_coord_can2,
      Group = paste0("CN-", cnB)
    ) %>%
    filter(!is.na(x), !is.na(y))
  
  plot_df <- bind_rows(df_A, df_B)
  if (nrow(plot_df) == 0) {
    warning("Skip file (no valid points): ", fname)
    return(NULL)
  }
  
  x_lab <- if (!is.na(rho1)) {
    paste0("First canonical variate pair\n(correlation coefficient = ", sprintf("%.2f", rho1), ")")
  }
  y_lab <- if (!is.na(rho2)) {
    paste0("Second canonical variate pair\n(correlation coefficient = ", sprintf("%.2f", rho2), ")")
  }
  
  colorA <- "#B24A4A"
  colorB <- "#3C78B5"

  color_map <- setNames(
    c(colorA, colorB),
    c(paste0("CN-", cnA), paste0("CN-", cnB))
  )

  title_txt <- paste0(
    "<span style='color:", colorA, ";'>CN-", cnA, "</span>",
    " and ",
    "<span style='color:", colorB, ";'>CN-", cnB, "</span>"
  )

  p <- ggplot(plot_df, aes(x = x, y = y, color = Group, label = CellType)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey65", linewidth = 0.4) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey65", linewidth = 0.4) +
    geom_point(size = 2.3) +
    geom_text_repel(
      size = 3.2,
      box.padding = 0.25,
      point.padding = 0.2,
      segment.color = "grey70",
      segment.size = 0.3,
      max.overlaps = Inf,
      family = "Arial"
    ) +
    scale_color_manual(values = color_map) +
    scale_x_continuous(limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1)) +
    scale_y_continuous(limits = c(-1, 1), breaks = c(-1, -0.5, 0, 0.5, 1)) +
    theme_classic(base_size = 10, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.6, color = "black"),
      axis.ticks = element_line(linewidth = 0.5, color = "black"),
      legend.position = "none",
      plot.title = ggtext::element_markdown(hjust = 0.5, face = "plain", family = "Arial"),
      axis.title = element_text(family = "Arial"),
      axis.text = element_text(family = "Arial"),
      plot.margin = margin(10, 15, 10, 10)
    ) +
    labs(
      title = title_txt,
      x = x_lab,
      y = y_lab
    )
  
  pdf_file <- file.path(out_dir, paste0("CCA_plot_CN", cnA, "_vs_CN", cnB, ".pdf"))
  
  ggsave(pdf_file, p, width = 6, height = 6, units = "in")
  
  message("Saved: ", pdf_file)
}

# =========================
# Loop
# =========================
for (f in coef_files) {
  plot_one_cca(f, OUT_DIR)
}