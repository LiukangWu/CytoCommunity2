Dataset      <- "./TNBC_Input"
Step2_Output <- "./Step2_Output"
Step3_Output <- "./Step3_Output"

# Read the list of images
image_names <- readLines(file.path(Dataset, "ImageNameList.txt"))

# Prepare output directory
if (dir.exists(Step3_Output)) unlink(Step3_Output, recursive = TRUE)
dir.create(Step3_Output, recursive = TRUE)

run_dirs <- list.dirs(path = Step2_Output, recursive = FALSE, full.names = TRUE)

library(diceR)

# Perform majority voting for each sample
for (i in seq_along(image_names)) {
  img_name <- image_names[i]
  idx      <- i - 1   
  
  message("Voting on image: ", img_name, " (index ", idx, ")")
  
  hard_list <- list()
  
  # Collect the soft allocation matrix of this sample from each Run
  for (run_dir in run_dirs) {
    cf <- file.path(run_dir, sprintf("ClusterAssignMatrix1_%d.csv", idx))
    if (!file.exists(cf)) next
    
    M <- as.matrix(read.csv(cf, header = FALSE))
    hard_list[[length(hard_list) + 1]] <- max.col(M)
  }
  
  if (length(hard_list) == 0) {
    warning("No soft assignment matrices found for image ", img_name,
            " (index ", idx, "); skipping.")
    next
  }
  
  hard_mat     <- do.call(cbind, hard_list)             
  final_labels <- majority_voting(hard_mat, is.relabelled = FALSE)
  
  out_file <- file.path(Step3_Output,
                        paste0(img_name, "_CNLabel_MajorityVoting.csv"))
  write.table(final_labels,
              file      = out_file,
              quote     = FALSE,
              row.names = FALSE,
              col.names = FALSE)
}
message("All images processed at ", Sys.time())
