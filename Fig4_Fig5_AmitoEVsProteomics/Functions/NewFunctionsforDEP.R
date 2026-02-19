createMetaData = function(data, col_range, row_range, sampleNameColumns, metaColNames) {

  meta_data = data.frame(data[row_range, col_range, drop = FALSE])
  
  if(length(metaColNames) != length(col_range)) {
    stop("Number of provided column names (metaColNames) must match the selected column range.")
  }
  
  if (is.numeric(sampleNameColumns)) {
    rownames(meta_data) = meta_data[[sampleNameColumns]]
  } else {
    rownames(meta_data) = meta_data[, sampleNameColumns]
  }
  
  colnames(meta_data) = metaColNames
  
  return(meta_data)
}


relevelMetaData = function(meta_data, ref_levels) {
  for (col in names(ref_levels)) {
    if (col %in% colnames(meta_data)) {
      meta_data[[col]] = as.factor(meta_data[[col]])
      meta_data[[col]] = relevel(meta_data[[col]], ref = ref_levels[[col]])
    } else {
      warning(paste("Column", col, "not found in meta_data. Skipping..."))
    }
  }
  
  return(meta_data)
}


createExpDesign = function(meta_data, ref_levels) {

  selected_cols = names(ref_levels)
  
  missing_cols = setdiff(selected_cols, colnames(meta_data))
  if (length(missing_cols) > 0)
    warning("Missing columns in meta_data: ", paste(missing_cols, collapse = ", "))
  
  expDesign = data.frame(label = rownames(meta_data), stringsAsFactors = FALSE)
  
  for (col in selected_cols) {
    if (col %in% colnames(meta_data)) {
      vals = as.character(meta_data[[col]])
      expDesign[[col]] = relevel(factor(vals), ref = ref_levels[[col]])
    }
  }
  return(expDesign)
}


createProteinData = function(data, proteinNameColumns, col_range) {
  
  data_t = t(data)
  
  colnames(data_t) = data_t[proteinNameColumns, ]
  
  clean_data = data.frame(data_t[-col_range, ], stringsAsFactors = FALSE)
  
  return(clean_data)
}

