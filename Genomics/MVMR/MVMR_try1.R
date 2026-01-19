# ============================================
# MVMR解析 in R
# ============================================
# 方法1: MRCIEU R-Universeから（推奨）
install.packages("MVMR", repos = c("https://mrcieu.r-universe.dev", "https://cloud.r-project.org"))


# 必要なパッケージの読み込み
library(MVMR)
library(dplyr)
library(ggplot2)

# ============================================
# 1. データの読み込み
# ============================================

# CSVファイルのパスを指定
data_path <- "/Users/yoshizawakazuki/Kazuki_module/Genomics/Meta_GWAS_Analysis/mvmr_strict_common_snps.csv"

# データ読み込み
mvmr_data <- read.csv(data_path, stringsAsFactors = FALSE)

# Union版を読み込み
data_path <- "/Users/yoshizawakazuki/Kazuki_module/Genomics/Meta_GWAS_Analysis/mvmr_union_all_snps.csv"

mvmr_data <- read.csv(data_path, stringsAsFactors = FALSE)

# 以降同じスクリプトで解析

# データの確認
cat("データサイズ:", nrow(mvmr_data), "SNPs\n")
cat("\nデータの最初の数行:\n")
print(head(mvmr_data))

cat("\nデータのサマリー:\n")
print(summary(mvmr_data))

# ============================================
# 2. データのクリーニング
# ============================================

# 欠損値のチェック
cat("\n欠損値の確認:\n")
print(colSums(is.na(mvmr_data)))

# 欠損値を除去
mvmr_data_clean <- mvmr_data %>%
  filter(!is.na(beta_neck) & !is.na(se_neck) &
           !is.na(beta_bmi) & !is.na(se_bmi) &
           !is.na(beta_snore) & !is.na(se_snore)) %>%
  filter(se_neck > 0 & se_bmi > 0 & se_snore > 0)

cat("\nクリーニング後のデータサイズ:", nrow(mvmr_data_clean), "SNPs\n")

# ============================================
# 3. MVMR用のデータフォーマット作成
# ============================================

# 曝露のbetaとSEを行列形式に変換
# 列1: Neck, 列2: BMI
exposure_beta <- cbind(
  mvmr_data_clean$beta_neck,
  mvmr_data_clean$beta_bmi
)

exposure_se <- cbind(
  mvmr_data_clean$se_neck,
  mvmr_data_clean$se_bmi
)

# アウトカム（Snore）
outcome_beta <- mvmr_data_clean$beta_snore
outcome_se <- mvmr_data_clean$se_snore

# SNP ID
snp_ids <- mvmr_data_clean$SNP

cat("\n曝露データの次元:", dim(exposure_beta), "\n")
cat("アウトカムデータの長さ:", length(outcome_beta), "\n")

# ============================================
# 4. MVMRデータオブジェクトの作成
# ============================================

mvmr_input <- format_mvmr(
  BXGs = exposure_beta,
  BYG = outcome_beta,
  seBXGs = exposure_se,
  seBYG = outcome_se,
  RSID = snp_ids
)

cat("\nMVMRデータオブジェクトを作成しました\n")

# ============================================
# 5. 操作変数の強度チェック（F統計量）
# ============================================

cat("\n", paste(rep("=", 60), collapse=""), "\n", sep="")
cat("操作変数の強度チェック（F統計量）\n")
cat("\n", paste(rep("=", 60), collapse=""), "\n", sep="")

# 条件付きF統計量を計算
# gencov = 0: 曝露間の遺伝的共分散（不明な場合は0）
f_stats <- strength_mvmr(
  r_input = mvmr_input,
  gencov = 0
)

print(f_stats)

cat("\n【解釈】\n")
cat("F統計量 > 10 であれば、弱操作変数バイアスのリスクは低い\n")

# ============================================
# 6. MVMR解析の実行（IVW法）
# ============================================

cat("\n" , "="*60, "\n", sep="")
cat("MVMR解析（逆分散加重法）\n")
cat("="*60, "\n", sep="")

mvmr_result <- ivw_mvmr(r_input = mvmr_input)

print(mvmr_result)

# 結果を見やすく整形
results_df <- data.frame(
  Exposure = c("Neck Circumference", "BMI"),
  Beta = mvmr_result[, "Estimate"],
  SE = mvmr_result[, "Std. Error"],
  Pvalue = mvmr_result[, "Pr(>|t|)"],
  CI_lower = mvmr_result[, "Estimate"] - 1.96 * mvmr_result[, "Std. Error"],
  CI_upper = mvmr_result[, "Estimate"] + 1.96 * mvmr_result[, "Std. Error"]
)

cat("\n【MVMR結果サマリー】\n")
print(results_df)


cat("\n【解釈】\n")
cat("Beta: 各曝露の独立した因果効果（他の曝露を調整済み）\n")
cat("P値 < 0.05 で統計的に有意\n")

# ============================================
# 7. 多面的効果のチェック（Q統計量）
# ============================================

cat("\n" , "="*60, "\n", sep="")
cat("多面的効果のテスト（Q統計量）\n")
cat("="*60, "\n", sep="")

pleiotropy_test <- pleiotropy_mvmr(r_input = mvmr_input)

print(pleiotropy_test)

cat("\n【解釈】\n")
cat("Q統計量のP値 < 0.05 の場合、多面的効果の存在が示唆される\n")
cat("多面的効果がある場合は、結果の解釈に注意が必要\n")

# ============================================
# 8. 結果の可視化
# ============================================

cat("\n結果を可視化します...\n")

# 森林プロット
p1 <- ggplot(results_df, aes(x = Exposure, y = Beta)) +
  geom_point(size = 4, color = "steelblue") +
  geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper), 
                width = 0.2, linewidth = 1, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
  coord_flip() +
  labs(
    title = "MVMR Results: Effect on Snoring",
    subtitle = paste0("Based on ", nrow(mvmr_data_clean), " SNPs"),
    x = "",
    y = "Causal Effect (Beta) with 95% CI"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank()
  )

