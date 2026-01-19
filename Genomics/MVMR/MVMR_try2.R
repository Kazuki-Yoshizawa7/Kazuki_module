## BMI WHR Fstats - 8.0くらいでギリ使えないか

# ============================================
# MVMR解析 in R (WHR + BMI -> Snoring)
# ============================================

# 必要に応じてパッケージをインストール
# install.packages("MVMR", repos = c("https://mrcieu.r-universe.dev", "https://cloud.r-project.org"))

# 必要なパッケージの読み込み
library(MVMR)
library(dplyr)
library(ggplot2)

# ============================================
# 1. データの読み込み
# ============================================

# CSVファイルのパスを指定
# Strict版（共通SNPのみ）
# data_path <- "/Users/yoshizawakazuki/Kazuki_module/Genomics/Meta_GWAS_Analysis/mvmr_whr_bmi_strict.csv"

# Union版（推奨：全SNP）
data_path <- "/Users/yoshizawakazuki/Kazuki_module/Genomics/MVMR/mvmr_whr_bmi_strict.csv"

# データ読み込み
mvmr_data <- read.csv(data_path, stringsAsFactors = FALSE)

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
  filter(!is.na(beta_whr) & !is.na(se_whr) &
           !is.na(beta_bmi) & !is.na(se_bmi) &
           !is.na(beta_snore) & !is.na(se_snore)) %>%
  filter(se_whr > 0 & se_bmi > 0 & se_snore > 0)

cat("\nクリーニング後のデータサイズ:", nrow(mvmr_data_clean), "SNPs\n")

# 各曝露で有意なSNP数を確認
cat("\n各曝露で有意なSNP数:\n")
cat("WHR (P < 5e-8):", sum(mvmr_data_clean$pval_whr < 5e-8, na.rm = TRUE), "\n")
cat("BMI (P < 5e-8):", sum(mvmr_data_clean$pval_bmi < 5e-8, na.rm = TRUE), "\n")

# ============================================
# 3. MVMR用のデータフォーマット作成
# ============================================

# 曝露のbetaとSEを行列形式に変換
# 列1: WHR, 列2: BMI
exposure_beta <- cbind(
  mvmr_data_clean$beta_whr,
  mvmr_data_clean$beta_bmi
)

exposure_se <- cbind(
  mvmr_data_clean$se_whr,
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
cat(paste(rep("=", 60), collapse=""), "\n", sep="")

# 条件付きF統計量を計算
# gencov = 0: 曝露間の遺伝的共分散（不明な場合は0）
f_stats <- strength_mvmr(
  r_input = mvmr_input,
  gencov = 0
)

print(f_stats)

cat("\n【解釈】\n")
cat("F統計量 > 10 であれば、弱操作変数バイアスのリスクは低い\n")
if (f_stats[1,1] > 10 & f_stats[1,2] > 10) {
  cat("✓ 両方の曝露で十分な操作変数強度があります\n")
} else {
  cat("⚠ 警告: F統計量が低い曝露があります。結果の解釈に注意が必要です\n")
}

# ============================================
# 6. MVMR解析の実行（IVW法）
# ============================================

cat("\n", paste(rep("=", 60), collapse=""), "\n", sep="")
cat("MVMR解析（逆分散加重法）\n")
cat(paste(rep("=", 60), collapse=""), "\n", sep="")

mvmr_result <- ivw_mvmr(r_input = mvmr_input)

print(mvmr_result)

# 結果を見やすく整形
results_df <- data.frame(
  Exposure = c("WHR (Waist-Hip Ratio)", "BMI"),
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
cat("P値 < 0.05 で統計的に有意\n\n")

# 結果の解釈を追加
for (i in 1:nrow(results_df)) {
  cat(results_df$Exposure[i], ":\n")
  cat("  Beta =", sprintf("%.4f", results_df$Beta[i]), "\n")
  cat("  95% CI = [", sprintf("%.4f", results_df$CI_lower[i]), ",", 
      sprintf("%.4f", results_df$CI_upper[i]), "]\n")
  cat("  P-value =", sprintf("%.2e", results_df$Pvalue[i]), "\n")
  
  if (results_df$Pvalue[i] < 0.05) {
    if (results_df$Beta[i] > 0) {
      cat("  → いびきのリスクを有意に増加させる\n\n")
    } else {
      cat("  → いびきのリスクを有意に減少させる\n\n")
    }
  } else {
    cat("  → 有意な効果は見られない\n\n")
  }
}

# ============================================
# 7. 多面的効果のチェック（Q統計量）
# ============================================

cat(paste(rep("=", 60), collapse=""), "\n", sep="")
cat("多面的効果のテスト（Q統計量）\n")
cat(paste(rep("=", 60), collapse=""), "\n", sep="")

pleiotropy_test <- pleiotropy_mvmr(r_input = mvmr_input)

print(pleiotropy_test)

cat("\n【解釈】\n")
cat("Q統計量のP値 < 0.05 の場合、多面的効果の存在が示唆される\n")
cat("多面的効果がある場合は、結果の解釈に注意が必要\n")

if (!is.null(pleiotropy_test$Qstat) && !is.na(pleiotropy_test$Qpval)) {
  if (pleiotropy_test$Qpval < 0.05) {
    cat("⚠ 多面的効果が検出されました（Q p-value =", 
        sprintf("%.2e", pleiotropy_test$Qpval), ")\n")
  } else {
    cat("✓ 多面的効果の証拠は検出されませんでした\n")
  }
}

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
    title = "MVMR Results: Independent Effects on Snoring",
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

print(p1)

# プロットを保存
ggsave("mvmr_whr_bmi_forest_plot.png", p1, width = 10, height = 6, dpi = 300)
cat("\n森林プロットを保存しました: mvmr_whr_bmi_forest_plot.png\n")

# ============================================
# 9. 散布図（曝露-アウトカム関連）
# ============================================

# WHR vs Snore
scatter_data <- data.frame(
  beta_whr = mvmr_data_clean$beta_whr,
  beta_bmi = mvmr_data_clean$beta_bmi,
  beta_snore = mvmr_data_clean$beta_snore,
  se_snore = mvmr_data_clean$se_snore
)

p2 <- ggplot(scatter_data, aes(x = beta_whr, y = beta_snore)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    title = "WHR - Snoring Association",
    x = "Beta (WHR)",
    y = "Beta (Snoring)"
  ) +
  theme_minimal(base_size = 12)

p3 <- ggplot(scatter_data, aes(x = beta_bmi, y = beta_snore)) +
  geom_point(alpha = 0.3, color = "darkgreen") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(
    title = "BMI - Snoring Association",
    x = "Beta (BMI)",
    y = "Beta (Snoring)"
  ) +
  theme_minimal(base_size = 12)

# 散布図を並べて表示
library(gridExtra)
p_combined <- grid.arrange(p2, p3, ncol = 2)

ggsave("mvmr_whr_bmi_scatter_plots.png", p_combined, width = 14, height = 6, dpi = 300)
cat("散布図を保存しました: mvmr_whr_bmi_scatter_plots.png\n")

# ============================================
# 10. 結果の保存
# ============================================

# 結果をCSVで保存
write.csv(results_df, "mvmr_whr_bmi_results.csv", row.names = FALSE)

# 詳細レポートをテキストファイルで保存
sink("mvmr_whr_bmi_analysis_report.txt")

cat(paste(rep("=", 70), collapse=""), "\n", sep="")
cat("MVMR Analysis Report: WHR + BMI -> Snoring\n")
cat(paste(rep("=", 70), collapse=""), "\n\n", sep="")

cat("Analysis Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Number of SNPs:", nrow(mvmr_data_clean), "\n\n")

cat("Exposures:\n")
cat("  1. WHR (Waist-Hip Ratio) - 中心性肥満の指標\n")
cat("  2. BMI - 全体的な肥満の指標\n\n")

cat("Outcome:\n")
cat("  - Snoring (いびき)\n\n")

cat(paste(rep("=", 70), collapse=""), "\n", sep="")
cat("F-Statistics (Instrument Strength)\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")
print(f_stats)

cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("MVMR Results (IVW)\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")
print(results_df)

cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("Pleiotropy Test\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")
print(pleiotropy_test)

cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("Interpretation\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")

for (i in 1:nrow(results_df)) {
  cat("\n", results_df$Exposure[i], ":\n", sep="")
  cat("  Beta = ", sprintf("%.4f", results_df$Beta[i]), "\n", sep="")
  cat("  95% CI = [", sprintf("%.4f", results_df$CI_lower[i]), ", ", 
      sprintf("%.4f", results_df$CI_upper[i]), "]\n", sep="")
  cat("  P-value = ", sprintf("%.2e", results_df$Pvalue[i]), "\n", sep="")
  
  if (results_df$Pvalue[i] < 0.05) {
    cat("  *** Statistically significant effect\n")
  } else {
    cat("  Not statistically significant\n")
  }
}

cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("Conclusion\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")
cat("\nこの解析は、WHRとBMIの独立した因果効果を推定しています。\n")
cat("両方が有意な場合、それぞれ異なるメカニズムでいびきに影響している可能性があります。\n")
cat("- WHRが有意: 中心性肥満（体脂肪の分布）が重要\n")
cat("- BMIが有意: 全体的な体重が重要\n")

sink()

cat("\n詳細レポートを保存しました: mvmr_whr_bmi_analysis_report.txt\n")

# ============================================
# 11. サマリー
# ============================================

cat("\n", paste(rep("=", 70), collapse=""), "\n", sep="")
cat("解析完了！\n")
cat(paste(rep("=", 70), collapse=""), "\n", sep="")

cat("\n保存されたファイル:\n")
cat("  1. mvmr_whr_bmi_results.csv - 結果のサマリー\n")
cat("  2. mvmr_whr_bmi_forest_plot.png - 森林プロット\n")
cat("  3. mvmr_whr_bmi_scatter_plots.png - 散布図\n")
cat("  4. mvmr_whr_bmi_analysis_report.txt - 詳細レポート\n")

cat("\n主な結果:\n")
print(results_df)