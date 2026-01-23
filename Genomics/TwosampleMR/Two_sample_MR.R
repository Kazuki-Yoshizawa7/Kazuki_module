# Sample MR R script: try running it 


library(TwoSampleMR)
library(ggplot2)

exposure_dat <- read_exposure_data(
  filename ="/Users/yoshizawakazuki/Desktop/Datasets/MVMR/GCST90179150_buildGRCh37.tsv",
  sep =  "\t",
  snp_col = "variant_id",
  beta_col = "beta",
  se_col = "standard_error",
  effect_allele_col = "effect_allele",
  other_allele_col = "other_allele",
  eaf_col = "effect_allele_frequency",
  pval_col = "p_value"
)
library(ieugwasr)

#usethis::edit_r_environ()


#Sys.setenv(OPENGWAS_API_KEY = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImFwaS1qd3QiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhcGkub3Blbmd3YXMuaW8iLCJhdWQiOiJhcGkub3Blbmd3YXMuaW8iLCJzdWIiOiJrYXp1a2lsaWtlc2xlZ29AaG90bWFpbC5jb20iLCJpYXQiOjE3Njc3ODIyMTUsImV4cCI6MTc2ODk5MTgxNX0.ebSUJhkkTjR0iOfyNbh7u6dmIlyyFcFsmr0Z5IQrym56LQI-inQJ5BtyA__GjAt5DUdULY4vsorUrAAMAypKNGl0OP1VyLwv-cLRLzyVJ0EzSRQtGhjHXK7xK1RgYnRWqk-YWtqbcoNvKvhIR0rcfpaansEhfsTMoFmWf9mrAW0YZzhrf1JuUGSdomNt8WiRqKSRncLe3I3TNgqLHda__EeVNhSinP2goGtfvs3Q2GZ-26z3ewOfGcWIbzIAOOeWWQsvvuOfEpk5_32K-qr2ZQdOu5WiT-XQv8HQ52yjqJN26bNqCsY_IEipW1l96Y7XP1JnzKO4iR6seZAT5lFC4A")
Sys.getenv("OPENGWAS_API_KEY")
exposure_dat <- subset(exposure_dat, pval.exposure < 5e-8)

exposure_dat <- clump_data(
  exposure_dat,
  clump_kb = 10000,
  clump_r2 = 0.001,
  pop = "EUR"
)

#outcome_dat <- "/Users/yoshizawakazuki/Desktop/snoring_metal/raw_gwas_stats/Campos_prePMID_Snoring-mainAnalysis.gz"

# アウトカムデータの読み込み
outcome_dat <- read_outcome_data(
  filename = "/Users/yoshizawakazuki/Desktop/snoring_metal/raw_gwas_stats/Campos_prePMID_Snoring-mainAnalysis.gz",
  sep = "\t",  # タブ区切りの場合。スペース区切りなら " " または "\\s+"
  snp_col = "SNP",  # ファイル内のSNP列名に合わせて変更
  beta_col = "BETA",  # ファイル内のベータ列名に合わせて変更
  se_col = "SE",  # 標準誤差列名
  effect_allele_col = "A1",  # 効果アレル列名
  other_allele_col = "A2",  # その他のアレル列名
  eaf_col = "FRQ",  # 効果アレル頻度列名（ない場合は省略可）
  pval_col = "P"  # p値列名
)

# エクスポージャーのSNPのみを抽出
outcome_dat <- outcome_dat[outcome_dat$SNP %in% exposure_dat$SNP, ]

# ハーモナイゼーション
dat <- harmonise_data(
  exposure_dat = exposure_dat,
  outcome_dat = outcome_dat
)

# パリンドロームSNPの処理
dat <- subset(dat, mr_keep == TRUE)

# データの確認
cat("ハーモナイゼーション後のSNP数:", nrow(dat), "\n")
cat("F統計量の計算...\n")

# F統計量の計算（弱い機器変数のチェック）
dat$F_stat <- (dat$beta.exposure^2) / (dat$se.exposure^2)
cat("平均F統計量:", mean(dat$F_stat), "\n")
cat("F < 10 のSNP数:", sum(dat$F_stat < 10), "\n")

# MR分析の実行
mr_results <- mr(dat)
print(mr_results)


# 異質性検定
heterogeneity <- mr_heterogeneity(dat)
print(heterogeneity)

# プレイオトロピー検定
pleiotropy <- mr_pleiotropy_test(dat)
print(pleiotropy)

# Leave-one-out分析
loo <- mr_leaveoneout(dat)

# Single SNP分析
single_snp <- mr_singlesnp(dat)

# 結果の可視化
p1 <- mr_scatter_plot(mr_results, dat)
print(p1[[1]])
ggsave("mr_scatter_snoring.png", p1[[1]], width = 8, height = 6)

p2 <- mr_forest_plot(single_snp)
print(p2[[1]])
ggsave("mr_forest_snoring.png", p2[[1]], width = 8, height = 10)

p3 <- mr_leaveoneout_plot(loo)
print(p3[[1]])
ggsave("mr_leaveoneout_snoring.png", p3[[1]], width = 8, height = 10)

p4 <- mr_funnel_plot(single_snp)
print(p4[[1]])
ggsave("mr_funnel_snoring.png", p4[[1]], width = 8, height = 6)


# 結果の保存
write.csv(mr_results, "mr_results_snoring.csv", row.names = FALSE)
write.csv(heterogeneity, "heterogeneity_snoring.csv", row.names = FALSE)
write.csv(pleiotropy, "pleiotropy_snoring.csv", row.names = FALSE)
write.csv(dat, "harmonised_data_snoring.csv", row.names = FALSE)

# オッズ比の計算
mr_report <- generate_odds_ratios(mr_results)
print(mr_report)
write.csv(mr_report, "mr_odds_ratios_snoring.csv", row.names = FALSE)

# Steiger filtering
steiger <- directionality_test(dat)
print(steiger)
write.csv(steiger, "steiger_snoring.csv", row.names = FALSE)

cat("\n分析完了！\n")


