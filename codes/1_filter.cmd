## $1 --- prefix of the original PLINK files 
##Delete individuals with missingness>0.1
$plink --bfile "$1" --mind 0.1 --make-bed --out "$1".1
##Delete snp with missingness>0.1
$plink --bfile "$1".1 --geno 0.1 --make-bed --out "$1".2
##Check for sex discrepancy
$plink --bfile "$1".2 --check-sex
##impute-sex
$plink --bfile "$1".2 --impute-sex --make-bed --out "$1".3
##Remove snp with low MAF
$plink --bfile "$1".3 --maf 0.01 --make-bed --out "$1".4
##Check the distribution of HWE p-values for all SNPs
##First, use a stringent HWE threshold for controls
$plink --bfile "$1".4 --hwe 1e-6 --make-bed --out "$1".step1
##Second, use a less stringent threshold for the case
$plink --bfile "$1".step1 --hwe 1e-10 --hwe-all --make-bed --out "$1".clean
##Remove duplicate SNPs
$plink --noweb --bfile "$1".clean --list-duplicate-vars
cut -f4 plink.dupvar | cut -f1 -d" " > Duplicates.list
$plink --noweb --bfile "$1".clean --exclude Duplicates.list --make-bed --out "$1".rmdup
