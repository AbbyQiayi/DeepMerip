#remotes::install_github("ZW-xjtlu/meripDeep")
#Example
library(meripDeep)
library(GenomicFeatures)
library(BSgenome.Hsapiens.UCSC.hg38)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)

GENE_ANNO_GTF = system.file("extdata", "example.gtf", package="exomePeak2")
txdb = makeTxDbFromGFF(GENE_ANNO_GTF)


# Define the cell lines
cell_lines = c("mm")
down = c('.01','.02','.03','.1','.2','.3')

for (cell_line in cell_lines) {
  for (down in down){
    setwd(paste0("/home/qiayi/project/coda/Deep_merip/data/bam/bam_mouse/mm_",down))
    # Update file names based on the current cell line
    f1_IP =  paste0(cell_line, "_IP_1_",down,"down.bam")
    f2_IP = paste0(cell_line, "_IP_2_",down,"down.bam")
    IP_BAM = c(f1_IP, f2_IP)
    
    f1_INPUT = paste0(cell_line, "_INPUT_1_",down,"down.bam")
    f2_INPUT = paste0(cell_line, "_INPUT_2_",down,"down.bam")
    INPUT_BAM = c(f1_INPUT, f2_INPUT)
    
    # Perform operations for each cell line
    # Example: Peak calling
    Peaks <- peakCalling(bam_IP = IP_BAM,
                         bam_input = INPUT_BAM,
                         txdb = TxDb.Hsapiens.UCSC.hg38.knownGene,
                         genome = BSgenome.Hsapiens.UCSC.hg38)
    
    # Save results for each cell line
    saveRDS(Peaks, file = paste0("/home/qiayi/project/coda/Deep_merip/data/exomePeak2_out/peak_", cell_line, "_IP_vs_INPUT_",down,".rds"))
    write.csv(as.data.frame(mcols(Peaks)), paste0("/home/qiayi/project/coda/Deep_merip/data/exomePeak2_out/peak_", cell_line, "_IP_vs_INPUT_",down,".csv"))
  }
}



#Peak Calling with GFF
# library(BSgenome.Hsapiens.UCSC.hg38)
# Peaks <- peakCalling(bam_IP = IP_BAM,
#                      bam_input = INPUT_BAM,
#                      txdb = txdb,
#                      genome = BSgenome.Hsapiens.UCSC.hg38)

# 
# length(Peaks)
# peaks <- mcols(Peaks)
# peaks <- as.data.frame(peaks)
# peaks <- unlist(Peaks_mm)
# Range <- as.data.frame(unlist(Peaks_mm))
# table(Range_mm$strand)


#Access to metadata columns
# saveRDS(Peaks, file = "/home/qiayi/project/coda/Deep_merip/data/exomePeak2_out/peak_NEB_IP_vs_INPUT_0.3dn.rds")

#Plot transcript topology
#list_grl should be a list of GRanges / GRangesList
#names of the list will be the label
# plotTopology(list_grl = Peaks, txdb = txdb, savePrefix = "topology")

# write.csv(peaks,"/home/qiayi/project/coda/Deep_merip/data/exomePeak2_out/peak_NEB_IP_vs_INPUT_0.3dn.csv")


