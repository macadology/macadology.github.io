---
title: "Taxonomic Classification - Kraken"
date: 2020-07-23
year: 2020
monthday: 07-23

categories:
  - metagenomics
tags:
  - kraken
  - taxonomic classification
header:
  image: /assets/images/2020/07-23/octopus.jpg
  teaser: /assets/images/2020/07-23/octopus.jpg
  caption: ""

toc: true
toc_sticky: true
---
## [MetaPhlAn2](http://huttenhower.sph.harvard.edu/metaphlan)
"(**Met**agenomic **Phyl**ogenetic **An**alysis) **MetaPhlAn** is a computational tool for profiling the composition of microbial communities from metagenomic shotgun sequencing data. MetaPhlAn relies on unique clade-specific marker genes identified from 3,000 reference genomes...

The generation of this catalog of marker genes (**marker catalog**) uses an intraclade CDS clustering and then an extraclade sequence uniqueness assessment; the method was based loosely on our previous system for detecting core genes.It is an offline procedure that we perform regularly as a relevant set of newly sequenced microbial genomes is available, and the catalog is downloaded automatically with the associated classifier.

The MetaPhlAn classifier compares metagenomic reads against
this precomputed marker catalog **using nucleotide BLAST searches** in order to provide clade abundances for one or more sequenced metagenomes.

...the MetaPhlAn classifier **normalizes** the total number of reads in each clade by the nucleo- tide length of its markers
