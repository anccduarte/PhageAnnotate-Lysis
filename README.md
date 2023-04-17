# PhageAnnotate - Lysis

Bacteriophages (phages) are the most abundant life form in the planet. They are bacterial viruses that recognize, infect and kill their hosts to propagate. As phages are very different from each other and from bacteria, and we have relatively few phage genes in our database compared to bacterial genes, we are unable to assign function to 50-90% of phage genes.

This project intends to use machine learning approaches to classify phage genes according to the function of the respective protein product. Phage gene products may be grouped into one of four functional classes: structural, lysis, DNA modification and replication, and packaging. The present work solely focuses on lysis proteins, establishing the foundation for a more powerful tool capable of classifying proteins into any of the previously mentioned functional classes. Here, gene products will be classified into one of the following classes: _endolysin_, _spanin_, _holin_ (lysis proteins) or _other_ (if the gene product does not belong to any of the first three classes). This approach does not require a similar gene to be known.

The main goals of this project are:
1) To review literature related to annotation tools applied to phage genomes;
2) To build a dataset of phage sequences coding for lysis proteins (positives) and non-lysis proteins (negatives);
3) To explore machine learning approaches capable of improving the accuracy of the predictions made by current annotation tools.
