# PhageAnnotate

Bacteriophages (phages) are the most abundant life form in the planet. They are bacterial viruses that recognize, infect and kill their hosts to propagate. As phages are very different from each other and from bacteria, and we have relatively few phage genes in our database compared to bacterial genes, we are unable to assign function to 50-90% of phage genes.

This project intends to use machine learning approaches to classify a phage gene according to their function (structural, replication or lysis) or into an "other" category. This approach does not require a similar gene to be known. The main goal of this project is:
1) To review literature related to annotation tools applied to phages genomes;
2) To build a dataset of phage protein sequences for a given function;
3) To start exploring machine learning approaches to improve the accuracy of predictions (i.e., classify proteins into one of the categories or, if not as "other", providing approach for functional annotation of phage proteins).

The project will establish the foundation by creating a powerful tool to classify phage proteins when homology-based alignments do not provide useful functional predictions, such as "hypothetical" or "unknown function".
