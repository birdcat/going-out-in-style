# going-out-in-style
Style transfer in three modalities.

This repo contains one third (the text component) of a three-modality project on unsupervised style transfer.

## Image
Available at [add Jacob's repo?]

## Audio
Available at https://github.com/jenchen1398/artistic-music-style-transfer/tree/master/pytorch.

## Text
The method for 'style transfer' implemented in this repo is actually something closer to 'style alignment' - it involves determining an unsupervised alignment between word embedding spaces for two different styles, then performing a mapping on text to transform the one into the other. Currently nothing fancier than just translating word by word, which is obviously not a proper translation, but that's where it's at.

[add links for datasets]

## References
[1] David Alvarez-Melis and Tommi Jaakkola. Gromov-Wasserstein Alignment of Word Embedding Spaces. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1881–1890. Association for Computational Linguistics, 2018. URL http://aclweb.org/anthology/D18-1214.