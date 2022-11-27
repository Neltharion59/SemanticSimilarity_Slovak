# Semantic textual similarity in Slovak

This GitHub contains source code used to produce data for a paper not yet published. Link to the paper will be added here once it is published. Note that this project is supposed to work with Slovak language texts.

In case of any questions, do not hesitate to contact the authors:
- Lukáš Radoský - lukas.radosky59@gmail.com, lukas.radosky@kinit.sk, lukas.radosky@fmph.uniba.sk
- Miroslav Blšták - miroslav.blstak@kinit.sk

The GitHub consists of two parts:
- **Optimizing** - contains the source code used to produce data for the paper, including dataset preprocessing, unsupervised method calculations, artificial beehive optimization of machine learning models using unsupervised methods as features and statistical calculations of results.
- **Ready-to-use** - contains modified subpart of the above described source code. It offers pretrained models ready for use in your project. The root of this folder demonstrates example usage. Model names specify optimization run (*1st* vs *2nd*), dataset version (*raw* vs *lemmatized*) and dataset name. We recommend using models from *2nd* optimization run. Unless you provide lemmatized input, we suggest using *raw* models. Models trained on *sick* or *all* dataset should cover your needs the best, as they were trained on the largest sample of data. You may also try to compare all existing models to find the best one for you.

Below, we display table describing all provided models and their configurations. The table was unnecessarily large for the paper.

