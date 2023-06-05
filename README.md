
# FAL-CUR: Fair Active Learning using Uncertainty and Representativeness on Fair Clustering

FAL-CUR is an active learning method that uses the uncertainty and representativeness on Fair Clustering. The method implement fair clustering (Abraham et al 2020) to achieve fairness in cluster level and select samples using uncertainty and represetative score. We show that FAL-CUR able to preserve fairness metric (Statistical Parity, Equal Opportunity and Equalized odds) while maintaining the model performance stable

# Getting Started

These instructions will help you to get a copy of FAL-CUR up and running on your local machine for development and testing purposes.

# Prerequisites

Before you begin, ensure you have met the following requirements:

Python version 3.7 or later <br>
Numpy version 1.18 or later <br>
Scipy version 1.4 or later <br>
Sklearn version 0.22 or later <br>

# Installation
1. Clone the FAL-CUR library 
```
git clone https://github.com/rickymaulanafajri/FAL-CUR
```
2. Navigate to the FAL-CUR 
```
cd FAL-CUR
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. To use FAL-CUR, please follow the example provided in the FAL-CUR.ipynb file.

# Citation
If you use the FAL-CUR method in your research, please consider citing our work. The citation details are as follows:
```
@article{FAL-CUR2023,
    title={FAL-CUR: Fair Active Learning using Uncertainty and Representativeness on Fair Clustering},
    author={Ricky Maulana Fajri, Akrati Saxena, Yulong Pei, Mykola Pechenizkiy},
    journal={ArXiv},
    volume={xx},
    number={yy},
    pages={zz-zz},
    year={2023}
}
```
