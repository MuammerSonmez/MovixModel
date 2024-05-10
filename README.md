# Movix Model

## About
This model predicts the price of a bicycle based on bicycle specifications
## Installation
Use the package manager pip to install Tensorflow, Numpy, Pandas, Seaborn, Scikit-Learn.
```bash
pip install tensorflow
pip install numpy
pip install pandas
pip install seaborn
pip install scikit-learn
```
If use the anaconda to install Tensorflow, Numpy, Pandas, Seaborn, Scikit-Learn.
```bash
conda install tensorflow
conda install numpy
conda install pandas
conda install seaborn
conda install scikit-learn
```
## Example
```python
yeniBisikletOzellikleri = [[1760,1757]]
yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)
model.predict(yeniBisikletOzellikleri)
```