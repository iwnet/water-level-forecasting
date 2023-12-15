Water level forecasting using Neural Networks
================================



This repository contains the source code for our work: **IWNET BDA: A Big Data Infrastructure for Predictive and Geotemporal Analytics of Inland Waterways**.

Our investigation involves the evaluation of multi-day ahead forecasts, as well as daily predictions. By integrating weather data into our predictive models, we aim to uncover patterns and trends that contribute to more nuanced and precise predictions. Although employed in the current setting as a proof-of-concept, the main principles of our approach can be generalized and utilized in any similar scenario.

Abstract
-------
The digitalization of existing traditional businesses such as logistics and transportation has offered opportunities for valuable insights and optimizations. This produces an enormous digital footprint that consists of fleet IoT data and business transactions that, combined with external publicly available information, can extract useful knowledge. Nevertheless, the process of collecting, storing, analyzing and processing this information requires both large-scale computing infrastructures and sophisticated scalable software. In this work, we present IWNET-BDA, an open source big-data framework that consumes data from European Inland  Waterway Transport Corridors. We describe its architectural components and computing infrastructure. We finally present two different use cases which IWNET-BDA was used to solve: a) a business problem of automatically locating areas of interest and tracking vessel activity to provide high level insights using geo-temporal data analytics and b) the generation of water level forecasts using recurrent neural networks in the Danube River.

Requirements
------------
You can install the requirements using the script 
```
pip3 install requirements.txt
```

Project structure
---------------
| File                             | comments    | |
|--------------------------------------|------------------------------|---------------------------|
| `main.py` | The main file where you can specify model configurations for training.   ||
| `datasets/` | Contains the dataset files, both for water levels and weather.
|  `models.py`| Contains various statistical and ML models used for training. 
|  `data_loading.py`| Loads the datasets into pandas DataFrames.    ||
|  other | The rest are tools for plotting results and various utility functions.     || 


Contact
-------
avontz@cslab.ece.ntua.gr 