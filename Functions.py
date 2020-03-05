{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as pyplot\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'function'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-19-bb29b565a718>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mfunction\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[1;33m*\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'function'"
     ]
    }
   ],
   "source": [
    "from function import *\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=pd.read_csv(\"world_data_really_tiny.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>country</th>\n",
       "      <th>lifeexp</th>\n",
       "      <th>unemployment</th>\n",
       "      <th>happiness</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Albania</td>\n",
       "      <td>77.6</td>\n",
       "      <td>6.09</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Bulgaria</td>\n",
       "      <td>75.0</td>\n",
       "      <td>3.24</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Iran</td>\n",
       "      <td>75.8</td>\n",
       "      <td>2.11</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Ukraine</td>\n",
       "      <td>71.9</td>\n",
       "      <td>1.53</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>South Africa</td>\n",
       "      <td>61.8</td>\n",
       "      <td>7.52</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Ukraine</td>\n",
       "      <td>71.9</td>\n",
       "      <td>1.53</td>\n",
       "      <td>Low</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Austria</td>\n",
       "      <td>81.4</td>\n",
       "      <td>1.43</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Croatia</td>\n",
       "      <td>77.3</td>\n",
       "      <td>5.53</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Denmark</td>\n",
       "      <td>80.7</td>\n",
       "      <td>1.36</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Serbia</td>\n",
       "      <td>75.7</td>\n",
       "      <td>4.96</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Indonesia</td>\n",
       "      <td>71.4</td>\n",
       "      <td>1.26</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Thailand</td>\n",
       "      <td>77.5</td>\n",
       "      <td>0.06</td>\n",
       "      <td>High</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         country  lifeexp  unemployment happiness\n",
       "0        Albania     77.6          6.09       Low\n",
       "1       Bulgaria     75.0          3.24       Low\n",
       "2           Iran     75.8          2.11       Low\n",
       "3        Ukraine     71.9          1.53       Low\n",
       "4   South Africa     61.8          7.52       Low\n",
       "5        Ukraine     71.9          1.53       Low\n",
       "6        Austria     81.4          1.43      High\n",
       "7        Croatia     77.3          5.53      High\n",
       "8        Denmark     80.7          1.36      High\n",
       "9         Serbia     75.7          4.96      High\n",
       "10     Indonesia     71.4          1.26      High\n",
       "11      Thailand     77.5          0.06      High"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head(12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12, 4)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>lifeexp</th>\n",
       "      <th>unemployment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>12.000000</td>\n",
       "      <td>12.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>74.833333</td>\n",
       "      <td>3.051667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>5.213328</td>\n",
       "      <td>2.377664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>61.800000</td>\n",
       "      <td>0.060000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>71.900000</td>\n",
       "      <td>1.412500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>75.750000</td>\n",
       "      <td>1.820000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>77.525000</td>\n",
       "      <td>5.102500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>81.400000</td>\n",
       "      <td>7.520000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         lifeexp  unemployment\n",
       "count  12.000000     12.000000\n",
       "mean   74.833333      3.051667\n",
       "std     5.213328      2.377664\n",
       "min    61.800000      0.060000\n",
       "25%    71.900000      1.412500\n",
       "50%    75.750000      1.820000\n",
       "75%    77.525000      5.102500\n",
       "max    81.400000      7.520000"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'matplotlib.pyplot' has no attribute 'histPlotAll'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-30-2f9e240ae31d>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpyplot\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistPlotAll\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m: module 'matplotlib.pyplot' has no attribute 'histPlotAll'"
     ]
    }
   ],
   "source": [
    "pyplot.histPlotAll(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'boxPlotAll' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-14-dcaa8bd90666>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mboxPlotAll\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'boxPlotAll' is not defined"
     ]
    }
   ],
   "source": [
    "boxPlotAll(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'HistPlotAll' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-15-5b012dfcd01a>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m HistPlotAll(dataset\n\u001b[0m\u001b[0;32m      2\u001b[0m            )\n",
      "\u001b[1;31mNameError\u001b[0m: name 'HistPlotAll' is not defined"
     ]
    }
   ],
   "source": [
    "HistPlotAll(dataset\n",
    "           )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "module 'matplotlib.pyplot' has no attribute 'histplotall'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-29-d6ca438f1fa8>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpyplot\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhistplotall\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m: module 'matplotlib.pyplot' has no attribute 'histplotall'"
     ]
    }
   ],
   "source": [
    "pyplot.histplotall(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'classComparePlot' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-17-b98133e53433>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mclassComparePlot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"happiness\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m\"lifeexp\"\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m\"unemployment\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'happiness'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mplotType\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'hist'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'classComparePlot' is not defined"
     ]
    }
   ],
   "source": [
    "classComparePlot(dataset[[\"happiness\",\"lifeexp\",\"unemployment\"]],'happiness',plotType='hist')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'Histogram' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-21-858726eb0d97>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mHistogram\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'Histogram' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "Histogram(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "77.6 is not a string",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-33-36d44cd2806e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mpyplot\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataset\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\pyplot.py\u001b[0m in \u001b[0;36mhist\u001b[1;34m(x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, normed, data, **kwargs)\u001b[0m\n\u001b[0;32m   2634\u001b[0m         \u001b[0malign\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0malign\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morientation\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morientation\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrwidth\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mrwidth\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlog\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlog\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2635\u001b[0m         \u001b[0mcolor\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcolor\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mlabel\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mlabel\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mstacked\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mstacked\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnormed\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mnormed\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2636\u001b[1;33m         **({\"data\": data} if data is not None else {}), **kwargs)\n\u001b[0m\u001b[0;32m   2637\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2638\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\__init__.py\u001b[0m in \u001b[0;36minner\u001b[1;34m(ax, data, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1587\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0minner\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0max\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1588\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mdata\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1589\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0max\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0mmap\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msanitize_sequence\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1590\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1591\u001b[0m         \u001b[0mbound\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnew_sig\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mbind\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0max\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\axes\\_axes.py\u001b[0m in \u001b[0;36mhist\u001b[1;34m(self, x, bins, range, density, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, stacked, normed, **kwargs)\u001b[0m\n\u001b[0;32m   6677\u001b[0m         \u001b[1;31m# Process unit information\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   6678\u001b[0m         \u001b[1;31m# Unit conversion is done individually on each dataset\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 6679\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_process_unit_info\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mxdata\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   6680\u001b[0m         \u001b[0mx\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m[\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconvert_xunits\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mxi\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mxi\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mx\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   6681\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\axes\\_base.py\u001b[0m in \u001b[0;36m_process_unit_info\u001b[1;34m(self, xdata, ydata, kwargs)\u001b[0m\n\u001b[0;32m   2121\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2122\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2123\u001b[1;33m         \u001b[0mkwargs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_process_single_axis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mxdata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mxaxis\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'xunits'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2124\u001b[0m         \u001b[0mkwargs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_process_single_axis\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mydata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0myaxis\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'yunits'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2125\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\axes\\_base.py\u001b[0m in \u001b[0;36m_process_single_axis\u001b[1;34m(data, axis, unit_name, kwargs)\u001b[0m\n\u001b[0;32m   2104\u001b[0m                 \u001b[1;31m# We only need to update if there is nothing set yet.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2105\u001b[0m                 \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mhave_units\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 2106\u001b[1;33m                     \u001b[0maxis\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate_units\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   2107\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   2108\u001b[0m             \u001b[1;31m# Check for units in the kwargs, and if present update axis\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\axis.py\u001b[0m in \u001b[0;36mupdate_units\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m   1492\u001b[0m         \u001b[0mneednew\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconverter\u001b[0m \u001b[1;33m!=\u001b[0m \u001b[0mconverter\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1493\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconverter\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mconverter\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1494\u001b[1;33m         \u001b[0mdefault\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mconverter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdefault_units\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1495\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mdefault\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munits\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1496\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_units\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdefault\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\category.py\u001b[0m in \u001b[0;36mdefault_units\u001b[1;34m(data, axis)\u001b[0m\n\u001b[0;32m    113\u001b[0m         \u001b[1;31m# default_units->axis_info->convert\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    114\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munits\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 115\u001b[1;33m             \u001b[0maxis\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_units\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mUnitData\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    116\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    117\u001b[0m             \u001b[0maxis\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munits\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\category.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m    179\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_counter\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mitertools\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcount\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    180\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mdata\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 181\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    182\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    183\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0mstaticmethod\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\matplotlib\\category.py\u001b[0m in \u001b[0;36mupdate\u001b[1;34m(self, data)\u001b[0m\n\u001b[0;32m    216\u001b[0m             \u001b[1;31m# OrderedDict just iterates over unique values in data.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    217\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mval\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbytes\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 218\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"{val!r} is not a string\"\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mformat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mval\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mval\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    219\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mconvertible\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    220\u001b[0m                 \u001b[1;31m# this will only be called so long as convertible is True.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: 77.6 is not a string"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pyplot.hist(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
