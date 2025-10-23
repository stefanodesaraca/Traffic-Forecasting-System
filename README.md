# Traffic-Forecasting-System
**Machine learning algorithms for traffic data forecasting and route-finding on the Norwegian road network**  
Forecast traffic at TRPs, build road-network graphs, and compute traffic-aware routes.

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![python](https://img.shields.io/badge/python-3.11%2B-green.svg)](https://www.python.org/)

----

## Overview

Traffic-Forecasting-System is an open-source system that:
- Reconstructs the Norwegian road network as a graph (nodes = intersections, edges = road links) using NVDB and Trafikkdata API data sources.
- Trains ML models (decision-tree and histogram-gradient-boosting regressors) to forecast traffic volume and average speed at traffic registration points (TRPs).
- Uses those forecasts, plus spatial interpolation (Ordinary Kriging), to evaluate traffic on road segments and produce up to N traffic-aware routes between two points.

### Key features
- Ingest Trafikkdata (GraphQL) and NVDB spatial data.
- Modular project-per-database architecture (isolated projects).
- Data cleaning with the additional implementation of MICE and zero-adjusted gamma regression to generate zero data when there is no traffic.
- Distributed training & hyperparameter tuning with Dask (GridSearchCV).
- Route-finding using A* and iterative path diversification, with weights accounting for length, lanes, speed limits and predicted traffic.
- Spatial interpolation of TRP predictions using PyKrige (Ordinary Kriging).

---

### Requirements
- Python 3.11+  
- PostgreSQL with PostGIS extension
- Dask
- Dask Distributed (for distributed computations)
- Dask-ML  
- Access to Trafikkdata API (public GraphQL endpoint) and NVDB data (public).
