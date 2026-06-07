# HELP International Clustering Submission

## Objective
Identify countries that should be prioritized for financial aid by clustering countries with similar socio-economic conditions.

## Dataset
- File used: `Country-data.csv`
- Total countries: 167
- Total columns: 10

## Approach
1. Loaded the dataset and cleaned column names.
2. Removed duplicate rows.
3. Converted all non-country columns to numeric format.
4. Filled missing values with median values.
5. Selected `income` and `gdpp` as the main clustering features.
6. Scaled features using `RobustScaler`.
7. Used PCA for 2D visualization.
8. Evaluated K-Means from `k=2` to `k=10` using silhouette score.
9. Compared with DBSCAN as a secondary clustering method.
10. Chose the cluster with the lowest average income as the priority-aid cluster.

## Model Result
- Best K-Means clusters: `2`
- Silhouette score: `0.7464`
- DBSCAN result: most countries fell into one main cluster, so K-Means was more useful for final segmentation.

## Cluster Insight
- Cluster `0` represents lower-income and lower-GDP countries.
- Cluster `1` represents higher-income and higher-GDP countries.
- The aid recommendation list was taken from Cluster `0`, sorted by lowest income first.

## Recommended Countries for HELP International

| Rank | Country | Income | GDP per Capita | Child Mortality | Life Expectancy |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Congo, Dem. Rep. | 609 | 334 | 116.0 | 57.5 |
| 2 | Liberia | 700 | 327 | 89.3 | 60.8 |
| 3 | Burundi | 764 | 231 | 93.6 | 57.7 |
| 4 | Niger | 814 | 348 | 123.0 | 58.8 |
| 5 | Central African Republic | 888 | 446 | 149.0 | 47.5 |
| 6 | Mozambique | 918 | 419 | 101.0 | 54.5 |
| 7 | Malawi | 1030 | 459 | 90.5 | 53.1 |
| 8 | Guinea | 1190 | 648 | 109.0 | 58.0 |
| 9 | Togo | 1210 | 488 | 90.3 | 58.7 |
| 10 | Sierra Leone | 1220 | 399 | 160.0 | 55.0 |
| 11 | Rwanda | 1350 | 563 | 63.6 | 64.6 |
| 12 | Madagascar | 1390 | 413 | 62.2 | 60.8 |
| 13 | Guinea-Bissau | 1390 | 547 | 114.0 | 55.6 |
| 14 | Comoros | 1410 | 769 | 88.2 | 65.9 |
| 15 | Eritrea | 1420 | 482 | 55.2 | 61.7 |
| 16 | Burkina Faso | 1430 | 575 | 116.0 | 57.9 |
| 17 | Haiti | 1500 | 662 | 208.0 | 32.1 |
| 18 | Uganda | 1540 | 595 | 81.0 | 56.8 |
| 19 | Afghanistan | 1610 | 553 | 90.2 | 56.2 |
| 20 | Gambia | 1660 | 562 | 80.3 | 65.5 |

## Conclusion
The clustering analysis clearly separates economically weaker countries from stronger ones. Based on low income and low GDP per capita, the listed countries should be prioritized by HELP International for aid consideration.
